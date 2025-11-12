import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeTalkerTextMLP,
    Qwen3OmniMoeTalkerTextSparseMoeBlock
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.attention import AttentionType
from vllm.attention.backends.registry import _Backend
from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2_audio import Qwen2AudioProcessingInfo
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
)
# from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3MoeLLMForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)

from vllm.distributed.eplb.eplb_state import EplbState

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.fused_moe import FusedMoE

from .qwen3_omni_moe_thinker import (
    Qwen3OmniMoeConditionalGenerationMixin,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
    Qwen3MoeLLMForCausalLM,
    Qwen3Omni_VisionTransformer
)

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)

from vllm_omni.model_executor.layers.fused_moe.layer import (
    FusedMoE,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)

Qwen3OmniMoeThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder

@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    """
    Qwen3 Omni MoE Talker - Converts text to audio codec codes.
    
    The talker is the second stage of Qwen3 Omni MoE's TTS pipeline:
    1. Thinker: Generates text response + hidden states
    2. Talker: Converts those to 8-layer audio codec codes
    3. Code2Wav: Converts codes to waveform
    
    ## Usage Example:
    
    ```python
    # Step 1: Run thinker to get text and hidden states
    thinker_outputs = thinker.generate(...)
    thinker_embeds = thinker.get_input_embeddings()(thinker_outputs.token_ids)
    thinker_hidden = thinker_outputs.hidden_states  # From last layer
    
    # Step 2: Project thinker outputs to talker dimension
    talker_inputs = talker.project_thinker_outputs(
        thinker_embeds=thinker_embeds,          # Text embeddings
        thinker_hidden_states=thinker_hidden,   # For multimodal regions
        is_multimodal_mask=multimodal_mask,      # Which positions are multimodal
    )
    
    # Step 3: Generate layer 0 audio codes
    hidden = talker.forward(
        input_ids=None,
        positions=positions,
        inputs_embeds=talker_inputs,
    )
    logits = talker.compute_logits(hidden)
    layer0_codes = sample(logits)
    
    # Step 4: Generate layers 1-num_layers-1 with code predictor
    residual_codes = talker.generate_residual_codes_for_token(
        layer0_code=layer0_codes[:, -1:],
        past_hidden=hidden[:, -1:, :],
    )
    
    # Step 5: Combine into full 8-layer codes
    full_codes = torch.cat([layer0_codes, residual_codes], dim=1)
    
    # Step 6: Convert to audio (using code2wav model)
    # audio_waveform = code2wav(full_codes)
    ```
    
    ## Key Components:
    - text_projection: Projects thinker text embeddings → talker dimension
    - hidden_projection: Projects thinker hidden states → talker dimension  
    - language_model: Main MoE transformer (generates layer 0)
    - codec_head: Projects to codec vocabulary (layer 0 logits)
    - code_predictor: Small transformer for layers 1-num_layers-1
    """
    logger = init_logger(__name__)
    
    # Weight mapping from HuggingFace to vLLM naming convention
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Main MoE transformer model
            "talker.model.": "language_model.model.",
            # Codec head remains separate (outputs audio codes, not text)
            "talker.codec_head.": "codec_head.",
            # Code predictor: Now matches HF structure exactly (has .model sub-module)
            # e.g., "talker.code_predictor.model.codec_embedding.0" → "code_predictor.model.codec_embedding.0"
            "talker.code_predictor.": "code_predictor.",
            # Projection layers
            "talker.text_projection.": "text_projection.",
            "talker.hidden_projection.": "hidden_projection.",
            # Fallback: strip talker prefix
            "talker.": "",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        talker_config: Qwen3OmniMoeTalkerConfig = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_config
        self.vocab_size = talker_config.text_config.vocab_size
        self.router_aux_loss_coef = talker_config.text_config.router_aux_loss_coef
        self.num_experts = talker_config.text_config.num_experts
        self.num_experts_per_tok = talker_config.text_config.num_experts_per_tok
        # thinker projection components for talker
        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(self.config)
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(self.config)
        self.codec_head = nn.Linear(self.config.text_config.hidden_size, self.config.text_config.vocab_size, bias=False)

        self.rope_deltas = None
        self.spatial_merge_size = self.config.spatial_merge_size

        self.language_model = Qwen3OmniMoeModel(vllm_config=vllm_config,
                                                talker_config=self.config,
                                                prefix=maybe_prefix(prefix, "language_model"),)

    def init_multi_modal(self, thinker_config):
        """
        Initialize multimodal components from the thinker.
        
        Unlike Qwen2.5 Omni which creates audio_tower and visual encoders here,
        Qwen3 Omni MoE has a cleaner separation: the thinker is the ONLY module
        that processes raw multimodal inputs. The talker only handles text-to-audio
        conversion using pre-processed embeddings from the thinker.
        
        This method exists for API compatibility and stores the thinker config
        for reference. The actual multimodal processing components (audio_tower,
        visual) are ONLY in the thinker, not duplicated in the talker.
        
        Args:
            thinker_config: Configuration from the thinker model (for reference only)
        """
        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)
        self.visual = Qwen3Omni_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=self.quant_config,
            prefix=maybe_prefix(self.prefix, "visual"),
            # attn_backend_override=attn_backend_override,
        )

    def project_thinker_outputs(
        self,
        thinker_embeds: torch.Tensor | None = None,
        thinker_hidden_states: torch.Tensor | None = None,
        is_multimodal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Project thinker outputs to talker's hidden dimension.
        
        The talker has a different hidden size than the thinker, so we need
        to project the inputs appropriately:
        - Text embeddings (from thinker's embedding layer) → text_projection
        - Hidden states (from thinker's last layer, for multimodal) → hidden_projection
        
        Args:
            thinker_embeds: Text embeddings from thinker [batch, seq, thinker_hidden]
            thinker_hidden_states: Hidden states from thinker's last layer [batch, seq, thinker_hidden]
            is_multimodal_mask: Boolean mask indicating multimodal positions [batch, seq]
        
        Returns:
            projected_embeds: [batch, seq, talker_hidden]
        """
        if thinker_embeds is None and thinker_hidden_states is None:
            raise ValueError("Either thinker_embeds or thinker_hidden_states must be provided")
        
        # If only embeddings provided, project all as text
        if thinker_hidden_states is None or is_multimodal_mask is None:
            return self.text_projection(thinker_embeds)
        
        # If only hidden states provided, project all as hidden
        if thinker_embeds is None:
            return self.hidden_projection(thinker_hidden_states)
        
        # Mixed case: use mask to decide which projection
        batch_size, seq_len, _ = thinker_embeds.shape
        output = torch.empty(
            (batch_size, seq_len, self.config.text_config.hidden_size),
            device=thinker_embeds.device,
            dtype=thinker_embeds.dtype,
        )
        
        # Project multimodal regions using hidden states
        if is_multimodal_mask.any():
            mm_hidden = thinker_hidden_states[is_multimodal_mask]
            projected_mm = self.hidden_projection(mm_hidden)
            output[is_multimodal_mask] = projected_mm
        
        # Project text regions using embeddings
        if (~is_multimodal_mask).any():
            text_embeds = thinker_embeds[~is_multimodal_mask]
            projected_text = self.text_projection(text_embeds)
            output[~is_multimodal_mask] = projected_text
        
        return output
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass through the talker model.
        
        For inference, the talker receives inputs_embeds that should already be
        projected to talker's hidden dimension. If receiving raw thinker outputs,
        use project_thinker_outputs() first.
        """
        # If intermediate_tensors is provided (pipeline parallel), 
        # inputs_embeds should be None
        if intermediate_tensors is not None:
            inputs_embeds = None

        # for profiling
        if inputs_embeds.shape[-1] == 2048:
            inputs_embeds = self.text_projection(inputs_embeds)

        # Pass through the main language model (MoE transformer)
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata = None,
    ) -> torch.Tensor | None:
        """Compute logits for audio codec codes (layer 0 of RVQ).
        
        This projects the hidden states to the codec vocabulary space.
        For full audio generation, layers except 0 would be predicted by
        the code_predictor after sampling.
        """
        logits = self.codec_head(hidden_states)
        return logits
    
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        """Create empty intermediate tensors for pipeline parallelism."""
        return self.language_model.make_empty_intermediate_tensors(
            batch_size, dtype, device
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                    input_key in ("pixel_values", "image_embeds")
                    and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                    input_key in ("pixel_values_videos", "video_embeds")
                    and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if (
                    input_key in ("input_audio_features")
                    and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_multimodal_embeddings(
            self, **kwargs: object
    ) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        # TODO: do projection for all multimodel
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                image_embeddings = self.hidden_projection(image_embeddings)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                video_video_embeddings_project = ()
                for video_embed in video_embeddings:
                    proj = nn.Linear(8192, 2048).to(device=video_embed.device,  dtype=torch.bfloat16)
                    video_embed = proj(video_embed)
                    video_embed_project = self.hidden_projection(video_embed)
                    video_video_embeddings_project += (video_embed_project, )
                multimodal_embeddings += tuple(video_video_embeddings_project)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                audio_embeddings = self.hidden_projection(audio_embeddings)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def get_input_embeddings(self):
        """Get the input embedding layer (for codec tokens)."""
        return self.language_model.get_input_embeddings()
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the talker model.
        
        The weight mapping translates from HuggingFace naming convention
        to vLLM's internal structure. Code predictor weights are routed
        to its custom loader for vocab extension support.
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "code2wav.", "code_predictor."],
        )
        # Don't apply mapper again since we already did it
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__, True, total_bytes / (1024**2), str(device)
            )
        except Exception:
            pass

        multi_model_weights = set()
        for name, param in self.visual.named_parameters():
            multi_model_weights.add("visual." + name)
        for name, param in self.audio_tower.named_parameters():
            multi_model_weights.add("audio_tower." + name)
        loaded.update(multi_model_weights)

        return loaded


class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    """
    MLP for projecting between thinker and talker hidden dimensions.
    
    The thinker and talker have different hidden sizes:
    - Thinker: config.thinker_hidden_size (e.g., 3584)
    - Talker: config.text_config.hidden_size (e.g., 2048)
    
    This MLP projects from thinker → talker dimension.
    Two instances are used:
    - text_projection: For text embeddings from thinker's embedding layer
    - hidden_projection: For hidden states from thinker's last transformer layer
    """
    
    def __init__(self, config: Qwen3OmniMoeTalkerConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.thinker_hidden_size, config.text_config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.text_config.intermediate_size, config.text_config.hidden_size, bias=True)
        self.act_fn = _ACTIVATION_REGISTRY[config.text_config.hidden_act]  # silu

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

class Qwen3OmniMoeModel(Qwen3MoeLLMForCausalLM):
    def __init__(self,
                 vllm_config,
                 talker_config,
                 prefix):
        super().__init__(vllm_config=vllm_config.with_hf_config(
                talker_config.text_config, architectures=["Qwen3MoeForCausalLM"]
            ),
            prefix=prefix,)
        
        self.config = talker_config

        # Remove the inherited LM head so the talker only exposes codec outputs.
        if hasattr(self, "lm_head"):
            del self.lm_head

        # Replace the base embed tokens with codec embedding (defined below).
        if hasattr(self.model, "embed_tokens"):
            del self.model.embed_tokens

        # Codec embedding for RVQ code generation
        self.model.codec_embedding = nn.Embedding(
            talker_config.text_config.vocab_size,
            talker_config.text_config.hidden_size
        )
        # Alias embed_tokens to codec_embedding for compatibility with helpers
        # self.model.embed_tokens = self.model.codec_embedding
        self.n_redundant_experts = vllm_config.parallel_config.num_redundant_experts
        # print(f"===============talker config : {talker_config}")
        index = 0
        for layer in self.model.layers:
            shared_expert_num = 1
            index += 1
            layer.mlp.shared_expert = FusedMoE(num_experts=shared_expert_num,
                                               top_k=self.config.text_config.num_experts_per_tok,
                                               hidden_size=self.config.text_config.hidden_size,
                                               intermediate_size=self.config.text_config.shared_expert_intermediate_size,
                                               reduce_results=False,
                                               renormalize=self.config.text_config.norm_topk_prob,
                                               quant_config=vllm_config.quant_config,
                                               prefix=f"{prefix}.layers.{index}.mlp.shared_expert",
                                               enable_eplb=False,
                                               num_redundant_experts=self.n_redundant_experts)
            layer.mlp.shared_expert_gate = ReplicatedLinear(self.config.text_config.hidden_size,
                                                            shared_expert_num,
                                                            bias=False,
                                                            quant_config=None,
                                                            prefix=f"{prefix}.layers.{index}.mlp.shared_expert_gate")

    def get_input_embeddings(self,
                             input_ids: torch.Tensor,
                             multimodal_embeddings: MultiModalEmbeddings | None = None,
                             generation_steps=None,
                             ) -> torch.Tensor:
        return self.model.codec_embedding(input_ids)
