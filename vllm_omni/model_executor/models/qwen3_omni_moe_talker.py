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


class Qwen3OmniMoeTalkerCodePredictorRotaryEmbedding(nn.Module):
    """Rotary positional embeddings for the code predictor."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position = position_ids[:, None, :].float()
        freqs = (inv_freq @ position).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep) for KV caching.
    Converts hidden states from (batch, num_kv_heads, seq_len, head_dim) to
    (batch, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen3OmniMoeTalkerCodePredictorAttention(Attention):
    """
    Multi-head self-attention used in the code predictor.
    
    Inherits from vLLM's Attention layer for optimized attention computation.
    Adds Q/K/V projections, RoPE, and query/key normalization on top.
    """

    def __init__(self, config, layer_idx: int, vllm_config: VllmConfig = None):
        """
        Initialize attention layer with vLLM backend support.
        
        Args:
            config: Code predictor configuration
            layer_idx: Layer index for unique prefix naming
            vllm_config: vLLM configuration for cache/quantization settings
        """
        self.num_heads = config.code_predictor_config.num_attention_heads
        self.num_key_value_heads = config.code_predictor_config.num_key_value_heads
        self.head_dim = getattr(config.code_predictor_config, "head_dim",
                                config.code_predictor_config.hidden_size // config.code_predictor_config.num_attention_heads)
        self.hidden_size = config.code_predictor_config.hidden_size
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for grouped attention.")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Calculate attention scale
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize parent vLLM Attention layer with unique prefix per layer
        # Note: Code predictor doesn't use KV caching, so we pass minimal config
        super().__init__(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=scale,
            num_kv_heads=self.num_key_value_heads,  # Self-attention: num_kv_heads = num_heads
            cache_config=vllm_config.cache_config if vllm_config else None,
            quant_config=vllm_config.quant_config if vllm_config else None,
            prefix=f"model.layers.{layer_idx}.self_attn",  # Unique per layer
            attn_type=AttentionType.DECODER,
        )
   
        # Q/K/V projection layers (not part of vLLM Attention)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Query/Key normalization (Qwen3 specific)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with Q/K/V projection, RoPE, and attention.
        
        Note: We override the parent's forward to handle our specific case
        of doing Q/K/V projection and RoPE application before attention.
        For the code predictor, we use simple PyTorch attention without
        KV caching since we're not doing incremental generation.
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply query/key normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary embeddings (expects [batch, heads, seq, head_dim])
        q_heads = q.transpose(1, 2).contiguous()
        k_heads = k.transpose(1, 2).contiguous()
        q_heads, k_heads = apply_rotary_pos_emb(q_heads, k_heads, cos, sin)
        v_heads = v.transpose(1, 2).contiguous()
        q_flat = q_heads.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        k_flat = k_heads.transpose(1, 2).contiguous().view(bsz, seq_len,
                                                           self.num_key_value_heads * self.head_dim)
        v_flat = v.view(bsz, seq_len, self.num_key_value_heads * self.head_dim)
        try:
            attn_output = super().forward(q_flat, k_flat, v_flat)
            attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.head_dim)
        except AssertionError:
            k_local = k_heads
            v_local = v_heads
            if self.num_key_value_groups > 1:
                k_local = repeat_kv(k_local, self.num_key_value_groups)
                v_local = repeat_kv(v_local, self.num_key_value_groups)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_heads, k_local, v_local, attn_mask=causal_mask, dropout_p=0.0, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                bsz, seq_len, self.num_heads * self.head_dim
            )

        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3OmniMoeTalkerCodePredictorMLP(nn.Module):
    """Feed-forward network used in the code predictor."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.code_predictor_config.hidden_size,
                                   config.code_predictor_config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.code_predictor_config.hidden_size,
                                 config.code_predictor_config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.code_predictor_config.intermediate_size,
                                   config.code_predictor_config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3OmniMoeTalkerCodePredictorDecoderLayer(nn.Module):
    """Decoder layer mirroring the Transformers implementation."""

    def __init__(self, config, layer_idx: int, vllm_config: VllmConfig = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3OmniMoeTalkerCodePredictorAttention(config, layer_idx, vllm_config)
        self.mlp = Qwen3OmniMoeTalkerCodePredictorMLP(config)
        self.input_layernorm = nn.RMSNorm(config.code_predictor_config.hidden_size,
                                          eps=config.code_predictor_config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.code_predictor_config.hidden_size,
                                                   eps=config.code_predictor_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        print(f"residual {residual.shape}")
        hidden_states = self.input_layernorm(hidden_states)
        print(f"hidden_states normed {hidden_states.shape}")
        hidden_states = self.self_attn(hidden_states, causal_mask, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig = None):
        super().__init__()
        self.config = config
        self.vocab_size = config.code_predictor_config.vocab_size
        self.num_code_groups = config.code_predictor_config.num_code_groups

        self.model = Qwen3OmniMoeTalkerCodePredictorModel(config, vllm_config)

        # Output heads for each residual layer (1-num_layers-1)
        self.lm_head = nn.ModuleList([
            nn.Linear(config.code_predictor_config.hidden_size, config.code_predictor_config.vocab_size, bias=False)
            for _ in range(self.num_code_groups - 1)
        ])
        
        # Reference to talker's codec embedding (will be set by parent model)
        # This is used to embed layer-0 codes in the same semantic space as the talker
        self.talker_codec_embedding = None

    def forward(
            self,
            inputs_embeds: torch.Tensor,  # [batch, seq_len, hidden_size]
            layer_idx: int,  # Which layer to predict (0-num_layers-2 for layers 1-num_layers-1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for code predictor.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            layer_idx: Which residual layer to predict (0-num_layers-2 for layers 1-num_layers-1)

        Returns:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            hidden_states: Output hidden states [batch_size, seq_len, hidden_size]
        """
        # Pass through base model
        hidden_states = self.model(inputs_embeds)

        # Get logits from corresponding head
        logits = self.lm_head[layer_idx](hidden_states)

        return logits, hidden_states

    def predict_residual_codes(
            self,
            initial_input: torch.Tensor,  # [batch, 2, hidden_size]: past_hidden + layer0_code
            num_codes: int = 7,  # Predict layers 1-num_layers-1
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.8,
    ) -> torch.Tensor:
        """
        Predict all residual codec codes (layers 1-num_layers-1) autoregressively.

        Args:
            initial_input: [batch_size, 2, hidden_size] - past_hidden + layer0_embedding
            num_codes: Number of layers to predict (num_layers-1 for layers 1-num_layers-1)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            predicted_codes: [batch_size, num_codes] - codes for layers 1-num_layers-1
        """
        batch_size = initial_input.shape[0]
        device = initial_input.device

        inputs_embeds = initial_input  # [batch, 2, hidden]
        predicted_codes = []

        # Predict each layer autoregressively
        for layer_idx in range(num_codes):
            # Forward pass for this layer
            logits, hidden_states = self.forward(
                inputs_embeds=inputs_embeds,
                layer_idx=layer_idx,
            )

            # Sample from the last position
            last_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = last_logits < torch.topk(last_logits, top_k)[0][..., -1, None]
                last_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                last_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(last_logits, dim=-1)
            next_code = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            predicted_codes.append(next_code)

            # Prepare input for next layer (if not the last one)
            if layer_idx < num_codes - 1:
                # Embed the predicted code for next layer
                next_embed = self.model.codec_embedding[layer_idx + 1](next_code)  # [batch, 1, hidden]
                # Concatenate with previous sequence
                inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

        # Stack all predicted codes
        predicted_codes = torch.cat(predicted_codes, dim=1)  # [batch, num_codes]

        return predicted_codes

    def generate_residual_codes_for_token(
            self,
            layer0_code: torch.Tensor,
            past_hidden: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.8,
    ) -> torch.Tensor:
        """
        Generate residual codes (layers except 0) for a single token/position.

        This should be called DURING generation, after each layer 0 code is sampled.

        Args:
            layer0_code: Layer 0 code just sampled [batch, 1]
            past_hidden: Hidden state from last transformer layer [batch, 1, hidden]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            residual_codes: [batch, num_layers-1] - Codes for layers except 0
        """
        # Embed the layer 0 code using talker's codec embedding (same as HF)
        # This maintains semantic consistency with the talker's generation
        if self.talker_codec_embedding is None:
            raise RuntimeError("talker_codec_embedding must be set before calling generate_residual_codes_for_token")
        layer0_embed = self.talker_codec_embedding(layer0_code)  # [batch, 1, hidden_talker]

        # Prepare initial input: [past_hidden, layer0_embed]
        # Both are now in talker's hidden dimension (e.g., 2048)
        initial_input = torch.cat([past_hidden, layer0_embed], dim=1)  # [batch, 2, hidden_talker]

        # Predict residual codes (layers num_layers-1)
        residual_codes = self.predict_residual_codes(
            initial_input=initial_input,
            num_codes=self.config.code_predictor_config.num_code_groups - 1,  # num_layers-1 codes
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # [batch, num_layers-1]

        return residual_codes

    def generate_full_codec_codes(
            self,
            layer0_code_ids: torch.Tensor,
            past_hidden_states: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.8,
    ) -> torch.Tensor:
        """
        Generate complete RVQ codec codes.

        This combines:
        - Layer 0: Already predicted by main talker (codec_head)
        - Layers 1-num_layers-1: Predicted by code_predictor

        Args:
            layer0_code_ids: Layer 0 codes from main talker [batch, seq_len]
            past_hidden_states: Hidden states from last layer of main talker [batch, seq_len, hidden]
            temperature: Sampling temperature for code predictor
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling

        Returns:
            all_codes: [batch, num_layers, seq_len] - Complete RVQ codes
        """
        batch_size, seq_len = layer0_code_ids.shape
        device = layer0_code_ids.device

        # We need to run code predictor for each position in the sequence
        all_residual_codes = []

        for pos in range(seq_len):
            # Get the layer 0 code and hidden state for this position
            current_code = layer0_code_ids[:, pos:pos + 1]  # [batch, 1]
            current_hidden = past_hidden_states[:, pos:pos + 1, :]  # [batch, 1, hidden_talker]

            # Embed the layer 0 code using talker's codec embedding (same as HF)
            # This matches transformers' use of self.get_input_embeddings()(input_ids)
            # which returns talker's codec_embedding, maintaining semantic consistency
            if self.talker_codec_embedding is None:
                raise RuntimeError("talker_codec_embedding must be set before calling generate_full_codec_codes")
            layer0_embed = self.talker_codec_embedding(current_code)  # [batch, 1, hidden_talker]

            # Prepare initial input: [past_hidden, layer0_embed]
            # Both are now in talker's hidden dimension (e.g., 2048)
            initial_input = torch.cat([current_hidden, layer0_embed], dim=1)  # [batch, 2, hidden_talker]

            print(f"initial_input shape {initial_input.shape}, \n"
                  f".num_code_groups {self.config.code_predictor_config.num_code_groups}, \n"
                  f"batch_size {batch_size}, \n"
                  f"seq_len {seq_len}, \n"
                  f"pos {pos}")

            # Predict residual codes (layers 1-num_layers)
            residual_codes = self.predict_residual_codes(
                initial_input=initial_input,
                num_codes=self.config.code_predictor_config.num_code_groups - 1,  # 16 codes
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # [batch, num_layers-1]

            all_residual_codes.append(residual_codes)

        # Stack residual codes across sequence
        all_residual_codes = torch.stack(all_residual_codes, dim=2)  # [batch, num_layers-1, seq_len]

        # Combine layer 0 with residual layers
        layer0_codes_expanded = layer0_code_ids.unsqueeze(1)  # [batch, 1, seq_len]
        all_codes = torch.cat([layer0_codes_expanded, all_residual_codes], dim=1)  # [batch, num_layers, seq_len]

        return all_codes

class Qwen3OmniMoeTalkerCodePredictorModel(nn.Module):
    """
    Base model for code predictor (without lm_head).
    
    Matches HuggingFace Transformers structure exactly.
    Contains the core transformer components.
    """
    
    def __init__(self, config, vllm_config: VllmConfig = None):
        super().__init__()
        self.config = config
        self.vocab_size = config.code_predictor_config.vocab_size
        self.hidden_size = config.code_predictor_config.hidden_size
        self.num_code_groups = config.code_predictor_config.num_code_groups
        self.head_dim = getattr(config.code_predictor_config, "head_dim",
                                config.code_predictor_config.hidden_size // config.code_predictor_config.num_attention_heads)
        
        # Codec embeddings for layers 1-num_layers-1 (layer 0 from main talker)
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(config.code_predictor_config.vocab_size, config.code_predictor_config.hidden_size)
            for _ in range(self.num_code_groups - 1)  # 7 embeddings
        ])
        
        # Transformer layers
        self.layers = nn.ModuleList([
            Qwen3OmniMoeTalkerCodePredictorDecoderLayer(config, layer_idx, vllm_config)
            for layer_idx in range(config.code_predictor_config.num_hidden_layers)
        ])
        
        # Normalization
        self.norm = nn.RMSNorm(config.code_predictor_config.hidden_size,
                               eps=config.code_predictor_config.rms_norm_eps)
        
        # Rotary positional embeddings
        self.rotary_emb = Qwen3OmniMoeTalkerCodePredictorRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.code_predictor_config.max_position_embeddings,
        )
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through base model, returns hidden states."""
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Get rotary embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        
        # Pass through transformer layers
        hidden_states = inputs_embeds
        print(f"self.layers {len(self.layers)}, "
              f"hidden_states shape {hidden_states.shape}, "
              f"causal_mask shape {causal_mask.shape}, "
              f"cos shape {cos.shape}, "
              f"sin shape {sin.shape}")
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, cos, sin)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(nn.Module,
                                                              SupportsMultiModal,
                                                              SupportsPP,
                                                              Qwen3OmniMoeConditionalGenerationMixin,):
    """
    Code Predictor for generating residual codec layers 1-num_layers-1.
    
    Matches HuggingFace structure with self.model and self.lm_head separation.
    This allows direct weight loading from HuggingFace checkpoints.
    
    Architecture:
    - model: Qwen3OmniMoeTalkerCodePredictorModel (embeddings + transformer)
    - lm_head: ModuleList of x output heads
    """
    logger = init_logger(__name__)

    # Weight mapping from HuggingFace to vLLM naming convention
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "talker.code_predictor.lm_head.": "code_predictor.lm_head.",
            "talker.code_predictor.model.": "code_predictor.model.",
            # Map talker's model codec embedding to our talker_codec_embedding
            "talker.model.codec_embedding.": "talker_codec_embedding.",
            "talker.code_predictor": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        talker_code_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_code_predictor_config
        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        # Create talker's codec embedding (for layer-0 codes)
        # This matches HF's usage of self.get_input_embeddings() in the talker
        # which returns codec_embedding with talker's hidden dimension
        self.talker_codec_embedding = nn.Embedding(
            self.config.text_config.vocab_size,
            self.config.text_config.hidden_size
        )

        self.code_predictor = Qwen3OmniMoeTalkerCodePredictor(self.config, vllm_config)
        
        # Set the reference in code_predictor to use talker's embedding
        self.code_predictor.talker_codec_embedding = self.talker_codec_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.8,
        generation_steps: int | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Generate full RVQ codec codes for the provided sequence.

        The code predictor consumes the layer-0 codec codes produced by the talker
        alongside the talker's hidden states, and autoregressively predicts the remaining
        residual layers (to num_codec_groups).

        Returns:
            A tensor of shape [batch, num_code_groups, seq_len] containing the complete set
            of codec codes
        """
        if input_ids is None:
            raise ValueError("`input_ids` containing layer-0 codec codes must be provided.")
        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` containing talker hidden states must be provided.")

        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # Ensure the tensors are contiguous for the autoregressive sampling loop
        inputs_embeds = inputs_embeds.contiguous()
        input_ids = input_ids.contiguous()

        print(f"inputs_embeds {inputs_embeds.shape}, input_ids {input_ids.shape}")

        return self.code_predictor.generate_full_codec_codes(
                layer0_code_ids=input_ids,
                past_hidden_states=inputs_embeds,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the talker model.

        The weight mapping translates from HuggingFace naming convention
        to vLLM's internal structure. Code predictor weights are routed
        to its custom loader for vocab extension support.
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "code2wav.", "talker.text_projection.",
                           "talker.hidden_projection.", "talker.model.", "talker.codec_head", ]
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
                self.__class__.__name__, True, total_bytes / (1024 ** 2), str(device)
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
        print(f"================ mm_input_by_modality {mm_input_by_modality}")
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
                print(f"multimodal_input {multimodal_input}\n"
                      f"multimodal_input pixel_values_videos shape "
                      f"{multimodal_input["pixel_values_videos"].shape}")
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                audio_embeddings = self.hidden_projection(audio_embeddings)
                multimodal_embeddings += tuple(audio_embeddings)
            print(f"multimodal_embeddings {multimodal_embeddings}")
        return multimodal_embeddings


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
