# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe unified model (thinker + talker + code2wav)."""

import os
from functools import cached_property
from typing import Dict, Iterable, NamedTuple, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig
)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .qwen3_omni_moe_thinker import (
    Qwen3OmniMoeConditionalGenerationMixin,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)
from .utils import add_prefix_to_loaded_weights

# Special token IDs for Qwen3 Omni MoE
# Reference: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/tokenizer_config.json

# Audio tokens (thinker vocabulary, for marking audio boundaries)
AUDIO_START_TOKEN_ID = 151669        # <|audio_start|> (audio_bos_token)
AUDIO_END_TOKEN_ID = 151670          # <|audio_end|> (audio_eos_token)
AUDIO_PAD_TOKEN_ID = 151675          # <|audio_pad|>

# TTS text tokens (thinker vocabulary, for text-to-speech control)
TTS_PAD_TOKEN_ID = 151671            # <tts_pad>
TTS_BOS_TOKEN_ID = 151672            # <tts_text_bos>
TTS_EOS_TOKEN_ID = 151673            # <tts_text_eod> (end of dialogue)
TTS_BOS_SINGLE_TOKEN_ID = 151674     # <tts_text_bos_single>

# Talker codec tokens (talker vocabulary, used for RVQ code generation)
TALKER_CODEC_PAD_TOKEN_ID = 4196     # Padding token
TALKER_CODEC_BOS_TOKEN_ID = 4197     # Beginning of speech
TALKER_CODEC_EOS_TOKEN_ID = 4198     # End of speech
TALKER_CODEC_NOTHINK_ID = 4203       # No-think mode
TALKER_CODEC_THINK_BOS_ID = 4204     # Think mode start
TALKER_CODEC_THINK_EOS_ID = 4205     # Think mode end

logger = init_logger(__name__)


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: Optional[dict] = None
    intermediate_tensors: Optional[IntermediateTensors] = None


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, Qwen3OmniMoeConditionalGenerationMixin
):
    """
    Unified Qwen3 Omni MoE model combining thinker, talker, and code2wav.
    
    Architecture:
    - Thinker: Multimodal understanding (text + audio + video) → text generation
    - Talker: Text embeddings → 8-layer RVQ codec codes
    - Code2Wav: 8-layer RVQ codes → audio waveform
    
    Usage:
        Set `model_stage` in vllm_config to one of: "thinker", "talker", "code2wav"
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        
        # Keep vllm_config for later submodule init
        self.vllm_config = vllm_config

        # Initialize thinker components
        thinker_config: Qwen3OmniMoeThinkerConfig = config.thinker_config
        self.thinker_config = thinker_config
        self.multimodal_config = multimodal_config

        # Initialize talker components
        talker_config: Qwen3OmniMoeTalkerConfig = config.talker_config
        self.talker_config = talker_config

        # Initialize talker code predictor components
        code_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig = config.talker_config
        self.code_predictor_config = code_predictor_config
        
        # Initialize code2wav components
        code2wav_config: Qwen3OmniMoeCode2WavConfig = config.code2wav_config
        self.code2wav_config = code2wav_config

        # Determine model stage
        self.model_stage = vllm_config.model_config.model_stage
        
        if self.model_stage == "thinker":
            # Initialize thinker model (multimodal processing + text generation)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=thinker_config,
                architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            logger.debug(f"=================thinker model {self.model}")
            self.talker = None
            self.code_predictor = None
            self.code2wav = None      
        elif self.model_stage == "talker":
            self.thinker = None
            # Initialize talker model (text embeddings → codec codes)
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=talker_config,
                architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"],
            )
            self.talker.init_multi_modal(thinker_config)
            self.model = self.talker
            logger.debug(f"=================talker model {self.model}")
            self.code_predictor = None
            self.code2wav = None
        elif self.model_stage == "code_predictor":
            self.thinker = None
            self.talker = None
            # Initialize talker model (text embeddings → codec codes)
            self.code_predictor = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker.code_predictor"),
                hf_config=code_predictor_config,
                architectures=["Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration"],
            )
            self.code_predictor.init_multi_modal(thinker_config)
            self.model = self.code_predictor
            logger.debug(f"=================code_predictor model {self.model}")
            self.code2wav = None 
        elif self.model_stage == "code2wav":
            self.thinker = None
            self.talker = None
            self.code_predictor = None
            # Initialize code2wav (codec codes → audio waveform)
            self.code2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=code2wav_config,
                architectures=["Qwen3OmniMoeCode2Wav"],
            )
            self.model = self.code2wav
            logger.debug(f"=================code2wav model {self.model}")
        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                f"Must be one of: 'thinker', 'talker', 'code2wav'"
            )

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors
            if self.model_stage == "thinker"
            else lambda: None
        )

    # ==================== Device utilities ====================
    
    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        thinker_device: Optional[Union[str, torch.device]] = None,
        talker_device: Optional[Union[str, torch.device]] = None,
        code_predictor_device: Optional[Union[str, torch.device]] = None,
        code2wav_device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Optionally move thinker/talker/code2wav to different devices.
        
        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                code2wav_device='cuda:2',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)
        if code_predictor_device is not None and self.code_predictor is not None:
            self.code_predictor.to(code_predictor_device)
        if code2wav_device is not None and self.code2wav is not None:
            self.code2wav.to(code2wav_device)

    @cached_property
    def sampler(self):
        """Get sampler from active model."""
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return get_sampler()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        """Get input embeddings for the active model stage."""
        if self.model_stage == "code2wav":
            # Code2wav doesn't use text embeddings
            return (
                torch.zeros_like(input_ids)
                .reshape(-1, 1)
                .repeat(1, self.vllm_config.model_config.get_hidden_size())
            )
        return self.model.get_input_embeddings(input_ids, multimodal_embeddings)

    def get_multimodal_embeddings(self, **kwargs):
        """Delegate to active model for multimodal processing."""
        return self.model.get_multimodal_embeddings(**kwargs)

    # ==================== Forward Pass ====================
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        generate_audio: bool = True,
        voice_type: str = "default",
        codec: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        logits_index: Optional[int] = None,
        additional_information: Optional[dict[str, object]] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors, OmniOutput]:
        """
        Unified forward pass for all model stages.
        
        Workflow:
        1) Thinker: multimodal understanding → text hidden states
        2) Talker -> Code Predictor: text embeddings → codec codes (layer 0 + code_predictor:residual layers)
        3) Code2wav: 8-layer RVQ codes → audio waveform
        
        Returns:
            OmniOutput with text_hidden_states and optional audio
        """
        
        # ========== Stage 1: Thinker ==========
        if self.model_stage == "thinker":
            # Normalize to batched inputs if needed
            added_batch_dim = False
            if input_ids is not None and input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                added_batch_dim = True
            if positions is not None and positions.ndim == 1:
                positions = positions.unsqueeze(0)
                added_batch_dim = True
            if inputs_embeds is not None and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
                added_batch_dim = True
            
            thinker_dev = self._module_device(self.thinker)
            
            # Handle None input_ids
            if input_ids is None:
                input_ids = torch.zeros(
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=thinker_dev,
                ).unsqueeze(0)
                added_batch_dim = True
            
            # Move to thinker device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)
            
            # Run thinker forward
            thinker_output = self.thinker(
                input_ids=input_ids,
                positions=positions[0] if positions.ndim > 1 else positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            
            if isinstance(thinker_output, tuple):
                embeds, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output
            
            # Return text-only output
            return OmniOutput(
                text_hidden_states=(
                    text_hidden_states.squeeze(0)
                    if added_batch_dim
                    else text_hidden_states
                ),
                multimodal_outputs=None,
            )
        
        # ========== Stage 2.1: Talker ==========
        elif self.model_stage == "talker":
            # Handle profile mode
            if input_ids is None and additional_information is None:
                input_ids = torch.zeros(
                    inputs_embeds.shape[0],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                additional_information = {}
                self.thinker_reply_part = torch.zeros_like(inputs_embeds)
                is_profile = True
            else:
                is_profile = False
            
            # Process thinker → talker transition
            if (
                input_ids is not None
                and additional_information is not None
                and not is_profile
            ):
                # Read from additional_information dict
                thinker_result = None
                if additional_information is not None and isinstance(
                    additional_information, dict
                ):
                    thinker_result = additional_information.get("thinker_result")
                    prompt_embeds = additional_information.get("prompt_embeds")
                    prompt_token_ids = additional_information.get("prompt_token_ids")
                    thinker_output_token_ids = additional_information.get(
                        "thinker_output_token_ids"
                    )
                else:
                    thinker_result = torch.zeros_like(inputs_embeds)
                    prompt_embeds = torch.zeros_like(inputs_embeds)
                    prompt_token_ids = torch.zeros(
                        inputs_embeds.shape[0],
                        dtype=torch.int64,
                        device=inputs_embeds.device,
                    )
                    thinker_output_token_ids = torch.zeros(
                        inputs_embeds.shape[0],
                        dtype=torch.int64,
                        device=inputs_embeds.device,
                    )
                
                if thinker_result is None:
                    thinker_result = torch.zeros_like(inputs_embeds)
                
                self.thinker_reply_part = thinker_result.squeeze(0)
                if self.thinker_reply_part.shape[1] > 1:
                    self.thinker_reply_part = self.thinker_reply_part[1:, :]
                
                # Prefill: project thinker outputs to talker
                input_ids, inputs_embeds = self._thinker_to_talker_prefill(
                    voice_type=voice_type,
                    output_prompt_embeds=thinker_result,
                    output_token_ids=thinker_output_token_ids,
                    thinker_prompt_embeds=prompt_embeds,
                    prompt_token_ids=prompt_token_ids,
                )
            elif not is_profile:
                # Decode: one-step generation
                input_ids, inputs_embeds = self._thinker_to_talker_decode_one_step(
                    output_prompt_embeds=(
                        self.thinker_reply_part[:1]
                        if self.thinker_reply_part.shape[0] >= 1
                        else torch.zeros(1, self.thinker_reply_part.shape[1])
                        .to(self._module_device(self.model))
                        .to(torch.bfloat16)
                        + (-1.25 * 2 ** (-123))
                    ),
                    output_token_ids=input_ids,
                )
                
                if self.thinker_reply_part.shape[0] >= 1:
                    self.thinker_reply_part = self.thinker_reply_part[1:, :]
            
            # Run talker forward
            with torch.inference_mode():
                talker_hidden = self.talker(
                    input_ids=input_ids,
                    positions=positions[0] if positions.ndim > 1 else positions,
                    inputs_embeds=inputs_embeds,
                )
            
            return OmniOutput(
                text_hidden_states=talker_hidden,
                multimodal_outputs=None,
            )

        # ========== Stage 2.2: Code Predictor ==========  
        elif self.model_stage == "code_predictor":
            if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)

            talker_hidden_states = (
                additional_information.get("talker_hidden_states")
                if isinstance(additional_information, dict)
                else inputs_embeds
            )
            if not isinstance(talker_hidden_states, torch.Tensor):
                hidden_size = inputs_embeds.shape[-1] if isinstance(inputs_embeds, torch.Tensor) else (
                    self.talker_config.text_config.hidden_size
                    if hasattr(self, "talker_config") and self.talker_config is not None
                    else self.config.text_config.hidden_size
                )
                device = inputs_embeds.device if isinstance(inputs_embeds, torch.Tensor) else (
                    input_ids.device if isinstance(input_ids, torch.Tensor)
                    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                batch = input_ids.shape[0] if input_ids.ndim > 1 else 1
                seq_len = input_ids.shape[-1] if input_ids.ndim > 1 else input_ids.shape[0]
                talker_hidden_states = torch.zeros(
                    batch,
                    seq_len,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                )
            elif talker_hidden_states.ndim == 2:
                talker_hidden_states = talker_hidden_states.unsqueeze(0)

            new_input_ids, inputs_embeds = self._talker_to_code_predictor(
                talker_hidden_states=talker_hidden_states,
                layer0_token_ids=input_ids,
            )
            if new_input_ids is not None:
                input_ids = new_input_ids

            print(f"input_ids shape {input_ids.shape}, new_input_ids shape {new_input_ids.shape} \n"
                  f"inputs_embeds shape {inputs_embeds.shape}")
            # Run code predictor forward
            with torch.inference_mode():
                code_predictor_hidden = self.code_predictor(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                )
            return OmniOutput(
                text_hidden_states=code_predictor_hidden,
                multimodal_outputs=None,
            )
        
        # ========== Stage 3: Code2Wav ==========
        elif self.model_stage == "code2wav":
            # Extract codec codes from input
            code = (
                input_ids
                if input_ids is not None
                else torch.zeros(
                    inputs_embeds.shape[0],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            
            # Remove EOS token if present
            if code[-1] == TALKER_CODEC_EOS_TOKEN_ID:
                code = code[:-1]
            
            # Generate audio from codec codes
            audio_tensor = self.generate_audio(code, voice_type)
            
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": audio_tensor},
            )
        
        # Fallback (shouldn't reach here)
        return OmniOutput(
            text_hidden_states=torch.zeros(
                [inputs_embeds.shape[0], self.talker.config.hidden_size],
                dtype=torch.bfloat16,
            ).to(self._module_device(self.model)),
            multimodal_outputs=None,
        )

    # ==================== Audio Generation ====================
    
    def generate_audio(self, code: torch.Tensor, voice_type: str) -> torch.Tensor:
        """
        Generate audio waveform from codec codes.
        
        Args:
            code: [8, T] - 8-layer RVQ codec codes
            voice_type: Voice type (not used in Qwen3, kept for compatibility)
        
        Returns:
            audio_tensor: [1, waveform_len] - Audio waveform
        """
        code2wav_dev = self._module_device(self.code2wav)
        
        # Convert to tensor if needed
        if isinstance(code, torch.Tensor):
            code_tensor = code.to(dtype=torch.long, device=code2wav_dev)
        else:
            code_tensor = torch.as_tensor(
                code, dtype=torch.long, device=code2wav_dev
            )
        
        # Ensure shape is [batch=1, 8, T]
        if code_tensor.ndim == 2:
            # [8, T] → [1, 8, T]
            code_tensor = code_tensor.unsqueeze(0)
        elif code_tensor.ndim == 1:
            # [T] → assume single layer, expand to 8 layers
            code_tensor = code_tensor.unsqueeze(0).unsqueeze(0)
            code_tensor = code_tensor.expand(1, 8, -1)
        
        # Use chunked decode for memory efficiency
        audio_tensor = self.code2wav.chunked_decode(
            code_tensor,
            chunk_size=300,
            left_context_size=25,
        )
        
        return audio_tensor

    # ==================== Thinker-Talker Projection ====================
    
    def _load_talker_embedding(self) -> torch.nn.Embedding:
        """Load talker embedding layer."""
        return self.talker.language_model.model.codec_embedding

    def _init_special_tokens_embeddings(self) -> Set[str]:
        """
        Initialize special token embeddings for thinker-talker projection.
        
        Following Transformers implementation:
        - TTS tokens (BOS/EOS/PAD) come from thinker's embedding, projected to talker space
        - Codec tokens (BOS/EOS/PAD/NOTHINK/THINK_*) come from talker's embedding
        - Speaker tokens are also from talker's embedding
        
        Note on projections:
        - text_projection: Used here for text token embeddings (thinker → talker dimension)
        - hidden_projection: Used at runtime for multimodal hidden states (audio/image/video)
          from thinker's last layer, not needed for special token initialization
        """
        # Get embeddings from both models
        # self.thinker_embedding = self.thinker.model.get_input_embeddings()
        self.talker_embedding = self._load_talker_embedding()
        
        # Get configuration
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config
        
        # Initialize TTS special tokens (from THINKER embedding, then projected)
        # These are used in line 4020-4029 of Transformers modeling_qwen3_omni_moe.py:
        # talker_special_tokens = torch.tensor(
        #     [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
        # )
        # tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        #     self.talker.text_projection(self.thinker.get_input_embeddings()(talker_special_tokens))
        # )
        
        # TODO: runtime calculate tts embedding
        # talker_special_tokens = torch.tensor(
        #     [[main_config.tts_bos_token_id, main_config.tts_eos_token_id, main_config.tts_pad_token_id]],
        #     device=self._module_device(self.thinker),
        #     dtype=torch.long,
        # )
        # # Get thinker embeddings and project to talker space
        # thinker_special_embeds = self.thinker_embedding(talker_special_tokens)  # [1, 3, thinker_hidden]
        # projected_special_embeds = self.talker.text_projection(thinker_special_embeds)  # [1, 3, talker_hidden]
        
        # self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = projected_special_embeds.chunk(3, dim=1)
        # Shape: each is [1, 1, talker_hidden_size]
        # TODO: end
        
        # Initialize codec special tokens (from TALKER embedding)
        # These are used in line 3857-3870 of Transformers modeling_qwen3_omni_moe.py:
        # codec_special_tokens = torch.tensor([[
        #     self.config.talker_config.codec_nothink_id,
        #     self.config.talker_config.codec_think_bos_id,
        #     self.config.talker_config.codec_think_eos_id,
        #     speaker_id,
        #     self.config.talker_config.codec_pad_id,
        #     self.config.talker_config.codec_bos_id,
        # ]])
        # self.talker.get_input_embeddings()(codec_special_tokens)
        
        codec_special_tokens = torch.tensor(
            [[
                talker_hf_config.codec_nothink_id,
                talker_hf_config.codec_think_bos_id,
                talker_hf_config.codec_think_eos_id,
                talker_hf_config.codec_pad_id,
                talker_hf_config.codec_bos_id,
                talker_hf_config.codec_eos_token_id,
            ]],
            device=self._module_device(self.talker),
            dtype=torch.long,
        )
        codec_embeds = self.talker_embedding(codec_special_tokens)  # [1, 6, talker_hidden]
        (
            self.embed_codec_nothink_token,
            self.embed_codec_think_bos_token,
            self.embed_codec_think_eos_token,
            self.embed_codec_pad_token,
            self.embed_codec_bos_token,
            self.embed_codec_eos_token,
        ) = codec_embeds.chunk(6, dim=1)
        
        # Speaker token IDs (for voice selection)
        # In Qwen3, speaker_id mapping is in talker_config.speaker_id
        if hasattr(talker_hf_config, "speaker_id") and talker_hf_config.speaker_id:
            self.tts_text_spk_token_ids = talker_hf_config.speaker_id
        else:
            # Default to audio_start_token_id if no speaker mapping
            self.tts_text_spk_token_ids = {
                "default": talker_hf_config.audio_start_token_id,
                "Ethan": talker_hf_config.audio_start_token_id,
                "prefix_caching": talker_hf_config.audio_start_token_id,
            }
        
        self.default_tts_text_spk_type = list(self.tts_text_spk_token_ids.keys())[0]
        
        return set(["thinker_embedding.weight", "talker_embedding.weight"])

    def _get_text_spk_token_id(self, voice_type: str) -> int:
        """Get speaker token ID for voice type."""
        if voice_type not in self.tts_text_spk_token_ids:
            return self.tts_text_spk_token_ids[self.default_tts_text_spk_type]
        return self.tts_text_spk_token_ids[voice_type]

    def _thinker_to_talker_prefill(
        self,
        voice_type: str,
        output_prompt_embeds: torch.Tensor,
        output_token_ids: torch.Tensor,
        thinker_prompt_embeds: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        multimodal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project thinker outputs to talker inputs during prefill stage.
        
        Returns:
            (input_ids, input_embeds) for talker
        """

        projected_embeds = self.talker.project_thinker_outputs(
            thinker_embeds=thinker_prompt_embeds,   # Text embeddings
            thinker_hidden_states=output_prompt_embeds,  # Multimodal hidden states
            is_multimodal_mask=multimodal_mask,      # Which tokens are multimodal
        )


        # DEBUG: Check special tokens in talker config
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config
        
        # Concatenate: thinker prompt + codec pad + first output token
        prompt_embeds = torch.cat(
            [
                projected_embeds,  # ✅ Now properly projected!
                self.embed_codec_pad_token,
                self.embed_codec_bos_token,
            ],
            dim=0,
        )
        
        # Construct input token IDs
        prompt_token_ids_processed = prompt_token_ids + [
            TALKER_CODEC_PAD_TOKEN_ID,
            output_token_ids[0],
        ]
        input_tokens_len = len(prompt_token_ids_processed)
        
        # Mask thinker tokens, keep only codec tokens
        if input_tokens_len > 2:
            prompt_token_ids_processed = [
                TALKER_CODEC_PAD_TOKEN_ID
            ] * (input_tokens_len - 2) + [
                TALKER_CODEC_PAD_TOKEN_ID,
                TALKER_CODEC_BOS_TOKEN_ID,
            ]
        else:
            prompt_token_ids_processed = [
                TALKER_CODEC_PAD_TOKEN_ID,
                TALKER_CODEC_BOS_TOKEN_ID,
            ][-input_tokens_len:]
        
        if isinstance(prompt_token_ids_processed, list):
            prompt_token_ids_processed = (
                torch.Tensor(prompt_token_ids_processed)
                .to(torch.int64)
                .to(self._module_device(self.talker))
            )
        
        return prompt_token_ids_processed, prompt_embeds

    def _thinker_to_talker_decode_one_step(
        self,
        output_prompt_embeds: torch.Tensor,
        output_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project thinker outputs to talker inputs during decode stage.
        
        Returns:
            (input_ids, input_embeds) for talker
        """
        # Add thinker embedding to talker token embedding
        processed_output_token_embeds = (
            output_prompt_embeds + self.talker.get_input_embeddings(output_token_ids)
        )
        return output_token_ids, processed_output_token_embeds


    def _talker_to_code_predictor(
        self,
        talker_hidden_states: Optional[torch.Tensor],
        layer0_token_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project talker outputs to code predictor inputs.

        Returns:
            (input_ids, input_embeds) for code predictor.
        """
        predictor = getattr(self, "code_predictor", None)
        device = self._module_device(predictor) if predictor is not None else (
            talker_hidden_states.device if isinstance(talker_hidden_states, torch.Tensor)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not isinstance(talker_hidden_states, torch.Tensor):
            raise ValueError("Talker hidden states must be provided for the code predictor stage.")

        inputs_embeds = talker_hidden_states.to(device=device, dtype=torch.bfloat16)
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if not isinstance(layer0_token_ids, torch.Tensor):
            raise ValueError("Layer-0 codec token ids must accompany talker hidden states.")
        input_ids = layer0_token_ids.to(device=device, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        return input_ids, inputs_embeds

    # ==================== Logits and Sampling ====================
    
    def compute_logits(
        self,
        hidden_states: Union[torch.Tensor, OmniOutput],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        print(f"================compute_logits hidden_states {hidden_states}, \n"
                    f"OmniOutput {OmniOutput}, \n "
              f"sampling_metadata {sampling_metadata}")
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        
        # Use active model for logits computation
        return self.model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample from logits."""
        return self.model.sample(logits, sampling_metadata)

    # ==================== Weight Loading ====================
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        code_predictor_weights = []
        code2wav_weights = []
        
        # Separate weights by component
        for k, v in weights:
            if k.startswith("thinker."):
                thinker_weights.append((k, v))
            elif k.startswith("talker."):
                if k.startswith("talker.code_predictor."):
                    code_predictor_weights.append((k, v))
                else:
                    talker_weights.append((k, v))
            elif k.startswith("code2wav."):
                code2wav_weights.append((k, v))
            else:
                logger.warning(f"Unknown weight prefix: {k}")
        
        # Load thinker weights
        if self.thinker and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(
                thinker_loaded, "thinker"
            )
            loaded_weights.update(thinker_loaded)
        
        # Load talker weights
        if self.talker and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)
            loaded_weights.update(self._init_special_tokens_embeddings())
        
        # Load talker code predictor weights
        if self.code_predictor and code_predictor_weights:
            talker_code_predictor_loaded = self.code_predictor.load_weights(code_predictor_weights)
            talker_code_predictor_loaded = add_prefix_to_loaded_weights(
                talker_code_predictor_loaded, "code_predictor"
            )
            loaded_weights.update(talker_code_predictor_loaded)
        
        # Load code2wav weights
        if self.code2wav and code2wav_weights:
            code2wav_loaded = self.code2wav.load_weights(code2wav_weights)
            code2wav_loaded = add_prefix_to_loaded_weights(
                code2wav_loaded, "code2wav"
            )
            loaded_weights.update(code2wav_loaded)
        
        # Log summary
        logger.info(
            "Loaded %d weights for Qwen3OmniMoe (stage=%s)",
            len(loaded_weights),
            self.model_stage,
        )
        
        return loaded_weights

