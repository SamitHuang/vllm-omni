# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe Code2Wav model."""

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeCode2WavTransformerModel,
        )

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)


logger = init_logger(__name__)


# ============================================================================
# Convolutional Blocks
# ============================================================================


class Qwen3OmniMoeCausalConvNet(nn.Module):
    """Causal 1D convolution with automatic padding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        """Calculate extra padding needed for causal convolution."""
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (
            self.kernel_size - self.padding
        )
        return ideal_length - length

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(
            hidden_state, (self.padding, extra_padding), mode="constant", value=0
        )
        return self.conv(hidden_state).contiguous()


class Qwen3OmniMoeCausalTransConvNet(nn.Module):
    """Causal transposed 1D convolution for upsampling."""
    
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride
        )

        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = self.left_pad

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = hidden_state[
            ..., self.left_pad : hidden_state.shape[-1] - self.right_pad
        ]
        return hidden_state.contiguous()


class Qwen3OmniMoeConvNeXtBlock(nn.Module):
    """ConvNeXt block for processing audio features."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3OmniMoeCausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,  # Depthwise
            dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_states = hidden_states

        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)

        hidden_states = self.gamma * hidden_states

        hidden_states = hidden_states.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]

        hidden_states = input_states + hidden_states

        return hidden_states


# ============================================================================
# Activation Functions
# ============================================================================


class SnakeBeta(nn.Module):
    """
    A modified Snake activation function with learnable parameters.
    
    SnakeBeta(x) = x + 1/β * sin²(x*α)
    
    Better for audio signals than ReLU/GELU due to periodic components.
    
    References:
        - https://huggingface.co/papers/2006.08195
    """
    
    def __init__(self, in_features: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features

        # Initialize learnable parameters
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SnakeBeta(x) = x + 1/β * sin²(x*α)
        
        Args:
            hidden_states: [B, C, T]
        
        Returns:
            Activated hidden states [B, C, T]
        """
        # Reshape to [B, C, T]
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states


# ============================================================================
# Decoder Blocks
# ============================================================================


class Qwen3OmniMoeCode2WavDecoderResidualUnit(nn.Module):
    """Residual unit with dilated causal convolutions."""
    
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3OmniMoeCausalConvNet(
            dim, dim, kernel_size=7, dilation=dilation
        )
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state

        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        
        return hidden_state + residual


class Qwen3OmniMoeCode2WavDecoderBlock(nn.Module):
    """
    Decoder block with progressive upsampling and residual units.
    
    Each block:
    1. Upsamples by a factor (e.g., 8x, 5x, 4x, 2x)
    2. Reduces channels by 2x
    3. Applies 3 residual units with different dilations
    """
    
    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx: int):
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block = [
            SnakeBeta(in_dim),
            Qwen3OmniMoeCausalTransConvNet(
                in_dim, out_dim, 2 * upsample_rate, upsample_rate
            ),
        ]

        # Add residual units with different dilations for multi-scale patterns
        for dilation in (1, 3, 9):
            block.append(Qwen3OmniMoeCode2WavDecoderResidualUnit(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            hidden = layer(hidden)
        return hidden


# ============================================================================
# Main Code2Wav Model
# ============================================================================


class Qwen3OmniMoeCode2Wav(nn.Module):
    """
    Qwen3 Omni MoE Code2Wav - Converts num_quantizers-layer RVQ codec codes to audio waveform.
    
    Architecture:
    1. Code Embedding: Embed and average num_quantizers RVQ layers
    2. Pre-Transformer: Add temporal context via sliding-window attention
    3. Upsampling: Progressive upsampling with ConvNeXt blocks
    4. Decoder: Multi-stage upsampling + residual units → waveform
    
    Input: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
    Output: [batch, 1, waveform_len] - Audio waveform [-1, 1]
    
    Total upsampling factor: ~1280x
    Example: 100 codec frames → 128,000 audio samples (8 seconds at 16kHz)
    """
    
    input_modalities = "audio"

    # Weight mapper
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "code2wav.pre_transformer.": "pre_transformer.",
            "code2wav.code_embedding.": "code_embedding.",
            "code2wav.upsample.": "upsample.",
            "code2wav.decoder.": "decoder.",
            "code2wav.": "",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        code2wav_config: Qwen3OmniMoeCode2WavConfig = (
            vllm_config.model_config.hf_config
        )

        self.config = code2wav_config

        print(f"self.config {self.config}")

        # Calculate total upsampling factor
        self.total_upsample = np.prod(
            self.config.upsample_rates + self.config.upsampling_ratios
        )
        
        # Pre-transformer
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(
            self.config
        )
        
        # Code embedding: Single embedding table for all RVQ layers
        self.code_embedding = nn.Embedding(
            self.config.codebook_size * self.config.num_quantizers, self.config.hidden_size
        )
        
        # Offset for each RVQ layer (layer 0: 0-1023, layer 1: 1024-2047, etc.)
        self.register_buffer(
            "code_offset",
            torch.arange(self.config.num_quantizers).view(1, -1, 1) * self.config.codebook_size,
            persistent=False,
        )

        # Upsampling blocks (e.g., 2x, 2x)
        upsample = []
        for factor in self.config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(
                            self.config.hidden_size, self.config.hidden_size, factor, factor
                        ),
                        Qwen3OmniMoeConvNeXtBlock(self.config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        # Decoder: Initial projection + progressive upsampling blocks
        decoder = [
            Qwen3OmniMoeCausalConvNet(
                self.config.hidden_size, self.config.decoder_dim, kernel_size=7
            )
        ]
        
        # Add decoder blocks (each upsamples and reduces channels)
        for i in range(len(self.config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(self.config, i))
        
        # Final projection to waveform
        output_dim = self.config.decoder_dim // 2 ** len(self.config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, kernel_size=7),
        ]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert num_quantizers-layer RVQ codes to audio waveform.
        
        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codec codes
        
        Returns:
            waveform: [batch, 1, waveform_len] - Audio waveform clipped to [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layers of codes, "
                f"got {codes.shape[1]}"
            )
        
        # Stage 1: Code Embedding
        # Add offset to separate layer vocabularies, then embed and average
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        # Shape: [batch, seq_len, hidden_size]
        
        # Stage 2: Pre-Transformer (add temporal context)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        # Shape: [batch, seq_len, hidden_size]
        
        # Stage 3: Upsampling
        hidden = hidden.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        # Shape: [batch, hidden_size, seq_len * upsample_factor]
        
        # Stage 4: Decoder (progressive upsampling to waveform)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        # Shape: [batch, 1, waveform_len]
        
        # Clamp to valid audio range
        return wav.clamp(min=-1.0, max=1.0)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        """
        Decode long sequences in chunks to avoid OOM.
        
        Uses overlapping chunks with left context to avoid boundary artifacts.
        
        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            chunk_size: Number of codec frames per chunk
            left_context_size: Number of overlapping frames for context
        
        Returns:
            waveform: [batch, 1, waveform_len] - Complete waveform
        """
        wavs = []
        start_index = 0
        
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = (
                left_context_size if start_index >= left_context_size else start_index
            )
            
            # Extract chunk with left context
            codes_chunk = codes[..., start_index - context_size : end_index]
            
            # Decode chunk
            wav_chunk = self(codes_chunk)
            
            # Remove context from output (context_size * total_upsample samples)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            
            start_index = end_index
        
        return torch.cat(wavs, dim=-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "talker."],  # Already loaded above
        )
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
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            pass
        
        return loaded


