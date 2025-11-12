"""Qwen3-Omni MoE Code Predictor with MTP (Multi-Token Prediction) support.

This module implements the code predictor component for Qwen3-Omni MoE models
following the DeepSeek MTP pattern for vLLM speculative decoding compatibility.

The code predictor generates residual RVQ (Residual Vector Quantization) codes
autoregressively, predicting layers 1 to N based on layer-0 codes from the talker.
"""

from collections.abc import Iterable
from typing import Any, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionType
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP, SupportsMultiModal
from .qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)
from .qwen3_omni import Qwen3OmniMoeConditionalGenerationMixin
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


# ============================================================================
# Rotary Embeddings and Helper Functions
# ============================================================================

class Qwen3OmniCodePredictorRotaryEmbedding(nn.Module):
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
    """Helper function for rotary embeddings."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: [bsz, num_heads, seq_len, head_dim]
        k: [bsz, num_key_value_heads, seq_len, head_dim]
        cos: [num_tokens, rotary_dim] where num_tokens = bsz * seq_len OR just seq_len
        sin: [num_tokens, rotary_dim] where num_tokens = bsz * seq_len OR just seq_len
    
    Returns:
        q_rot, k_rot with RoPE applied
    """
    bsz, _, seq_len, head_dim = q.shape
    
    # Reshape cos/sin based on their current shape
    if cos.shape[0] == seq_len:
        # cos/sin is [seq_len, rotary_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rotary_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rotary_dim]
    else:
        # cos/sin is [bsz * seq_len, rotary_dim] - need to reshape
        cos = cos.view(bsz, seq_len, -1).unsqueeze(1)  # [bsz, 1, seq_len, rotary_dim]
        sin = sin.view(bsz, seq_len, -1).unsqueeze(1)  # [bsz, 1, seq_len, rotary_dim]
    
    # Ensure rotary_dim matches head_dim (may need to pad or slice)
    rotary_dim = cos.shape[-1]
    if rotary_dim < head_dim:
        # Partial rotary - only apply to first rotary_dim dimensions
        q_rot = q[..., :rotary_dim]
        k_rot = k[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]
        k_pass = k[..., rotary_dim:]
        
        q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
    else:
        # Full rotary
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
    
    return q, k


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped query attention."""
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


# ============================================================================
# Code Predictor Attention Layer
# ============================================================================

class Qwen3OmniCodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor with vLLM optimization."""

    def __init__(self, config, layer_idx: int, vllm_config: VllmConfig = None):
        super().__init__()

        self.num_heads = config.code_predictor_config.num_attention_heads
        self.num_key_value_heads = config.code_predictor_config.num_key_value_heads
        self.head_dim = getattr(config.code_predictor_config, "head_dim",
                                config.code_predictor_config.hidden_size // config.code_predictor_config.num_attention_heads)
        self.hidden_size = config.code_predictor_config.hidden_size
        
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        scale = 1.0 / math.sqrt(self.head_dim)

        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Query/Key normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q_heads = q.transpose(1, 2).contiguous()
        k_heads = k.transpose(1, 2).contiguous()
        q_heads, k_heads = apply_rotary_pos_emb(q_heads, k_heads, cos, sin)
        v_heads = v.transpose(1, 2).contiguous()

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


# ============================================================================
# Code Predictor MLP Layer
# ============================================================================

class Qwen3OmniCodePredictorMLP(nn.Module):
    """Feed-forward network for code predictor."""

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

# ============================================================================
# MTP Layer (Multi-Token Prediction Layer)
# ============================================================================

class Qwen3OmniCodePredictorMTPLayer(nn.Module):
    """MTP layer for speculative decoding - predicts next residual code layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Qwen3OmniCodePredictorDecoderLayer
        self.self_attn = Qwen3OmniCodePredictorAttention(
            config, layer_idx,
            vllm_config=type('VllmConfig', (), {
                'cache_config': cache_config,
                'quant_config': quant_config,
                'model_config': model_config
            })())
        self.mlp = Qwen3OmniCodePredictorMLP(config)
        self.input_layernorm = RMSNorm(config.code_predictor_config.hidden_size,
                                       eps=config.code_predictor_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.code_predictor_config.hidden_size,
                                                eps=config.code_predictor_config.rms_norm_eps)

    def mtp_block(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_mask, cos, sin)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None, "inputs_embeds required for MTP"
        
        # Mask position 0 (not needed for MTP)
        inputs_embeds[positions == 0] = 0

        hidden_states = torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        
        # Get position info for RoPE
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get RoPE embeddings
        head_dim = self.self_attn.head_dim
        rotary_emb = Qwen3OmniCodePredictorRotaryEmbedding(head_dim,
                                                           max_position_embeddings=self.config.code_predictor_config.max_position_embeddings)
        cos, sin = rotary_emb(hidden_states, position_ids)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        
        # Forward through MTP block
        hidden_states = self.mtp_block(hidden_states, causal_mask, cos, sin)
        
        return hidden_states


# ============================================================================
# Base Code Predictor Model (matches HF structure)
# ============================================================================

class Qwen3OmniCodePredictorBaseModel(nn.Module):
    """
    Base model for code predictor - matches HF Qwen3OmniMoeTalkerCodePredictorModel structure.
    
    This is a simple transformer that processes inputs_embeds and outputs hidden states.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.code_predictor_config
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups
        
        # Codec embeddings (for layers 1-num_code_groups-1)
        self.codec_embedding = nn.ModuleList([
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
            for _ in range(config.num_code_groups - 1)
        ])
        
        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3OmniCodePredictorMTPLayer(
                vllm_config.model_config.hf_config,
                f"{prefix}.layers.{idx}",
                model_config=vllm_config.model_config,
                layer_idx=idx,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for idx in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # RoPE
        from vllm.model_executor.layers.rotary_embedding import get_rope
        self.rotary_emb = get_rope(
            config.head_dim,
            rotary_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass matching HF structure.
        
        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            
        Returns:
            Object with .last_hidden_state attribute
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        
        # Create positions tensor if not provided
        # positions must be [num_tokens] or [batch_size, seq_len]
        if position_ids is None:
            if cache_position is not None:
                positions = cache_position  # [num_tokens]
            else:
                positions = torch.arange(seq_len, device=inputs_embeds.device)  # [seq_len]
        else:
            positions = position_ids.flatten()  # Ensure [num_tokens]
        
        # Extract cos/sin from rotary_emb cache
        # The cos_sin_cache is [max_pos, rotary_dim * 2]
        cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, positions)  # [num_tokens, rotary_dim * 2]
        cos, sin = cos_sin.chunk(2, dim=-1)  # Each [num_tokens, rotary_dim]
        
        # Create causal mask
        device = inputs_embeds.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        
        # Forward through decoder layers
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer.mtp_block(hidden_states, causal_mask, cos, sin)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Return in HF-compatible format
        from collections import namedtuple
        Output = namedtuple('Output', ['last_hidden_state', 'past_key_values'])
        return Output(last_hidden_state=hidden_states, past_key_values=None)

    def get_input_embeddings(self):
        """Return codec embeddings for HF compatibility."""
        return self.codec_embedding


# ============================================================================
# Multi-Token Predictor (MTP Module) - DEPRECATED, kept for reference
# ============================================================================

class Qwen3OmniCodePredictorMTP(nn.Module):
    """Multi-token predictor for speculative decoding - DEPRECATED."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        
        # MTP configuration
        self.mtp_start_layer_idx = 0  # Code predictor layers start at 0
        self.num_mtp_layers = config.code_predictor_config.num_hidden_layers
        
        # Codec embeddings
        self.codec_embedding = nn.ModuleList([
            VocabParallelEmbedding(
                config.code_predictor_config.vocab_size,
                config.code_predictor_config.hidden_size,
            )
            for _ in range(config.code_predictor_config.num_code_groups - 1)
        ])
        
        # MTP layers (one per residual code layer to predict)
        self.layers = torch.nn.ModuleDict({
            str(idx): Qwen3OmniCodePredictorMTPLayer(
                config,
                f"{prefix}.layers.{idx}",
                model_config=vllm_config.model_config,
                layer_idx=idx,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for idx in range(self.num_mtp_layers)
        })

        self.norm = RMSNorm(config.code_predictor_config.hidden_size,
                               eps=config.code_predictor_config.rms_norm_eps)

        self.logits_processor = LogitsProcessor(config.code_predictor_config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass for MTP prediction."""
        if inputs_embeds is None:
            # Embed the current code using appropriate embedding layer
            embed_layer_idx = min(spec_step_idx, len(self.embed_tokens) - 1)
            inputs_embeds = self.embed_tokens[embed_layer_idx](input_ids)
        
        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states = self.layers[str(current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

        # Final normalization
        if current_step_idx == self.num_mtp_layers:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Compute logits for the current MTP step."""
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(current_step_idx)]
        
        # Apply shared head normalization and projection
        hidden_states = mtp_layer.shared_head['norm'](hidden_states)
        logits = self.logits_processor(
            mtp_layer.shared_head['head'],
            hidden_states,
            sampling_metadata
        )
        return logits

class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """
    Code predictor wrapper matching HF structure.
    
    Structure:
    - self.model: Qwen3OmniCodePredictorBaseModel (transformer)
    - self.lm_head: ModuleList of output heads
    """
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        talker_code_predictor_config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_code_predictor_config
        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        # Base transformer model (matches HF structure)
        self.model = Qwen3OmniCodePredictorBaseModel(
            vllm_config=vllm_config,
            prefix=prefix
        )

        # Output heads for each residual layer (1-num_layers-1)
        self.lm_head = nn.ModuleList([
            nn.Linear(self.config.code_predictor_config.hidden_size, self.config.code_predictor_config.vocab_size, bias=False)
            for _ in range(self.num_code_groups - 1)
        ])

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

# ============================================================================
# Top-Level Code Predictor MTP Model
# ============================================================================

@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    """
    Code Predictor for generating residual codec layers 1-num_layers-1.
    
    Matches HuggingFace structure with self.code_predictor separation.
    This allows direct weight loading from HuggingFace checkpoints.
    
    Architecture:
    - code_predictor.model: Base transformer model
    - code_predictor.lm_head: ModuleList of output heads
    """
    logger = init_logger(__name__)

    # Weight mapping from HuggingFace to vLLM naming convention
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "talker.code_predictor.lm_head.": "code_predictor.lm_head.",
            "talker.code_predictor.model.": "code_predictor.model.",
            "talker.code_predictor": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        talker_code_predictor_config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_code_predictor_config
        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        # Create the MTP code predictor
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "code_predictor.model")
        )
        
        # Projection for talker hidden states (2048) -> code predictor dimension (1024)
        if (hasattr(self.config, 'text_config') and 
            self.config.text_config.hidden_size != self.config.code_predictor_config.hidden_size):
            self.hidden_projection = nn.Linear(
                self.config.text_config.hidden_size,
                self.config.code_predictor_config.hidden_size,
                bias=False
            )
        else:
            self.hidden_projection = None

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

        # Generate full codec codes using MTP
        # This will be the parallel prediction implementation
        batch_size, seq_len = input_ids.shape
        
        # Project hidden states if needed
        if self.hidden_projection is not None:
            inputs_embeds = self.hidden_projection(inputs_embeds)
        
        # For now, use sequential generation (TODO: implement parallel)
        # Result will be [batch, num_code_groups, seq_len]
        # - all_codes_per_position will collect [batch, num_code_groups, 1] for each position
        all_codes_per_position = []
        
        # Generate residual layers for each position
        for pos in range(seq_len):
            layer0_code = input_ids[:, pos:pos+1]  # [batch, 1]
            hidden = inputs_embeds[:, pos:pos+1, :]  # [batch, 1, hidden_size]
            
            # Embed layer 0
            layer0_embed = self.code_predictor.model.codec_embedding[0](layer0_code)  # [batch, 1, hidden_size]
            
            # Combine
            mtp_input = torch.cat([hidden, layer0_embed], dim=1)  # [batch, 2, hidden_size]
            
            # Forward through code_predictor model to get hidden states
            outputs = self.code_predictor.model(
                inputs_embeds=mtp_input,  # Combined hidden + layer0 embedding
                attention_mask=None,  # Will create causal mask internally
                position_ids=None,  # Will create from cache_position
                past_key_values=None,  # No kv cache for now
                use_cache=False,
                cache_position=None,
            )
            
            hidden_state = outputs.last_hidden_state  # [batch, 2, hidden_size]
            
            # Predict all residual layers (layers 1 to num_code_groups-1)
            pos_codes = [layer0_code]  # Start with layer 0: [batch, 1]
            for layer_idx in range(self.num_code_groups - 1):
                # Use the corresponding lm_head for this layer
                logits = self.code_predictor.lm_head[layer_idx](hidden_state[:, -1:, :])  # [batch, 1, vocab_size]
                
                # Sample
                probs = F.softmax(logits / temperature, dim=-1)
                code = torch.multinomial(probs.squeeze(1), num_samples=1)  # [batch, 1]
                pos_codes.append(code)
            
            # Stack all layers for this position: [batch, num_code_groups, 1]
            pos_all_layers = torch.stack(pos_codes, dim=1)  # [batch, num_code_groups, 1]
            all_codes_per_position.append(pos_all_layers)
        
        # Concatenate across positions: [batch, num_code_groups, seq_len]
        result = torch.cat(all_codes_per_position, dim=2)
        
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the code predictor model.

        The weight mapping translates from HuggingFace naming convention
        to vLLM's internal structure.
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "code2wav.", "talker.text_projection.",
                           "talker.hidden_projection.", "talker.model.", "talker.codec_head"]
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
                self.__class__.__name__, True, total_bytes / (1024 ** 2), str(device)
            )
        except Exception:
            pass

        # Add multimodal weights (for API compatibility)
        multi_model_weights = set()
        if hasattr(self, 'visual'):
            for name, param in self.visual.named_parameters():
                multi_model_weights.add("visual." + name)
        if hasattr(self, 'audio_tower'):
            for name, param in self.audio_tower.named_parameters():
                multi_model_weights.add("audio_tower." + name)
        loaded.update(multi_model_weights)

        return loaded

    def init_multi_modal(self, thinker_config):
        """
        Initialize multimodal components from the thinker.
        
        For API compatibility with the talker. The code predictor doesn't
        actually process multimodal inputs directly.
        """
        from .qwen3_omni_moe_thinker import (
            Qwen3OmniMoeAudioEncoder,
            Qwen3Omni_VisionTransformer,
        )
        
        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)
        self.visual = Qwen3Omni_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=self.quant_config,
            prefix=maybe_prefix(self.prefix, "visual"),
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        """For API compatibility - code predictor doesn't use multimodal inputs."""
        return {}

    def _process_multimodal_inputs(self, *args, **kwargs):
        """For API compatibility - code predictor doesn't process multimodal inputs."""
        return None

