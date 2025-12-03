# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

# Try to import all sageattention variants
try:
    from sageattention import sageattn
    _SAGE_ATTN_AVAILABLE = True
except ImportError:
    logger.warning(
        "SageAttentionBackend is not available. You may install sageattention by running `uv pip install sageattention==2.2.0 --no-build-isolation`"
    )
    sageattn = None
    _SAGE_ATTN_AVAILABLE = False

# Try to import quantized variants
try:
    from sageattention import (
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp16_triton,
    )
    _SAGE_QUANTIZED_AVAILABLE = True
except ImportError:
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp16_triton = None
    _SAGE_QUANTIZED_AVAILABLE = False


def _get_gpu_compute_capability() -> tuple[int, int]:
    """Get GPU compute capability (major, minor)."""
    if not torch.cuda.is_available():
        return (0, 0)
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return capability


def _select_quantized_sage_function():
    """
    Select the best quantized SageAttention function based on GPU capability.
    
    Returns:
        The best available quantized sageattention function, or None if not available.
    """
    if not _SAGE_QUANTIZED_AVAILABLE:
        return None
    
    major, minor = _get_gpu_compute_capability()
    sm_version = major * 10 + minor
    
    # SM90 (H100) - prefer FP8 variants
    if sm_version >= 90:
        if sageattn_qk_int8_pv_fp8_cuda_sm90 is not None:
            # logger.info("Using SageAttention INT8 QK + FP8 PV (SM90) for GPU compute capability %d.%d", major, minor)
            return sageattn_qk_int8_pv_fp8_cuda_sm90
        elif sageattn_qk_int8_pv_fp8_cuda is not None:
            # logger.info("Using SageAttention INT8 QK + FP8 PV (CUDA) for GPU compute capability %d.%d", major, minor)
            return sageattn_qk_int8_pv_fp8_cuda
    
    # SM80+ (A100, etc.) - prefer CUDA variants
    if sm_version >= 80:
        if sageattn_qk_int8_pv_fp16_cuda is not None:
            # logger.info("Using SageAttention INT8 QK + FP16 PV (CUDA) for GPU compute capability %d.%d", major, minor)
            return sageattn_qk_int8_pv_fp16_cuda
        elif sageattn_qk_int8_pv_fp16_triton is not None:
            # logger.info("Using SageAttention INT8 QK + FP16 PV (Triton) for GPU compute capability %d.%d", major, minor)
            return sageattn_qk_int8_pv_fp16_triton
    
    return None


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # SageAttention supports head sizes: 32, 64, 96, 128, 160, 192, 224, 256
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        # Not used in current implementation
        return None


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        
        if not _SAGE_ATTN_AVAILABLE:
            raise ImportError(
                "sageattention is not installed. Please install it by running: pip install sageattention"
            )
        
        # Check if quantization is requested via environment variable
        # Options: "auto", "int8_fp8", "int8_fp16", "none", or specific variant names
        quant_mode = os.environ.get("SAGE_ATTN_QUANT_MODE", "auto").lower()
        
        # Select the attention function to use
        if quant_mode == "none":
            self.sage_fn = sageattn
            self.quantized = False
        elif quant_mode == "auto":
            # Try to use quantized version if available, fallback to regular
            quantized_fn = _select_quantized_sage_function()
            if quantized_fn is not None:
                self.sage_fn = quantized_fn
                self.quantized = True
            else:
                self.sage_fn = sageattn
                self.quantized = False
                logger.info("Quantized SageAttention not available, using regular SageAttention")
        elif quant_mode == "int8_fp8":
            # Prefer SM90 variant if available
            major, minor = _get_gpu_compute_capability()
            sm_version = major * 10 + minor
            if sm_version >= 90 and sageattn_qk_int8_pv_fp8_cuda_sm90 is not None:
                self.sage_fn = sageattn_qk_int8_pv_fp8_cuda_sm90
            elif sageattn_qk_int8_pv_fp8_cuda is not None:
                self.sage_fn = sageattn_qk_int8_pv_fp8_cuda
            else:
                raise ValueError("INT8 QK + FP8 PV variant not available")
            self.quantized = True
        elif quant_mode == "int8_fp16":
            # Prefer CUDA variant
            if sageattn_qk_int8_pv_fp16_cuda is not None:
                self.sage_fn = sageattn_qk_int8_pv_fp16_cuda
            elif sageattn_qk_int8_pv_fp16_triton is not None:
                self.sage_fn = sageattn_qk_int8_pv_fp16_triton
            else:
                raise ValueError("INT8 QK + FP16 PV variant not available")
            self.quantized = True
        else:
            # Try to match specific variant name
            variant_map = {
                "int8_fp8_cuda": sageattn_qk_int8_pv_fp8_cuda,
                "int8_fp8_cuda_sm90": sageattn_qk_int8_pv_fp8_cuda_sm90,
                "int8_fp16_cuda": sageattn_qk_int8_pv_fp16_cuda,
                "int8_fp16_triton": sageattn_qk_int8_pv_fp16_triton,
            }
            if quant_mode in variant_map and variant_map[quant_mode] is not None:
                self.sage_fn = variant_map[quant_mode]
                self.quantized = True
            else:
                logger.warning(f"Unknown quant_mode '{quant_mode}', falling back to regular SageAttention")
                self.sage_fn = sageattn
                self.quantized = False
        

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # SageAttention expects input in NHD format (batch_size, seq_len, num_heads, head_size)
        # which matches the input format from vllm-omni
        out: torch.Tensor = self.sage_fn(
            q=query,
            k=key,
            v=value,
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return out

