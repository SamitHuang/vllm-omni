# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
cache-dit integration adapter for vllm-omni.

This module provides functions to enable cache-dit acceleration on diffusion
pipelines in vllm-omni, supporting both single and dual-transformer architectures.
"""

import logging
from typing import Any, Callable

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

try:
    import cache_dit
    from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, ParamsModifier, TaylorSeerCalibratorConfig

    CACHE_DIT_AVAILABLE = True
except ImportError:
    CACHE_DIT_AVAILABLE = False
    logger.warning("cache-dit is not installed. Cache-dit acceleration will not be available.")


# Special forward patterns for dual-transformer models
SPECIAL_FORWARD_PATTERNS: dict[str, Callable] = {}


def enable_cache_dit_for_wan22(pipeline: Any, od_config: OmniDiffusionConfig) -> None:
    """Enable cache-dit for Wan2.2 dual-transformer architecture.

    Wan2.2 uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Wan2.2 pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.
    """

    cache_config = od_config.cache_config

    # Parse cache config
    num_inference_steps = cache_config.get("num_inference_steps")

    # Build DBCacheConfig for primary transformer
    primary_cache_config = DBCacheConfig(
        num_inference_steps=num_inference_steps,
        Fn_compute_blocks=cache_config.get("Fn_compute_blocks", 1),
        Bn_compute_blocks=cache_config.get("Bn_compute_blocks", 0),
        max_warmup_steps=cache_config.get("max_warmup_steps", 8),
        max_cached_steps=cache_config.get("max_cached_steps", -1),
        max_continuous_cached_steps=cache_config.get("max_continuous_cached_steps", -1),
        residual_diff_threshold=cache_config.get("residual_diff_threshold", 0.08),
    )

    # FIXME: secondary cache shares the same config with primary cache for now, but we should support different config for secondary transformer in the future
    # Build DBCacheConfig for secondary transformer (can use same or different config)
    secondary_cache_config = DBCacheConfig(
        num_inference_steps=num_inference_steps,
        Fn_compute_blocks=cache_config.get("Fn_compute_blocks", 1),
        Bn_compute_blocks=cache_config.get("Bn_compute_blocks", 0),
        max_warmup_steps=cache_config.get("max_warmup_steps", 8),
        max_cached_steps=cache_config.get("max_cached_steps", -1),
        max_continuous_cached_steps=cache_config.get("max_continuous_cached_steps", -1),
        residual_diff_threshold=cache_config.get("residual_diff_threshold", 0.08),
    )

    # Build calibrator configs if TaylorSeer is enabled
    primary_calibrator = None
    secondary_calibrator = None
    if cache_config.get("enable_taylorseer", False):
        taylorseer_order = cache_config.get("taylorseer_order", 1)
        primary_calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        secondary_calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Build ParamsModifier for each transformer
    primary_modifier = ParamsModifier(
        cache_config=primary_cache_config,
        calibrator_config=primary_calibrator,
    )
    secondary_modifier = ParamsModifier(
        cache_config=secondary_cache_config,
        calibrator_config=secondary_calibrator,
    )

    logger.info(
        "Enabling cache-dit on Wan2.2 dual transformers with BlockAdapter: "
        f"Fn={primary_cache_config.Fn_compute_blocks}, "
        f"Bn={primary_cache_config.Bn_compute_blocks}, "
        f"W={primary_cache_config.max_warmup_steps}, "
    )

    transformer = pipeline.transformer
    transformer_2 = pipeline.transformer_2
    transformer_blocks = transformer.blocks
    transformer_2_blocks = transformer_2.blocks
    # Enable cache-dit using BlockAdapter for both transformers simultaneously
    cache_dit.enable_cache(
        BlockAdapter(
            transformer=[transformer, transformer_2],
            blocks=[transformer_blocks, transformer_2_blocks],
            forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
            params_modifiers=[primary_modifier, secondary_modifier],
            has_separate_cfg=True,
        ),
    )


def enable_cache_dit_for_flux(pipeline: Any, od_config: OmniDiffusionConfig) -> None:
    """Enable cache-dit for Flux dual-transformer architecture.

    Flux uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Flux pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.
    """

    raise NotImplementedError("cache-dit is not implemented for Flux pipeline.")
    
# Register special forward patterns
def may_enable_cache_dit(pipeline: Any, od_config: OmniDiffusionConfig) -> None:
    """Enable cache-dit on the pipeline if configured.

    This function checks if cache-dit is enabled in the config and applies it
    to the appropriate transformer(s) in the pipeline. It handles both
    single-transformer and dual-transformer architectures.

    It also stores the cached num_inference_steps on the pipeline for later
    validation during inference.

    Args:
        pipeline: The diffusion pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.

    Raises:
        ValueError: If cache_config is missing required parameters.
        ImportError: If cache-dit is not installed.
    """
    cache_config_dict = od_config.cache_config

    num_inference_steps = cache_config_dict.get("num_inference_steps", 50)

    # Store the cached num_inference_steps on the pipeline for later validation
    # This allows us to warn if num_inference_steps changes during inference
    pipeline._cache_dit_num_inference_steps = num_inference_steps

    # Check if this is a special dual-transformer model
    model_class_name = od_config.model_class_name
    if model_class_name in SPECIAL_FORWARD_PATTERNS:
        logger.info(f"Detected special dual-transformer model: {model_class_name}")
        SPECIAL_FORWARD_PATTERNS[model_class_name](pipeline, od_config)
        return

    # Build DBCacheConfig
    cache_config = DBCacheConfig(
        num_inference_steps=num_inference_steps,
        Fn_compute_blocks=cache_config_dict.get("Fn_compute_blocks", 1),
        Bn_compute_blocks=cache_config_dict.get("Bn_compute_blocks", 0),
        max_warmup_steps=cache_config_dict.get("max_warmup_steps", 8),
        max_cached_steps=cache_config_dict.get("max_cached_steps", -1),
        max_continuous_cached_steps=cache_config_dict.get("max_continuous_cached_steps", -1),
        residual_diff_threshold=cache_config_dict.get("residual_diff_threshold", 0.08),
    )

    # Build calibrator config if TaylorSeer is enabled
    calibrator_config = None
    if cache_config_dict.get("enable_taylorseer", False):
        taylorseer_order = cache_config_dict.get("taylorseer_order", 1)
        calibrator_config = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    logger.info(
        f"Enabling cache-dit on {model_class_name} transformer: "
        f"Fn={cache_config.Fn_compute_blocks}, "
        f"Bn={cache_config.Bn_compute_blocks}, "
        f"W={cache_config.max_warmup_steps}, "
        f"steps={num_inference_steps}, "
    )

    # Enable cache-dit on the transformer
    cache_dit.enable_cache(
        pipeline.transformer,
        cache_config=cache_config,
        calibrator_config=calibrator_config,
    )

