# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache backend implementation.

This module provides the TeaCache backend that implements the CacheBackend
interface using the hooks-based TeaCache system.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.registry import CacheType, register_cache_backend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook

logger = init_logger(__name__)


class TeaCacheBackend(CacheBackend):
    """
    TeaCache implementation using hooks.

    TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique
    that speeds up diffusion inference by reusing transformer block computations
    when consecutive timestep embeddings are similar.

    The backend applies TeaCache hooks to the transformer which intercept the
    forward pass and implement the caching logic transparently.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.2))
        >>> backend.apply(pipeline)
        >>> # Generate with cache enabled
        >>> backend.reset(pipeline.transformer)  # Reset before each generation
        >>> # Access config attributes: backend.config.rel_l1_thresh
    """

    def apply(self, pipeline: Any) -> None:
        """
        Apply TeaCache to transformer using hooks.

        This creates a TeaCacheConfig from the backend's DiffusionCacheConfig
        and applies the TeaCache hook to the transformer.

        Args:
            pipeline: Diffusion pipeline instance. Extracts transformer and model_type:
                     - transformer: pipeline.transformer
                     - model_type: pipeline.__class__.__name__
        """
        # Extract transformer and model_type from pipeline
        transformer = pipeline.transformer
        model_type = pipeline.__class__.__name__

        # Create TeaCacheConfig from DiffusionCacheConfig with model_type
        # Access parameters via attribute access: config.rel_l1_thresh
        # rel_l1_thresh already has a default value of 0.2 in DiffusionCacheConfig
        try:
            teacache_config = TeaCacheConfig(
                model_type=model_type,
                rel_l1_thresh=self.config.rel_l1_thresh,
                coefficients=self.config.coefficients,
            )
        except Exception as e:
            logger.error(f"Failed to create TeaCacheConfig: {e}")
            raise ValueError(
                f"Invalid TeaCache configuration: {e}. "
                f"Expected keys: rel_l1_thresh, coefficients (optional). "
                f"model_type is automatically extracted from pipeline.__class__.__name__."
            )

        # Apply hook to transformer
        apply_teacache_hook(transformer, teacache_config)
        logger.info(
            f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
            f"model_type={teacache_config.model_type}"
        )

    def reset(self, transformer: torch.nn.Module) -> None:
        """
        Reset TeaCache state for new generation.

        Clears all cached residuals and resets counters/accumulators.
        Should be called before each generation to ensure clean state.

        Args:
            transformer: Transformer module to reset cache on
        """
        if hasattr(transformer, "_hook_registry"):
            hook = transformer._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
            if hook is not None:
                transformer._hook_registry.reset_hook(TeaCacheHook._HOOK_NAME)
                logger.debug("TeaCache state reset")
            else:
                logger.warning("TeaCache hook not found, nothing to reset")
        else:
            logger.warning("Transformer has no hook registry, TeaCache may not be applied")


# Register TeaCache backend in the global registry
register_cache_backend(CacheType.TEA_CACHE, TeaCacheBackend)
