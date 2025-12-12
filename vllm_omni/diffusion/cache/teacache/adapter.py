# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache adapter implementation.

This module provides the TeaCache adapter that implements the CacheAdapter
interface using the hooks-based TeaCache system.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheAdapter
from vllm_omni.diffusion.cache.registry import CacheType, register_cache_adapter
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook

logger = init_logger(__name__)


class TeaCacheAdapter(CacheAdapter):
    """
    TeaCache implementation using hooks.

    TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique
    that speeds up diffusion inference by reusing transformer block computations
    when consecutive timestep embeddings are similar.

    The adapter applies TeaCache hooks to the transformer which intercept the
    forward pass and implement the caching logic transparently.

    Example:
        >>> adapter = TeaCacheAdapter({"rel_l1_thresh": 0.2})
        >>> adapter.apply(pipeline)
        >>> # Generate with cache enabled
        >>> adapter.reset(pipeline.transformer)  # Reset before each generation
    """

    def apply(self, pipeline: Any) -> None:
        """
        Apply TeaCache to transformer using hooks.

        This creates a TeaCacheConfig from the adapter's config dict and
        applies the TeaCache hook to the transformer.

        Args:
            pipeline: Diffusion pipeline instance. Extracts transformer and model_type:
                     - transformer: pipeline.transformer
                     - model_type: pipeline.__class__.__name__
        """
        # Extract transformer and model_type from pipeline
        transformer = pipeline.transformer
        model_type = pipeline.__class__.__name__

        # Remove model_type from config if present (shouldn't be there anymore)
        config_without_model_type = {k: v for k, v in self.config.items() if k != "model_type"}

        # Create TeaCacheConfig from dict with model_type
        try:
            teacache_config = TeaCacheConfig(model_type=model_type, **config_without_model_type)
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


# Register TeaCache adapter in the global registry
register_cache_adapter(CacheType.TEA_CACHE, TeaCacheAdapter)
