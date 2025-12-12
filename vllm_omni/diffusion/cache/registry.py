# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Cache backend registry for diffusion models.

This module provides a registry pattern for cache backends, allowing dynamic
registration and instantiation of different cache types (TeaCache, DeepCache, etc.).
"""

from enum import Enum
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


class CacheType(Enum):
    """Supported cache backend types."""

    NONE = "none"
    TEA_CACHE = "tea_cache"
    # Future cache types can be added here:
    # DEEP_CACHE = "deep_cache"
    # DISTRI_FUSION = "distri_fusion"


# Global registry mapping cache types to backend classes
CACHE_BACKEND_REGISTRY: dict[CacheType, type[CacheBackend]] = {}


def register_cache_backend(cache_type: CacheType, backend_class: type[CacheBackend]) -> None:
    """
    Register a cache backend class for a given cache type.

    This allows extending the system with new cache types without modifying
    the core cache infrastructure.

    Args:
        cache_type: CacheType enum value
        backend_class: CacheBackend subclass to register

    Example:
        >>> register_cache_backend(CacheType.TEA_CACHE, TeaCacheBackend)
    """
    if not issubclass(backend_class, CacheBackend):
        raise TypeError(f"{backend_class} must be a subclass of CacheBackend")

    CACHE_BACKEND_REGISTRY[cache_type] = backend_class
    logger.debug(f"Registered cache backend: {cache_type.value} -> {backend_class.__name__}")


def get_cache_backend(cache_type: str, config: DiffusionCacheConfig | dict[str, Any]) -> CacheBackend:
    """
    Factory function to get cache backend instance.

    Converts string cache type to enum, looks up in registry, and instantiates
    the appropriate backend class with the provided configuration.

    Args:
        cache_type: String name of cache type ("tea_cache", "deep_cache", etc.)
        config: DiffusionCacheConfig instance or dictionary to pass to backend constructor.
                If dict, will be converted to DiffusionCacheConfig.

    Returns:
        Instantiated CacheBackend subclass

    Raises:
        ValueError: If cache_type is unknown or not registered

    Example:
        >>> backend = get_cache_backend("tea_cache", DiffusionCacheConfig(rel_l1_thresh=0.2))
        >>> backend.apply(pipeline)
    """
    # Normalize cache type string
    cache_type_str = cache_type.lower().strip()

    # Convert string to enum
    try:
        cache_enum = CacheType(cache_type_str)
    except ValueError:
        available = [ct.value for ct in CacheType if ct != CacheType.NONE]
        raise ValueError(f"Unknown cache type: '{cache_type}'. Available types: {available}")

    # Check if it's the special "none" case
    if cache_enum == CacheType.NONE:
        raise ValueError("Cannot instantiate backend for cache_type='none'. Use setup_cache() which handles this case.")

    # Lookup in registry
    if cache_enum not in CACHE_BACKEND_REGISTRY:
        raise ValueError(
            f"Cache type '{cache_type}' is not registered. Registered types: {list(CACHE_BACKEND_REGISTRY.keys())}"
        )

    backend_class = CACHE_BACKEND_REGISTRY[cache_enum]

    # Convert dict to DiffusionCacheConfig if needed
    if isinstance(config, dict):
        config = DiffusionCacheConfig.from_dict(config)

    # Instantiate and return
    return backend_class(config)
