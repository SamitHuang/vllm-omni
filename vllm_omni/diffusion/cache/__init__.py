# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Cache module for diffusion model inference acceleration.

This module provides a unified cache backend system for different caching strategies:
- TeaCache: Timestep Embedding Aware Cache for adaptive transformer caching
- (Future: DeepCache, DistriFusion, etc.)

The cache backend system uses a registry pattern where cache types can be
dynamically registered and instantiated. Cache is configured via OmniDiffusionConfig
or DIFFUSION_CACHE_BACKEND environment variable.
"""

from vllm_omni.diffusion.cache.apply import setup_cache
from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.registry import (
    CacheType,
    get_cache_backend,
    register_cache_backend,
)
from vllm_omni.diffusion.cache.teacache import (
    CacheContext,
    TeaCacheConfig,
    apply_teacache_hook,
)

# Import teacache backend to trigger registration
from vllm_omni.diffusion.cache.teacache.adapter import TeaCacheBackend  # noqa: F401

__all__ = [
    "CacheBackend",
    "CacheType",
    "TeaCacheConfig",
    "CacheContext",
    "setup_cache",
    "get_cache_backend",
    "register_cache_backend",
    "apply_teacache_hook",
]
