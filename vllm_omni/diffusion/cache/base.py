# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base cache backend interface for diffusion models.

This module defines the abstract base class that all cache backends must implement.
Cache backends provide a unified interface for applying different caching strategies
<<<<<<< HEAD
(TeaCache, DeepCache, etc.) to transformer models using hooks.
=======
(cache-dit, TeaCache, etc.) to diffusion pipelines.

All cache backends must implement:
- enable(pipeline): Enable cache on the pipeline
- refresh(pipeline, num_inference_steps): Refresh cache context when steps change
- is_enabled(): Check if cache is enabled
>>>>>>> bc91b0a (tmp save)
"""

from abc import ABC, abstractmethod
from typing import Any


<<<<<<< HEAD
from vllm_omni.diffusion.data import DiffusionCacheConfig


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Cache backends apply caching strategies to transformer models to accelerate
    inference. Each backend type (TeaCache, DeepCache, etc.) implements the
    apply() and reset() methods to manage cache lifecycle.

    Attributes:
        config: DiffusionCacheConfig instance containing cache-specific configuration parameters
    """

    def __init__(self, config: DiffusionCacheConfig):
=======
class BaseCacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Cache backends provide a unified interface for cache acceleration on diffusion
    pipelines. Each backend type (cache-dit, TeaCache, etc.) implements the
    enable(), refresh(), and is_enabled() methods to manage cache lifecycle.

    Attributes:
        cache_config: Cache configuration (dict or DiffusionCacheConfig instance).
        enabled: Whether cache is enabled on the pipeline.
    """

    def __init__(self, cache_config: Any = None):
>>>>>>> bc91b0a (tmp save)
        """
        Initialize cache backend with configuration.

        Args:
<<<<<<< HEAD
            config: DiffusionCacheConfig instance with cache-specific parameters
=======
            cache_config: Cache-specific configuration (dict or DiffusionCacheConfig instance).
>>>>>>> bc91b0a (tmp save)
        """
        self.cache_config = cache_config
        self.enabled = False

    @abstractmethod
<<<<<<< HEAD
    def apply(self, pipeline: Any) -> None:
=======
    def enable(self, pipeline: Any) -> None:
>>>>>>> bc91b0a (tmp save)
        """
        Enable cache on the pipeline.

        This method should apply the cache acceleration to the pipeline's transformer(s).
        Called once during pipeline initialization.

        Args:
<<<<<<< HEAD
            pipeline: Diffusion pipeline instance. The backend can extract:
                     - transformer: via pipeline.transformer
                     - model_type: via pipeline.__class__.__name__
=======
            pipeline: The diffusion pipeline instance.
>>>>>>> bc91b0a (tmp save)
        """
        raise NotImplementedError("Subclasses must implement enable()")

    @abstractmethod
    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """
        Refresh cache context with new num_inference_steps.

        This method should update the cache context when num_inference_steps changes
        during inference. For some backends, this may reset cache state.

        Args:
            pipeline: The diffusion pipeline instance.
            num_inference_steps: New number of inference steps.
            verbose: Whether to log refresh operations.
        """
        raise NotImplementedError("Subclasses must implement refresh()")

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if cache is enabled on this pipeline.

        Returns:
            True if cache is enabled, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement is_enabled()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled})"


# Backward compatibility alias
CacheAdapter = BaseCacheBackend
