# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base cache adapter interface for diffusion models.

This module defines the abstract base class that all cache adapters must implement.
Cache adapters provide a unified interface for applying different caching strategies
(TeaCache, DeepCache, etc.) to transformer models using hooks.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionCacheConfig


class CacheAdapter(ABC):
    """
    Abstract base class for cache adapters.

    Cache adapters apply caching strategies to transformer models to accelerate
    inference. Each adapter type (TeaCache, DeepCache, etc.) implements the
    apply() and reset() methods to manage cache lifecycle.

    Attributes:
        config: DiffusionCacheConfig instance containing cache-specific configuration parameters
    """

    def __init__(self, config: DiffusionCacheConfig):
        """
        Initialize cache adapter with configuration.

        Args:
            config: DiffusionCacheConfig instance with cache-specific parameters
        """
        self.config = config

    @abstractmethod
    def apply(self, pipeline: Any) -> None:
        """
        Apply cache to transformer using hooks.

        This method should register the appropriate hooks on the transformer
        to enable caching during inference. Called once during pipeline
        initialization.

        Args:
            pipeline: Diffusion pipeline instance. The adapter can extract:
                     - transformer: via pipeline.transformer
                     - model_type: via pipeline.__class__.__name__
        """
        raise NotImplementedError("Subclasses must implement apply()")

    @abstractmethod
    def reset(self, transformer: torch.nn.Module) -> None:
        """
        Reset cache state for new generation.

        This method should clear any cached values and reset counters/accumulators.
        Called at the start of each generation to ensure clean state.

        Args:
            transformer: Transformer module to reset cache on
        """
        raise NotImplementedError("Subclasses must implement reset()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
