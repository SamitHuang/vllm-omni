# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
cache-dit integration adapter for vllm-omni.

This module provides functions to enable cache-dit acceleration on diffusion
pipelines in vllm-omni, supporting both single and dual-transformer architectures.
"""

from typing import Any, Callable, Optional

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


# Registry of custom cache-dit enablers for specific models
# Maps model class names to their cache-dit enablement functions
# Models in this registry require custom handling (e.g., dual-transformer architectures)
# Will be populated after function definitions
CUSTOM_DIT_ENABLERS: dict[str, Callable] = {}


def _build_db_cache_config(cache_config: Any) -> DBCacheConfig:
    """Build DBCacheConfig with optional SCM (Step Computation Masking) support.

    Args:
        cache_config: DiffusionCacheConfig instance.

    Returns:
        DBCacheConfig instance with SCM support if configured.
    """

    return DBCacheConfig(
        # we will refresh the context when gets num_inference_steps in the first inference request
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )


def enable_cache_for_wan22(pipeline: Any, od_config: OmniDiffusionConfig) -> Callable[[int], None]:
    """Enable cache-dit for Wan2.2 dual-transformer architecture.

    Wan2.2 uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Wan2.2 pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    cache_config = od_config.cache_config

    # Build DBCacheConfig for primary transformer
    primary_cache_config = _build_db_cache_config(cache_config)

    # FIXME: secondary cache shares the same config with primary cache for now,
    # but we should support different config for secondary transformer in the future
    # Build DBCacheConfig for secondary transformer (can use same or different config)
    secondary_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator configs if TaylorSeer is enabled
    primary_calibrator = None
    secondary_calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
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

    # from https://github.com/vipshop/cache-dit/pull/542
    def _split_inference_steps(num_inference_steps: int) -> tuple[int, int]:
        """Split inference steps into high-noise and low-noise steps for Wan2.2.

        This is an internal helper function specific to Wan2.2's dual-transformer
        architecture that uses boundary_ratio to determine the split point.

        Args:
            num_inference_steps: Total number of inference steps.

        Returns:
            A tuple of (num_high_noise_steps, num_low_noise_steps).
        """
        if pipeline.config.boundary_ratio is not None:
            boundary_timestep = pipeline.config.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # Set timesteps to calculate the split
        device = next(pipeline.transformer.parameters()).device
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = pipeline.scheduler.timesteps
        num_high_noise_steps = 0  # high-noise steps for transformer
        for t in timesteps:
            if boundary_timestep is None or t >= boundary_timestep:
                num_high_noise_steps += 1
        # low-noise steps for transformer_2
        num_low_noise_steps = num_inference_steps - num_high_noise_steps
        return num_high_noise_steps, num_low_noise_steps

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for both transformers with new num_inference_steps.

        Args:
            pipeline: The Wan2.2 pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        num_high_noise_steps, num_low_noise_steps = _split_inference_steps(num_inference_steps)
        # Refresh context for high-noise transformer
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_high_noise_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_high_noise_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy, total_steps=num_high_noise_steps
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

        # Refresh context for low-noise transformer
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer_2, num_inference_steps=num_low_noise_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer_2,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_low_noise_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy, total_steps=num_low_noise_steps
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


def enable_cache_for_flux(pipeline: Any, od_config: OmniDiffusionConfig) -> Callable[[int], None]:
    """Enable cache-dit for Flux dual-transformer architecture.

    Flux uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Flux pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    raise NotImplementedError("cache-dit is not implemented for Flux pipeline.")


def enable_cache_for_dit(pipeline: Any, od_config: OmniDiffusionConfig) -> Callable[[int], None]:
    """Enable cache-dit for regular single-transformer DiT models.

    Args:
        pipeline: The diffusion pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    cache_config = od_config.cache_config

    # Build DBCacheConfig with optional SCM support
    db_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator config if TaylorSeer is enabled
    calibrator_config = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator_config = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    model_class_name = od_config.model_class_name
    logger.info(
        f"Enabling cache-dit on {model_class_name} transformer: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit on the transformer
    cache_dit.enable_cache(
        pipeline.transformer,
        cache_config=db_cache_config,
        calibrator_config=calibrator_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The diffusion pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_inference_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


# Register custom cache-dit enablers after function definitions
CUSTOM_DIT_ENABLERS.update(
    {
        "WanPipeline": enable_cache_for_wan22,
        "FluxPipeline": enable_cache_for_flux,
    }
)


def may_enable_cache_dit(pipeline: Any, od_config: OmniDiffusionConfig) -> Optional[Callable[[int], None]]:
    """Enable cache-dit on the pipeline if configured.

    This function checks if cache-dit is enabled in the config and applies it
    to the appropriate transformer(s) in the pipeline. It handles both
    single-transformer and dual-transformer architectures.

    It also stores the cached num_inference_steps on the pipeline for later
    validation during inference and returns a refresh function that can be
    called to update the cache context with new num_inference_steps.

    Args:
        pipeline: The diffusion pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps,
        or None if cache-dit is not enabled.
    """
    if not CACHE_DIT_AVAILABLE:
        logger.warning("cache-dit is not available, skipping cache-dit setup.")
        return None

    # Check if this model has a custom cache-dit enabler
    model_class_name = od_config.model_class_name
    if model_class_name in CUSTOM_DIT_ENABLERS:
        logger.info(f"Using custom cache-dit enabler for model: {model_class_name}")
        refresh_func = CUSTOM_DIT_ENABLERS[model_class_name](pipeline, od_config)
        return refresh_func

    # For regular single-transformer models
    refresh_func = enable_cache_for_dit(pipeline, od_config)

    return refresh_func
