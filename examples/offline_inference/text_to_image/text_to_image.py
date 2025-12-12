# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with Qwen-Image.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Diffusion model name or local path. Supported models: Qwen/Qwen-Image, Tongyi-MAI/Z-Image-Turbo",
    )
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="Text prompt for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale specific to Qwen-Image.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_image_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--cache_adapter",
        type=str,
        default="tea_cache",
        choices=["cache_dit", "tea_cache"],
        help=(
            "Cache adapter to use for acceleration. "
            "Options: 'cache_dit' (DBCache + SCM + TaylorSeer), 'tea_cache' (Timestep Embedding Aware Cache). "
            "Default: 'tea_cache' (Timestep Embedding Aware Cache enabled)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    # Configure cache based on adapter type
    cache_config = None
    if args.cache_adapter == "cache_dit":
        # cache-dit configuration: Hybrid DBCache + SCM + TaylorSeer
        # All parameters marked with [cache-dit only] in DiffusionCacheConfig
        cache_config = {
            # DBCache parameters [cache-dit only]
            "Fn_compute_blocks": 8,  # Number of forward compute blocks
            "Bn_compute_blocks": 0,  # Number of backward compute blocks
            "max_warmup_steps": 4,  # Maximum warmup steps
            "residual_diff_threshold": 0.12,  # Residual difference threshold for caching
            # TaylorSeer parameters [cache-dit only]
            "enable_taylorseer": True,  # Enable TaylorSeer forecasting
            "taylorseer_order": 1,  # TaylorSeer polynomial order
            # SCM (Step Computation Masking) parameters [cache-dit only]
            "scm_steps_mask_policy": "fast",  # SCM mask policy: "slow", "medium", "fast", "ultra"
            "scm_steps_policy": "dynamic",  # SCM steps policy: "dynamic" or "static"
        }
        print("✓ Using cache-dit adapter with DBCache + SCM + TaylorSeer")
        print(f"  Configuration: Fn={cache_config['Fn_compute_blocks']}, "
              f"threshold={cache_config['residual_diff_threshold']}, "
              f"SCM={cache_config['scm_steps_mask_policy']}")
    elif args.cache_adapter == "tea_cache":
        # TeaCache configuration
        # All parameters marked with [tea_cache only] in DiffusionCacheConfig
        cache_config = {
            # TeaCache parameters [tea_cache only]
            "rel_l1_thresh": 0.2,  # Threshold for accumulated relative L1 distance
            # Note: model_type is automatically detected from pipeline.__class__.__name__
            # Note: coefficients will use model-specific defaults based on model_type
            #       (e.g., QwenImagePipeline or FluxPipeline)
        }
        print("✓ Using TeaCache adapter")
        print(f"  Configuration: rel_l1_thresh={cache_config['rel_l1_thresh']}")
        print("  Note: model_type will be auto-detected from pipeline class name")

    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        cache_adapter=args.cache_adapter,
        cache_config=cache_config,
    )

    # Time profiling for generation
    adapter_info = f" (cache_adapter: {args.cache_adapter})" if args.cache_adapter else " (no cache)"
    print(f"\n{'='*60}")
    print(f"Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache adapter: {args.cache_adapter if args.cache_adapter else 'None (no acceleration)'}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"{'='*60}\n")

    generation_start = time.perf_counter()
    images = omni.generate(
        args.prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_images_per_prompt,
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "qwen_image_output"
    if args.num_images_per_prompt <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {save_path}")


if __name__ == "__main__":
    main()
