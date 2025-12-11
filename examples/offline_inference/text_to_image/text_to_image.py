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
        "--cache_backend",
        type=str,
        default="tea_cache",
        choices=["cache_dit", "tea_cache"],
        help=(
            "Cache backend to use for acceleration. "
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

    # Configure cache based on backend type
    cache_config = None
    if args.cache_backend == "cache_dit":
        # cache-dit configuration: Hybrid DBCache + SCM + TaylorSeer
        cache_config = {
            # DBCache parameters
            "Fn_compute_blocks": 8,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.12,
            # TaylorSeer parameters
            "enable_taylorseer": True,
            "taylorseer_order": 1,
            # SCM (Step Computation Masking) parameters
            "scm_steps_mask_policy": "fast",
            "scm_steps_policy": "dynamic",
        }
        print("Using cache-dit backend with DBCache + SCM + TaylorSeer")
    elif args.cache_backend == "tea_cache":
        # TeaCache configuration
        cache_config = {
            # TeaCache parameters
            "rel_l1_thresh": 0.2,  # Threshold for accumulated relative L1 distance
            # model_type will be auto-detected from pipeline class name
            # coefficients will use model-specific defaults if not provided
        }
        print("Using TeaCache backend")

    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
    )

    # Time profiling for generation
    backend_info = f" (cache_backend: {args.cache_backend})" if args.cache_backend else " (no cache)"
    print(f"\n{'='*60}")
    print(f"Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {args.cache_backend if args.cache_backend else 'None (no acceleration)'}")
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
