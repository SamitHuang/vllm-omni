# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Example script for image-conditioned generation with Qwen-Image 2.1.

Qwen-Image 2.1 uses a single pipeline (QwenImage21Pipeline) for both
text-to-image and image-conditioned generation. Pass one or more condition
images (at most 4) via --image to run the image-conditioned path.

Usage (single image):
    python qwen_image_21_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars" \
        --negative-prompt "blurry, low quality, text, watermark" \
        --output qwen_image_21_edit.png \
        --num-inference-steps 50 \
        --cfg-scale 4.0

Usage (multiple images, up to 4):
    python qwen_image_21_edit.py \
        --image input1.png input2.png \
        --prompt "Combine these images into a single scene" \
        --negative-prompt "blurry, low quality" \
        --output qwen_image_21_edit.png \
        --num-inference-steps 50 \
        --cfg-scale 4.0

Notes:
    - Qwen-Image 2.1 only supports true classifier-free guidance
      (--cfg-scale); there is no guidance-distilled guidance_scale.
      --cfg-scale > 1 only takes effect when --negative-prompt is provided.
    - --height/--width must be multiples of 32. When omitted, the output size
      is derived from the last condition image's aspect ratio at ~1024x1024.
    - Cache acceleration backends (cache_dit / tea_cache) are not supported.

For more options, run:
    python qwen_image_21_edit.py --help
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    get_model_class_name,
)
from vllm_omni.platforms import current_omni_platform

MAX_INPUT_IMAGES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-conditioned generation with Qwen-Image 2.1.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2.1",
        help="Diffusion model name or local path.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input condition image file(s) (PNG, JPG, etc.). At most 4 images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt. Required for true classifier-free guidance (--cfg-scale > 1) to take effect.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="W",
        help="Output image width in pixels (multiple of 32). Default: None (derived from the condition image).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="H",
        help="Output image height in pixels (multiple of 32). Default: None (derived from the condition image).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help=(
            "True classifier-free guidance scale (default: 4.0). Enabled by setting cfg_scale > 1 and providing "
            "a negative prompt. Qwen-Image 2.1 supports true CFG only; there is no guidance_scale."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_image_21_edit.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num-outputs-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism (ring attention is not supported).",
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization (also tiles condition-image encoding).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=None,
        help="Disable torch.compile and force eager execution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.image) > MAX_INPUT_IMAGES:
        raise ValueError(f"Qwen-Image 2.1 supports at most {MAX_INPUT_IMAGES} input images, got {len(args.image)}")

    input_images = []
    for image_path in args.image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        input_images.append(Image.open(image_path).convert("RGBA"))

    # Use single image or list based on number of inputs
    input_image = input_images[0] if len(input_images) == 1 else input_images

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    omni_kwargs = {
        "model": args.model,
        "ulysses_degree": args.ulysses_degree,
        "cfg_parallel_size": args.cfg_parallel_size,
        "vae_use_tiling": args.vae_use_tiling,
    }
    if args.enforce_eager is not None:
        omni_kwargs["enforce_eager"] = args.enforce_eager
    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni)
    print("Pipeline loaded")

    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  CFG scale (true CFG): {args.cfg_scale}")
    if args.height is not None or args.width is not None:
        print(f"  Output size: {args.width or 'auto'}x{args.height or 'auto'}")
    if isinstance(input_image, list):
        print(f"  Number of input images: {len(input_image)}")
    else:
        print(f"  Input image size: {input_image.size}")
    print(f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, cfg_parallel_size={args.cfg_parallel_size}")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()

    prompt_dict = build_image_to_image_prompt(
        model_class_name=model_class_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
    )

    diffusion_params = OmniDiffusionSamplingParams(
        generator=generator,
        seed=args.seed,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_outputs_per_prompt,
        height=args.height,
        width=args.width,
    )

    outputs = omni.generate(prompt_dict, sampling_params_list=[diffusion_params])
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    if not outputs:
        raise ValueError("No output generated from omni.generate()")

    images = None
    for output in outputs:
        images = getattr(output, "images", None)
        if images:
            break

    if not images:
        images = extract_images_from_outputs(outputs)

    if not images:
        raise ValueError("No images found in request_output")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "qwen_image_21_edit"

    if len(images) <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {os.path.abspath(output_path)}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()
