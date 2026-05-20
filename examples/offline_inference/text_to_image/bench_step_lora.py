"""End-to-end benchmark: step-wise batched LoRA vs request-mode LoRA.

Generates `--num-prompts` images with a small synthetic Qwen-Image LoRA in two
modes and reports per-mode timing, throughput, and speedup.

Usage:
    python bench_step_lora.py --model Qwen/Qwen-Image \
        --num-prompts 16 --num-inference-steps 8 --height 512 --width 512
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id


def write_qwen_image_lora(adapter_dir: Path) -> str:
    """Create a tiny PEFT-format LoRA on transformer_blocks.0.attn.to_qkv."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    dim = 24 * 128  # num_attention_heads * attention_head_dim for Qwen-Image
    rank = 1
    module_name = "transformer_blocks.0.attn.to_qkv"

    lora_a = torch.zeros((rank, dim), dtype=torch.float32)
    lora_a[0, :] = 0.01  # gather a small fraction of every input channel
    lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
    lora_b[:dim, 0] = 0.2  # apply a bounded perturbation, only on the Q slice

    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": lora_a,
            f"base_model.model.{module_name}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": [module_name]}),
        encoding="utf-8",
    )
    return str(adapter_dir)


def _extract_images(outputs) -> list:
    """Pull PIL.Image instances out of the list[OmniRequestOutput]."""
    images = []
    for out in outputs:
        req_out = getattr(out, "request_output", None)
        if req_out is None or not hasattr(req_out, "images") or not req_out.images:
            raise RuntimeError(f"Output missing images: {out}")
        images.extend(req_out.images)
    return images


def run_mode(
    *,
    model: str,
    lora_dir: str | None,
    num_prompts: int,
    num_inference_steps: int,
    height: int,
    width: int,
    step_execution: bool,
    max_num_seqs: int,
    seed: int,
    save_dir: Path | None,
    tag: str,
) -> dict:
    omni = Omni(
        model=model,
        mode="text-to-image",
        step_execution=step_execution,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
    )

    lora_request = None
    if lora_dir is not None:
        lora_request = LoRARequest(
            lora_name=Path(lora_dir).stem,
            lora_int_id=stable_lora_int_id(lora_dir),
            lora_path=lora_dir,
        )

    prompts = [{"prompt": f"a photo of a cat #{i} sitting on a laptop keyboard"} for i in range(num_prompts)]
    sampling_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        true_cfg_scale=0.0,
        num_outputs_per_prompt=1,
        seed=seed,
        lora_request=lora_request,
        lora_scale=1.0 if lora_request is not None else 0.0,
    )

    _ = omni.generate(prompts[:1], sampling_params, use_tqdm=False)

    torch.accelerator.synchronize()
    start = time.perf_counter()
    outputs = omni.generate(prompts, sampling_params, use_tqdm=False)
    torch.accelerator.synchronize()
    elapsed = time.perf_counter() - start

    if len(outputs) != num_prompts:
        raise RuntimeError(f"Expected {num_prompts} outputs, got {len(outputs)}")

    images = _extract_images(outputs)
    saved_paths: list[Path] = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            p = save_dir / f"{tag}_p{i:02d}.png"
            img.save(p)
            saved_paths.append(p)

    omni.close()
    return {
        "wall_seconds": elapsed,
        "throughput_imgs_per_s": num_prompts / elapsed,
        "images": images,
        "saved_paths": saved_paths,
    }


def _pixel_diff(a, b) -> dict[str, float]:
    """L1/Linf pixel difference between two PIL images, in [0, 255]."""
    import numpy as np

    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    diff = np.abs(arr_a - arr_b)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "psnr": float(20.0 * np.log10(255.0 / max(np.sqrt((diff**2).mean()), 1e-8))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen-Image")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument(
        "--mode",
        choices=["all", "request", "step", "no_lora"],
        default="all",
        help="Which mode(s) to run. 'all' = no_lora baseline + request_mode + step_mode_batched.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save generated images and report per-mode pixel diffs.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="qi_lora_") as tmp:
        lora_dir = write_qwen_image_lora(Path(tmp) / "adapter")
        print(f"[bench] PEFT adapter at: {lora_dir}\n")

        results: dict[str, dict] = {}

        modes = []
        if args.mode in {"all", "no_lora"}:
            modes.append(("no_lora_request (lora=None, step_execution=False, max_num_seqs=1)", None, False, 1))
        if args.mode in {"all", "request"}:
            modes.append(("request_mode (lora=on, step_execution=False, max_num_seqs=1)", lora_dir, False, 1))
        if args.mode in {"all", "step"}:
            modes.append(
                (
                    f"step_mode_batched (lora=on, step_execution=True, max_num_seqs={args.num_prompts})",
                    lora_dir,
                    True,
                    args.num_prompts,
                )
            )

        for label, mode_lora_dir, step_execution, max_num_seqs in modes:
            print(f"[bench] === {label} ===")
            tag = label.split(" ", 1)[0]
            metrics = run_mode(
                model=args.model,
                lora_dir=mode_lora_dir,
                num_prompts=args.num_prompts,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                step_execution=step_execution,
                max_num_seqs=max_num_seqs,
                seed=args.seed,
                save_dir=args.save_dir,
                tag=tag,
            )
            results[label] = metrics
            print(
                f"[bench] {label}: wall={metrics['wall_seconds']:.3f}s "
                f"throughput={metrics['throughput_imgs_per_s']:.3f} imgs/s"
            )
            if metrics["saved_paths"]:
                print(f"[bench] saved {len(metrics['saved_paths'])} images, e.g. {metrics['saved_paths'][0]}")
            print()

        print("[bench] === Summary ===")
        for label, m in results.items():
            print(f"  {label}: wall={m['wall_seconds']:.3f}s throughput={m['throughput_imgs_per_s']:.3f} imgs/s")

        if args.save_dir is not None and len(results) >= 2:
            print("\n[bench] === Pixel diffs (0-255 scale) ===")
            labels = list(results)
            for i, a_label in enumerate(labels):
                for b_label in labels[i + 1 :]:
                    a_imgs = results[a_label]["images"]
                    b_imgs = results[b_label]["images"]
                    if len(a_imgs) != len(b_imgs):
                        continue
                    diffs = [_pixel_diff(a, b) for a, b in zip(a_imgs, b_imgs)]
                    mean_abs = sum(d["mean_abs"] for d in diffs) / len(diffs)
                    max_abs = max(d["max_abs"] for d in diffs)
                    mean_psnr = sum(d["psnr"] for d in diffs) / len(diffs)
                    print(
                        f"  {a_label.split(' ', 1)[0]} vs {b_label.split(' ', 1)[0]}: "
                        f"mean|diff|={mean_abs:.3f}  max|diff|={max_abs:.1f}  avg PSNR={mean_psnr:.2f} dB"
                    )


if __name__ == "__main__":
    main()
