"""End-to-end benchmark: step-wise batched LoRA vs request-mode LoRA.

Generates `--num-prompts` images with a small synthetic Qwen-Image LoRA in two
modes and reports per-mode timing, throughput, and speedup.

Usage:
    python bench_step_lora.py --model /home/public/yx/models/Qwen/Qwen-Image \
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
    lora_a[0, 0] = 1.0
    lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
    lora_b[:dim, 0] = 0.05  # tiny perturbation, only on Q slice

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


def run_mode(
    *,
    model: str,
    lora_dir: str,
    num_prompts: int,
    num_inference_steps: int,
    height: int,
    width: int,
    step_execution: bool,
    max_num_seqs: int,
    seed: int,
) -> dict[str, float]:
    omni = Omni(
        model=model,
        mode="text-to-image",
        step_execution=step_execution,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
    )
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
        lora_scale=1.0,
    )

    _ = omni.generate(prompts[:1], sampling_params, use_tqdm=False)

    torch.accelerator.synchronize()
    start = time.perf_counter()
    outputs = omni.generate(prompts, sampling_params, use_tqdm=False)
    torch.accelerator.synchronize()
    elapsed = time.perf_counter() - start

    if len(outputs) != num_prompts:
        raise RuntimeError(f"Expected {num_prompts} outputs, got {len(outputs)}")

    omni.close()
    return {
        "wall_seconds": elapsed,
        "throughput_imgs_per_s": num_prompts / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/public/yx/models/Qwen/Qwen-Image")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument(
        "--mode",
        choices=["both", "request", "step"],
        default="both",
        help="Which mode(s) to run.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="qi_lora_") as tmp:
        lora_dir = write_qwen_image_lora(Path(tmp) / "adapter")
        print(f"[bench] PEFT adapter at: {lora_dir}\n")

        results: dict[str, dict[str, float]] = {}

        modes = []
        if args.mode in {"both", "request"}:
            modes.append(("request_mode (step_execution=False, max_num_seqs=1)", False, 1))
        if args.mode in {"both", "step"}:
            modes.append(
                (f"step_mode_batched (step_execution=True, max_num_seqs={args.num_prompts})", True, args.num_prompts)
            )

        for label, step_execution, max_num_seqs in modes:
            print(f"[bench] === {label} ===")
            metrics = run_mode(
                model=args.model,
                lora_dir=lora_dir,
                num_prompts=args.num_prompts,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                step_execution=step_execution,
                max_num_seqs=max_num_seqs,
                seed=args.seed,
            )
            results[label] = metrics
            print(
                f"[bench] {label}: wall={metrics['wall_seconds']:.3f}s "
                f"throughput={metrics['throughput_imgs_per_s']:.3f} imgs/s"
            )
            print()

        if "both" == args.mode and len(results) == 2:
            keys = list(results)
            base, opt = keys[0], keys[1]
            speedup = results[base]["wall_seconds"] / results[opt]["wall_seconds"]
            print("[bench] === Summary ===")
            print(f"  {base}: {results[base]['wall_seconds']:.3f}s")
            print(f"  {opt}:  {results[opt]['wall_seconds']:.3f}s")
            print(f"  Speedup (base/opt): {speedup:.2f}x")


if __name__ == "__main__":
    main()
