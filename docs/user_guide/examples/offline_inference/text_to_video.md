# Text-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video>.

## Supported Models

| Model | Model ID |
|-------|----------|
| Wan2.2 T2V | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| LTX-2 | `Lightricks/LTX-2` |

## Wan2.2 T2V

The `Wan-AI/Wan2.2-T2V-A14B-Diffusers` pipeline generates short videos from text prompts.

### Local CLI Usage

```bash
python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --guidance-scale 4.0 \
  --guidance-scale-high 3.0 \
  --flow-shift 12.0 \
  --num-inference-steps 40 \
  --fps 16 \
  --output t2v_out.mp4
```

Key arguments:

- `--prompt`: text description (string).
- `--height/--width`: output resolution (defaults 480x832, i.e. 480P). Dimensions should align with Wan VAE downsampling (multiples of 8).
- `--num-frames`: Number of frames (Wan default is 81).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high).
- `--negative-prompt`: optional list of artifacts to suppress (the PR demo used a long Chinese string).
- `--boundary-ratio`: Boundary split ratio for low/high DiT. Default `0.875` uses both transformers for best quality. Set to `1.0` to load only the low-noise transformer (saves noticeable memory with good quality, recommended if memory is limited). Set to `0.0` loads only the high-noise transformer (not recommended, lower quality).
- `--fps`: frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: path to save the generated video.
- `--vae-use-slicing`: enable VAE slicing for memory optimization.
- `--vae-use-tiling`: enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.

> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

## LTX-2

The `Lightricks/LTX-2` pipeline generates videos with optional audio from text prompts.

### Local CLI Usage

```bash
python text_to_video.py \
  --model Lightricks/LTX-2 \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 480 --width 768 --num-frames 41 \
  --num-inference-steps 20 \
  --guidance-scale 3.0 \
  --flow-shift 1.0 \
  --boundary-ratio 1.0 \
  --enforce-eager \
  --seed 42 \
  --output ltx2_output.mp4
```

### Optimization Recipes

Benchmark results on 8×H800 (480×768, 41 frames, 20 steps):

| Configuration | Time (s) | Speedup | Type |
|---|---|---|---|
| Baseline (eager, 1 GPU) | 10.66 | 1.00× | — |
| 2-GPU Ulysses SP | 8.22 | 1.30× | Lossless |
| 4-GPU Ulysses SP | 7.47 | 1.43× | Lossless |
| Cache-DiT (1 GPU) | 6.43 | 1.66× | Lossy |
| **4-GPU Ulysses SP + Cache-DiT** | **6.04** | **1.77×** | **Best combo** |

Best combo (4-GPU Ulysses SP + Cache-DiT):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python text_to_video.py \
  --model Lightricks/LTX-2 \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 480 --width 768 --num-frames 41 \
  --num-inference-steps 20 \
  --guidance-scale 3.0 \
  --flow-shift 1.0 \
  --boundary-ratio 1.0 \
  --enforce-eager \
  --ulysses-degree 4 \
  --cache-backend cache_dit \
  --seed 42 \
  --output ltx2_best_combo.mp4
```

A helper script with all recipes is provided:

```bash
bash examples/offline_inference/text_to_video/ltx2_optimization_recipes.sh best-combo
# Or run all recipes: bash ltx2_optimization_recipes.sh all
```

!!! note "LTX-2 Notes"
    - LTX-2 supports Ulysses-SP, Ring-SP, and TP parallelism. See [parallelism docs](../../diffusion/parallelism_acceleration.md).
    - Cache-DiT provides the largest single-method speedup (~1.7×) but is lossy — compare output quality.
    - FP8 quantization reduces VRAM but does not improve speed on compute-bound H800 GPUs.
    - `--enforce-eager` is recommended for single-request benchmarks to avoid torch.compile warmup.
    - Requires `pip install av` for video encoding.

## Example materials

??? abstract "text_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_video/text_to_video.py"
    ``````

??? abstract "ltx2_optimization_recipes.sh"
    ``````bash
    --8<-- "examples/offline_inference/text_to_video/ltx2_optimization_recipes.sh"
    ``````
