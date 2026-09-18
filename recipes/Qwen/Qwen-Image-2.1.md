# Qwen-Image-2.1

> Text-to-image and image-conditioned generation with Qwen-Image 2.1

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image-2.1`
- Task: Text-to-image generation and image-conditioned generation (editing)
- Mode: Offline inference and online serving with optional step-wise execution
- Maintainer: Community

> **Model id:** `Qwen/Qwen-Image-2.1` follows the reference diffusers
> documentation and may change before the official release — the final
> checkpoint name is subject to the official release.

## When to use this recipe

Use this recipe as a starting point for running `Qwen/Qwen-Image-2.1`. A single
pipeline class, `QwenImage21Pipeline`, handles both pure text-to-image requests
(no condition image) and image-conditioned requests with up to 4 condition
images: prompt and condition images are encoded together by a Qwen3-VL text
encoder, so editing-style requests go through the same serving path as
text-to-image.

## References

- Related offline examples:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py),
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Supported models table:
  [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Feature compatibility matrix:
  [`docs/user_guide/diffusion_features.md`](../../docs/user_guide/diffusion_features.md)
- Sibling recipes: [`Qwen-Image.md`](./Qwen-Image.md),
  [`Qwen-Image-Edit.md`](./Qwen-Image-Edit.md)

## Offline Inference

### Text-to-image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A ceramic teapot on a wooden table" \
  --negative-prompt "blurry, low quality, text, watermark" \
  --output qwen_image_21_t2i.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

### Image-conditioned generation (editing)

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-2.1 \
  --color-format RGBA \
  --seed 42 \
  --image qwen_bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars" \
  --negative-prompt "blurry, low quality, text, watermark" \
  --output qwen_image_21_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

or via the bundled launcher:

```bash
bash examples/offline_inference/image_to_image/run_qwen_image_21_edit.sh
```

Up to 4 condition images can be passed via `--image`:

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-2.1 \
  --color-format RGBA \
  --seed 42 \
  --image input1.png input2.png \
  --prompt "Combine these images into a single scene" \
  --negative-prompt "blurry, low quality" \
  --output qwen_image_21_multi.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

Use `--color-format RGBA` to preserve transparency in condition images. The
shared example otherwise loads inputs as RGB. Add `--width` and `--height`
(multiples of 32) to set the output size; when omitted, Qwen-Image 2.1 derives
it from the last condition image's aspect ratio at approximately 1024×1024.
Use `--num-outputs-per-prompt` to generate multiple images and `--vae-use-tiling`
to reduce VAE memory usage. The launcher accepts these options as well.

## Online Serving

Start the server:

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091
```

To enable the step-wise runtime:

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

Step execution batches compatible requests at the same KV-cache phase. Once
running requests have completed their first denoising step, newly arriving
requests wait for that batch to finish before starting their own prefill.
`--max-num-seqs` controls batch capacity; mixed prefill/decode admission is not
supported yet.

### CUDA Graph decode

Qwen-Image-2.1 automatically uses CUDA Graph for supported fixed-shape denoising
decode steps when `enforce_eager=False` (the default). Prefill remains eager.
This works with both request execution and `--step-execution`.

To disable graph capture:

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni --enforce-eager --vae-use-tiling
```

The same control is available as `Omni(..., enforce_eager=True)` and as
`enforce_eager: true` on a stage in the deployment YAML. It disables both graph
capture and automatic `torch.compile`. Otherwise, this transformer uses its
model-specific decode graphs instead of automatic `torch.compile`.

The autoregressive engine's `compilation_config.cudagraph_mode` does not control
this diffusion path; use `enforce_eager` to force eager execution.

TP/SP/ring parallelism, HSDP, offload/cache hooks, compiled blocks, quantized
KV caches, dynamic LoRA, and padded text masks fall back to eager decode.
Graphs keep separate entries for different image layouts and copy request-owned
prefix K/V into static buffers before replay. Those buffers require additional
memory; graph entries are bounded by the transformer's cache limit.

### Verification

For a direct API smoke test:

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "A ceramic teapot on a wooden table",
    "negative_prompt": "blurry, low quality, text, watermark",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42
  }'
```

For image-conditioned generation, pass condition images through the
OpenAI-compatible multimodal endpoint (`/v1/chat/completions` with
`image_url` content parts, or `/v1/images/edits`), same as
[`Qwen-Image-Edit.md`](./Qwen-Image-Edit.md).

## Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `true_cfg_scale` | 4.0 | Only takes effect when `negative_prompt` is provided. Qwen-Image 2.1 supports true CFG only — there is no `guidance_scale` parameter. |
| `num_inference_steps` | 50 | Number of denoising steps. |
| `height` / `width` | 1024×1024 | Must be multiples of 32. For image-conditioned requests, leaving them unset derives the output size from the last condition image's aspect ratio at ~1024×1024. |
| `negative_prompt` | None | Required to activate true classifier-free guidance (`true_cfg_scale > 1`). |
| condition images | — | Up to 4 input images per request; more than 4 raises an error. |
| `seed` | — | Fix for reproducible outputs. |

### Prefix KV cache

The transformer caches the text/condition-image prefix across denoising steps
(a prefill/decode split keyed by CFG branch). The cache is enabled
automatically when the checkpoint declares `causal_condition: true` and is
re-created per generation, so no state leaks between requests. It changes the
compute path — cached prefix vs. full recompute at every step — so exact
numerical reproduction against a no-cache reference (e.g. the diffusers
pipeline) requires matching the cache setting on both sides.

## Known Limitations

- Cache acceleration backends (`cache_dit`, `tea_cache`) are not supported;
  `QwenImage21Pipeline` is registered in `_NO_CACHE_ACCELERATION`.
- Sequence parallelism supports Ulysses only; ring attention is not supported.
- Only true CFG is exposed; there is no `guidance_scale` knob.
- VAE tiling (`--vae-use-tiling`, also implied by `--vae-patch-parallel-size > 1`)
  decodes in 512px tiles with 384px stride (32 latent pixels per tile at 16x
  spatial compression). On CUDA OOM the tile size is automatically halved
  (keeping 25% overlap) and decoding retried, down to 128px tiles. Tile sizes
  are not yet exposed as engine arguments.
