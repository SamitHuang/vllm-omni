# Image-To-Image

This example edits an input image with `Qwen/Qwen-Image-Edit` using the `image_edit.py` CLI.

## Local CLI Usage

### Single Image Editing

```bash
python image_edit.py \
  --image qwen_bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output output_image_edit.png \
  --num_inference_steps 50 \
  --cfg_scale 4.0
```

### Multiple Image Editing (Qwen-Image-Edit-2509+)

For multiple image inputs, use `Qwen/Qwen-Image-Edit-2509` or later version:

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Edit-2509 \
  --image img1.png img2.png img3.png \
  --prompt "Combine these images into a single scene" \
  --output output_image_edit.png \
  --num_inference_steps 50 \
  --cfg_scale 4.0
```

Key arguments:

- `--model`: model name or path. Use `Qwen/Qwen-Image-Edit-2509` or later for multiple image support.
- `--image`: path(s) to the source image(s) (PNG/JPG, converted to RGB). Can specify multiple images.
- `--prompt` / `--negative_prompt`: text description (string).
- `--cfg_scale`: true CFG scale for Qwen-Image-Edit (quality vs. fidelity).
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--output`: path to save the generated PNG.
