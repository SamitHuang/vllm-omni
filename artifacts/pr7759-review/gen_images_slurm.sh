#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=60
#SBATCH --job-name=qi21-imgout
#SBATCH --output=/home/mozf/vllm-omni/artifacts/pr7759-review/gen_images_%j.out

set -euo pipefail
cd /home/mozf/vllm-omni

export HF_HOME=/mnt/lustre/hf-models
export VLLM_LOGGING_LEVEL=INFO
OUT=/home/mozf/vllm-omni/artifacts/pr7759-review/e2e_images
mkdir -p "$OUT"

echo "node: $(hostname)"

PY=.venv/bin/python

echo "=== T2I ==="
$PY examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A red ceramic teapot on a wooden table." \
  --negative-prompt "blurry, low quality" \
  --seed 42 \
  --height 1024 --width 1024 \
  --num-inference-steps 20 \
  --output "$OUT/t2i_teapot.png"

echo "=== I2I (edit the T2I output) ==="
$PY examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-2.1 \
  --image "$OUT/t2i_teapot.png" \
  --prompt "Make the teapot blue." \
  --seed 42 \
  --num-inference-steps 20 \
  --cfg-scale 1.0 \
  --output "$OUT/i2i_teapot_blue.png"

ls -la "$OUT"
