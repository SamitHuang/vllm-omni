#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=90
#SBATCH --job-name=qi21-e2e
#SBATCH --output=/home/mozf/vllm-omni/artifacts/pr7759-review/e2e_slurm_%j.out

set -euo pipefail
cd /home/mozf/vllm-omni

export HF_HOME=/mnt/lustre/hf-models
export QWEN_IMAGE_21_TEST_MODEL=Qwen/Qwen-Image-2.1
export VLLM_LOGGING_LEVEL=INFO

echo "node: $(hostname)"
nvidia-smi -L | head -4
.venv/bin/python -c 'import vllm; print("vllm", vllm.__version__)'

.venv/bin/python -m pytest tests/e2e/online_serving/test_qwen_image_21.py \
  -o addopts="" -v -s 2>&1
