#!/bin/bash

# Qwen3 Omni MoE Offline Inference Example
# This script demonstrates text-to-speech generation using Qwen3 Omni MoE

# Set PYTHONPATH to include vllm-omni
# export PYTHONPATH=/path/to/vllm-omni:$PYTHONPATH

# Optional: Use HuggingFace mirror (for China users)
# export HF_ENDPOINT=https://hf-mirror.com

# --model Qwen/Qwen3-Omni-30B-A3B-Instruct \

# Run end-to-end inference
 python end2end.py \
     --model "/workspace/models/Qwen3-Omni-30B-A3B-Instruct" \
     --prompts "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words." \
     --voice-type "default" \
     --output-wav output_audio \
     --prompt_type text \
     --max-model-len 16384 \

#python end2end_mtp.py \
#    --model ${MODEL_PATH} \
#    --prompts "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words." \
#    --voice-type "default" \
#    --output-wav output_audio \
#    --prompt_type text \
#    # --num-speculative-tokens 3

# Output audio will be saved in ./output_audio/
