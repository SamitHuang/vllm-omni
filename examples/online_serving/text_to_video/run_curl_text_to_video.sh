#!/bin/bash
# Wan2.2 text-to-video curl example

OUTPUT_PATH="wan22_output.mp4"

curl -X POST http://localhost:8091/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "height": 480,
    "width": 832,
    "num_frames": 33,
    "fps": 16,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "guidance_scale_2": 4.0,
    "boundary_ratio": 0.875,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
