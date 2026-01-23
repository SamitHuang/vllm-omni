# export VLLM_TORCH_PROFILER_DIR=./profiles
# export VLLM_PROFILER_MAX_ITERS=1
# notes:
# guidance_scale_high equivalent to guidance_scale_2 in sglang
# boundary_ratio influence what timesteps to run low/high DiT (transformer, transformer_2), the two DiTs share the same arch.

python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  --height 480 \
  --width 832 \
  --num_frames 33 \
  --guidance_scale 1.0 \
  --guidance_scale_high 1.0 \
  --boundary_ratio  0.875 \
  --num_inference_steps 40 \
  --fps 16 \
  --output t2v_out.mp4
