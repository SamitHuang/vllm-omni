 sglang generate \
    --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --negative-prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --height 480 --width 832  --num-frames 33 \
    --guidance-scale 4.0 \
    --guidance-scale-2 4.0 \
    --boundary-ratio 0.875 \
    --num-inference-steps 40 \
    --fps 16 --save-output --output-path . --output-file-name "t2v_out.mp4" \
    --dit-layerwise-offload false \
    --dit-cpu-offload false \

# --text-encoder-cpu-offload false --image-encoder-cpu-offload false  --vae-cpu-offload false  \

# There is a bug in sglang diffusion guidance scale parsing, if set to 1.0, won't override, but pipeline scale is still 4.0
# so, to disable CFG, we should set guidance scale < 1.0
