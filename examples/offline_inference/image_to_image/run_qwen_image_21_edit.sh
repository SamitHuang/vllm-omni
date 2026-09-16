python qwen_image_21_edit.py \
    --model Qwen/Qwen-Image-2.1 \
    --image qwen_bear.png \
    --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
    --negative-prompt "blurry, low quality, text, watermark" \
    --output qwen_image_21_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0
