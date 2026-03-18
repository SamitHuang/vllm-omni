#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# LTX-2 optimization recipes for vLLM-Omni.
# Benchmarked on 8×H800 with 480×768, 41 frames, 20 steps.
#
# Results:
#   Baseline (eager, 1 GPU)              : 10.66s (1.00×)
#   2-GPU Ulysses SP (eager)             :  8.22s (1.30×)
#   4-GPU Ulysses SP (eager)             :  7.47s (1.43×)
#   Cache-DiT (eager, 1 GPU)             :  6.43s (1.66×)
#   4-GPU Ulysses SP + Cache-DiT (eager) :  6.04s (1.77×)

set -euo pipefail

MODEL="${MODEL:-Lightricks/LTX-2}"
PROMPT="${PROMPT:-A serene lakeside sunrise with mist over the water.}"
SCRIPT="examples/offline_inference/text_to_video/text_to_video.py"

COMMON_ARGS=(
    --model "$MODEL"
    --prompt "$PROMPT"
    --height 480 --width 768 --num-frames 41
    --num-inference-steps 20
    --guidance-scale 3.0
    --flow-shift 1.0
    --seed 42
    --boundary-ratio 1.0
)

RECIPE="${1:-help}"

case "$RECIPE" in
    baseline)
        # 1-GPU baseline with eager execution (no torch.compile overhead)
        echo "=== LTX-2 Baseline: 1 GPU, enforce-eager ==="
        CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --enforce-eager \
            --output /tmp/ltx2_baseline.mp4
        ;;
    ulysses2)
        # 2-GPU Ulysses sequence parallelism (lossless, ~1.3× speedup)
        echo "=== LTX-2: 2-GPU Ulysses SP ==="
        CUDA_VISIBLE_DEVICES=0,1 python "$SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --enforce-eager \
            --ulysses-degree 2 \
            --output /tmp/ltx2_ulysses2.mp4
        ;;
    ulysses4)
        # 4-GPU Ulysses sequence parallelism (lossless, ~1.4× speedup)
        echo "=== LTX-2: 4-GPU Ulysses SP ==="
        CUDA_VISIBLE_DEVICES=0,1,2,3 python "$SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --enforce-eager \
            --ulysses-degree 4 \
            --output /tmp/ltx2_ulysses4.mp4
        ;;
    cache-dit)
        # Cache-DiT lossy acceleration (1 GPU, ~1.7× speedup)
        echo "=== LTX-2: Cache-DiT (1 GPU) ==="
        CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --enforce-eager \
            --cache-backend cache_dit \
            --output /tmp/ltx2_cachedit.mp4
        ;;
    best-combo)
        # Best combo: 4-GPU Ulysses SP + Cache-DiT (~1.8× speedup)
        echo "=== LTX-2: 4-GPU Ulysses SP + Cache-DiT (best combo) ==="
        CUDA_VISIBLE_DEVICES=0,1,2,3 python "$SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --enforce-eager \
            --ulysses-degree 4 \
            --cache-backend cache_dit \
            --output /tmp/ltx2_best_combo.mp4
        ;;
    all)
        echo "Running all recipes sequentially..."
        for r in baseline ulysses2 ulysses4 cache-dit best-combo; do
            bash "$0" "$r"
            echo ""
        done
        ;;
    *)
        echo "Usage: $0 {baseline|ulysses2|ulysses4|cache-dit|best-combo|all}"
        echo ""
        echo "Recipes:"
        echo "  baseline    - 1 GPU, eager (reference)"
        echo "  ulysses2    - 2-GPU Ulysses SP (lossless)"
        echo "  ulysses4    - 4-GPU Ulysses SP (lossless)"
        echo "  cache-dit   - Cache-DiT lossy acceleration (1 GPU)"
        echo "  best-combo  - 4-GPU Ulysses SP + Cache-DiT"
        echo "  all         - Run all recipes sequentially"
        echo ""
        echo "Environment variables:"
        echo "  MODEL  - Model path (default: Lightricks/LTX-2)"
        echo "  PROMPT - Text prompt"
        exit 1
        ;;
esac
