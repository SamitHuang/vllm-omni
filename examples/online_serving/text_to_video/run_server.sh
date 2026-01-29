#!/bin/bash
# Wan2.2 online serving startup script

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8091}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"

echo "Starting Wan2.2 server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Boundary ratio: $BOUNDARY_RATIO"
echo "Flow shift: $FLOW_SHIFT"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --boundary-ratio "$BOUNDARY_RATIO" \
    --flow-shift "$FLOW_SHIFT"
