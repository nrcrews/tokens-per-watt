#!/usr/bin/env bash
# Side-by-side demo on a two-GPU host: GPU 0 uncapped, GPU 1 capped, same model and prompt.
# Usage: sudo scripts/demo.sh MODEL CAP_WATTS
set -euo pipefail

MODEL=${1:?model, e.g. Qwen/Qwen3-8B}
CAP=${2:?power limit for GPU 1 in watts}
DEFAULT=$(nvidia-smi -i 1 --query-gpu=power.default_limit --format=csv,noheader,nounits)

cleanup() {
  nvidia-smi -i 1 -pl "$DEFAULT" >/dev/null
  kill "$SERVER0" "$SERVER1" 2>/dev/null || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --port 8000 >/tmp/tpw-gpu0.log 2>&1 &
SERVER0=$!
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" --port 8001 >/tmp/tpw-gpu1.log 2>&1 &
SERVER1=$!
for port in 8000 8001; do
  until curl -sf "localhost:$port/health" >/dev/null; do sleep 2; done
done
nvidia-smi -i 1 -pl "$CAP"

tmux new-session -d -s tpw "tpw-demo --url http://localhost:8000 --gpu 0 --model $MODEL --label 'uncapped'; read"
tmux split-window -h -t tpw "tpw-demo --url http://localhost:8001 --gpu 1 --model $MODEL --label 'capped at ${CAP} W'; read"
tmux attach -t tpw
