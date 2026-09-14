#!/bin/bash
# Inference-speed benchmark driver: LLaVA-7B / Qwen-4B / Qwen-8B on ONE GPU, native backend,
# identical FFAA 3-pass protocol, fixed 320-frame subset. Writes timing jsons to qwen/bench/.
set -e
cd /datasets/work/vLLM/temp/PAAS_qwen3vl
GPU="${GPU:-0}"
FRAMES=qwen/bench/bench_frames.json
BEST=runs/mids_head/mids_v1_20260727_191229/best.pth
VPY=venv/bin/python
GPY=python3.12
V2FFAA=/datasets/work/vLLM/temp/PAAS_ensemble_v2/ffaa
LLAVA=/datasets/work/vLLM/temp/PAAS_ensemble_v2/weights/ffaa_llava_mids
mkdir -p qwen/bench

echo "==== [1/4] Qwen-8B gen (vLLM, GPU $GPU) ===="
CUDA_VISIBLE_DEVICES=$GPU $VPY qwen/bench_speed.py gen --backend qwen \
  --model runs/qwen8b_merged --frames $FRAMES \
  --out qwen/bench/ans_qwen8b.json --timing qwen/bench/time_qwen8b_gen.json

echo "==== [2/4] Qwen-4B gen (vLLM, GPU $GPU) ===="
CUDA_VISIBLE_DEVICES=$GPU $VPY qwen/bench_speed.py gen --backend qwen \
  --model runs/qwen_merged --frames $FRAMES \
  --out qwen/bench/ans_qwen4b.json --timing qwen/bench/time_qwen4b_gen.json

echo "==== [3/4] LLaVA-7B gen (HF, GPU $GPU) ===="
PYTHONPATH=$V2FFAA CUDA_VISIBLE_DEVICES=$GPU $GPY qwen/bench_speed.py gen --backend llava \
  --model $LLAVA --frames $FRAMES \
  --out qwen/bench/ans_llava.json --timing qwen/bench/time_llava_gen.json --batch-size 16

echo "==== [4/4] MIDS head timing (T5+CLIP, GPU $GPU; model-independent) ===="
CUDA_VISIBLE_DEVICES=$GPU $GPY qwen/bench_speed.py mids \
  --answers qwen/bench/ans_qwen8b.json --mids $BEST --timing qwen/bench/time_mids.json

echo "==== BENCH DONE ===="
for f in qwen/bench/time_*_gen.json qwen/bench/time_mids.json; do echo "-- $f --"; cat "$f"; echo; done
