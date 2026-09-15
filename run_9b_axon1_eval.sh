#!/usr/bin/env bash
# 9B axon1 held-out eval: gen 9B answers (3 vLLM shards, GPUs 1,2,3) -> score with 9B from-scratch head
# -> merge results.txt. Mirrors eval_axon1_qwen35.sh for Qwen3.5-9B. Resumable (gen skips done frames).
set -e
cd /datasets/work/vLLM/temp/PAAS_qwen3vl
BASELINE=/datasets/work/vLLM/temp/PAAS_ensemble_v2/runs/test_axon0model_axon1datatest/results_ffaa.txt
MERGED=runs/qwen35_9b_merged
BEST=runs/mids_head/mids_v1_20260808_183146/best.pth
OUTD=runs/eval/axon1
VPY=venv/bin/python
GPY=python3.12
GPUS=(1 2 3); N=${#GPUS[@]}
mkdir -p "$OUTD"

echo "==== 9B axon1 GEN ($N vLLM shards on GPUs ${GPUS[*]}) $(date +%F_%T) ===="
pids=()
for idx in "${!GPUS[@]}"; do
  g=${GPUS[$idx]}
  CUDA_VISIBLE_DEVICES=$g $VPY qwen/gen_score_axon1.py gen \
    --baseline "$BASELINE" --merged "$MERGED" \
    --out "$OUTD/qwen35_9b_axon1_$idx.json" --which-part $idx --n-divided $N \
    > "$OUTD/gen9b_$idx.log" 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "gen shard $i FAILED (see $OUTD/gen9b_$i.log)"; fail=1; }; done
[ "$fail" = 1 ] && { echo "9B AXON1 GEN FAILED"; exit 1; }
echo "==== 9B axon1 gen done $(date +%F_%T) ===="

echo "==== 9B axon1 SCORE ($N MIDS shards) $(date +%F_%T) ===="
pids=()
for idx in "${!GPUS[@]}"; do
  g=${GPUS[$idx]}
  CUDA_VISIBLE_DEVICES=$g $GPY qwen/gen_score_axon1.py score \
    --answers "$OUTD/qwen35_9b_axon1_$idx.json" --mids "$BEST" \
    --out "$OUTD/qwen35_9b_axon1_$idx.txt" --device cuda:0 \
    > "$OUTD/score9b_$idx.log" 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "score shard $i FAILED"; fail=1; }; done
[ "$fail" = 1 ] && { echo "9B AXON1 SCORE FAILED"; exit 1; }

echo "==== merge -> qwen35_9b_axon1.txt ===="
{ echo "# columns: OK truth pred fake_score match_score frame_key";
  for idx in "${!GPUS[@]}"; do grep -v '^#' "$OUTD/qwen35_9b_axon1_$idx.txt"; done; } > "$OUTD/qwen35_9b_axon1.txt"
echo "  merged $(grep -vc '^#' "$OUTD/qwen35_9b_axon1.txt") lines"
echo "==== 9B AXON1 EVAL DONE $(date +%F_%T) ===="
