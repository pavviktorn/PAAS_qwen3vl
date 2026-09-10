#!/bin/bash
# 8B axon1 held-out eval: gen (4 vLLM shards, GPUs 0-3) -> score (4 MIDS shards) -> merge results.txt.
# gen is the long pole (~614k frames). Resumable: gen skips frames already in each shard json.
set -e
cd /datasets/work/vLLM/temp/PAAS_qwen3vl
BASELINE=/datasets/work/vLLM/temp/PAAS_ensemble_v2/runs/test_axon0model_axon1datatest/results_ffaa.txt
MERGED=runs/qwen8b_merged
BEST=runs/mids_head/mids_v1_20260727_191229/best.pth
OUTD=runs/eval/axon1
VPY=venv/bin/python
GPY=python3.12
mkdir -p "$OUTD"

echo "==== axon1 STAGE gen (4 vLLM shards) $(date +%H:%M:%S) ===="
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $VPY qwen/gen_score_axon1.py gen \
    --baseline "$BASELINE" --merged "$MERGED" \
    --out "$OUTD/qwen8b_axon1_$g.json" --which-part $g --n-divided 4 \
    > "$OUTD/gen8b_$g.log" 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "gen shard $i FAILED (see $OUTD/gen8b_$i.log)"; fail=1; }; done
[ "$fail" = 1 ] && { echo "GEN FAILED"; exit 1; }
echo "==== axon1 gen done $(date +%H:%M:%S) ===="

echo "==== axon1 STAGE score (4 MIDS shards) $(date +%H:%M:%S) ===="
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $GPY qwen/gen_score_axon1.py score \
    --answers "$OUTD/qwen8b_axon1_$g.json" --mids "$BEST" \
    --out "$OUTD/qwen8b_axon1_$g.txt" --device cuda:0 \
    > "$OUTD/score8b_$g.log" 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "score shard $i FAILED"; fail=1; }; done
[ "$fail" = 1 ] && { echo "SCORE FAILED"; exit 1; }

echo "==== merge shards -> qwen8b_axon1.txt ===="
{ echo "# columns: OK truth pred fake_score match_score frame_key";
  for g in 0 1 2 3; do grep -v '^#' "$OUTD/qwen8b_axon1_$g.txt"; done; } > "$OUTD/qwen8b_axon1.txt"
echo "  merged $(grep -vc '^#' "$OUTD/qwen8b_axon1.txt") lines"

# also merge the pre-existing 4B axon0ws shards for the compare (one file per model)
if [ ! -f "$OUTD/qwen4b_axon1.txt" ]; then
  { echo "# columns: OK truth pred fake_score match_score frame_key";
    for g in 0 1 2 3; do grep -v '^#' "$OUTD/qwen_axon1_axon0ws_$g.txt"; done; } > "$OUTD/qwen4b_axon1.txt"
fi
echo "==== AXON1 EVAL DONE $(date +%H:%M:%S) ===="
