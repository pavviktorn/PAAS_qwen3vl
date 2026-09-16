#!/usr/bin/env bash
set -e
cd /datasets/work/vLLM/temp/PAAS_qwen3vl
WSHEAD="/datasets/work/vLLM/temp/PAAS_qwen3vl/runs/mids_head/mids_v1_20260810_072504/best.pth"
OUTD=runs/eval/axon1
echo "==== WS axon1 SCORE (3 MIDS shards, warm-start head) $(date +%T) ===="
pids=()
for idx in 0 1 2; do
  g=$((idx+1))
  CUDA_VISIBLE_DEVICES=$g python3.12 qwen/gen_score_axon1.py score \
    --answers $OUTD/qwen35_9b_axon1_$idx.json --mids "$WSHEAD" \
    --out $OUTD/qwen35_9b_ws_axon1_$idx.txt --device cuda:0 > $OUTD/ws_score_$idx.log 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ws score shard $i FAILED"; fail=1; }; done
[ "$fail" = 1 ] && { echo "WS AXON1 SCORE FAILED"; exit 1; }
{ echo "# columns: OK truth pred fake_score match_score frame_key";
  for idx in 0 1 2; do grep -v '^#' $OUTD/qwen35_9b_ws_axon1_$idx.txt; done; } > $OUTD/qwen35_9b_ws_axon1.txt
echo "  merged $(grep -vc '^#' $OUTD/qwen35_9b_ws_axon1.txt) lines"
echo "==== WS testset SCORE (GPU1) $(date +%T) ===="
CUDA_VISIBLE_DEVICES=1 python3.12 qwen/score_frames_mids.py --mids "$WSHEAD" \
  --answers /datasets/newout/vqa_info_2+13+4+3_fmt/temp_qwen_9b/mids_qwen_testset.json \
  --out runs/eval/qwen35_9b_ws_testset.txt --device cuda:0 > $OUTD/ws_testset.log 2>&1
echo "==== WS SCORE DONE $(date +%T) ===="
