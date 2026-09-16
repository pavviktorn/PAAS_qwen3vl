#!/usr/bin/env bash
# 9B conditioning-distillation continuation (resume base LoRA adapter) + merge distilled -> qwen35_9b_merged.
# Mirrors the 4B-best recipe (dist_merge_qwen35.sh), 9B config r64/alpha128, gentle LRs. GPUs 1,2,3 only.
set -e
cd /datasets/work/vLLM/temp/PAAS_qwen3vl
BASE=base_models/Qwen3.5-9B
BASE_ADAPTER=runs/qwen35_9b_lora/adapter_final
DIST_OUT=runs/qwen35_9b_lora_distill
MERGED=runs/qwen35_9b_merged
CONDMIX=/datasets/newout/vqa_info_2+13+4+3_fmt/temp_qwen/effaa_cond_mix.json
EVAL=/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json
VPY=venv/bin/python
GPUS="${GPUS:-1,2,3}"; IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
[ -d "$BASE_ADAPTER" ] || { echo "ERROR: base adapter $BASE_ADAPTER missing"; exit 1; }
echo "==== 9B distill continuation $(date +%F_%T) | GPUs=$GPUS ===="
CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=8 \
  "$VPY" -m torch.distributed.run --nproc_per_node="$NG" --master_port 29619 \
    qwen/train_qwen_lora.py --model "$BASE" --resume-adapter "$BASE_ADAPTER" \
    --train "$CONDMIX" --eval "$EVAL" --image-root /datasets/newout \
    --out "$DIST_OUT" --epochs 1 --bs 8 --grad-accum 4 --lr 2e-5 \
    --lora-r 64 --lora-alpha 128 --train-merger --merger-lr 1e-5
echo "==== merge distilled adapter -> $MERGED $(date +%F_%T) ===="
CUDA_VISIBLE_DEVICES="${G[0]}" "$VPY" qwen/merge_lora.py \
    --base "$BASE" --adapter "$DIST_OUT/adapter_final" --out "$MERGED"
echo "==== 9B DISTILL+MERGE DONE $(date +%F_%T) ===="
