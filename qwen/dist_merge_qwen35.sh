#!/bin/bash
# Qwen3.5-4B conditioning-distillation continuation (resume base adapter on effaa_cond_mix, 700k)
# then merge the DISTILLED adapter -> runs/qwen35_4b_merged (overwrites the base-only auto-merge).
# Gentle continuation LRs (LoRA 2e-5 / merger 1e-5), 1 epoch, matching the 4B/8B recipe.
set -e
cd /datasets/work/vLLM/temp/PAAS_qwen3vl
BASE=base_models/Qwen3.5-4B
BASE_ADAPTER=runs/qwen35_4b_lora/adapter_final
DIST_OUT=runs/qwen35_4b_lora_distill
MERGED=runs/qwen35_4b_merged
CONDMIX=/datasets/newout/vqa_info_2+13+4+3_fmt/temp_qwen/effaa_cond_mix.json
EVAL=/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json
VPY=venv/bin/python

[ -d "$BASE_ADAPTER" ] || { echo "ERROR: base adapter $BASE_ADAPTER missing"; exit 1; }
echo "==== distillation continuation $(date +%F_%T) ===="
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 \
  "$VPY" -m torch.distributed.run --nproc_per_node=4 --master_port 29618 \
    qwen/train_qwen_lora.py \
    --model "$BASE" \
    --resume-adapter "$BASE_ADAPTER" \
    --train "$CONDMIX" --eval "$EVAL" --image-root /datasets/newout \
    --out "$DIST_OUT" \
    --epochs 1 --bs 8 --grad-accum 4 --lr 2e-5 \
    --lora-r 32 --lora-alpha 48 \
    --train-merger --merger-lr 1e-5

echo "==== merge distilled adapter -> $MERGED $(date +%F_%T) ===="
CUDA_VISIBLE_DEVICES=0 "$VPY" qwen/merge_lora.py \
    --base "$BASE" --adapter "$DIST_OUT/adapter_final" --out "$MERGED"
echo "==== DIST+MERGE DONE $(date +%F_%T) ===="
