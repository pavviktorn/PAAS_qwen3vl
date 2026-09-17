#!/usr/bin/env bash
# PAAS_qwen3vl -- end-to-end FFAA-Qwen finetuning, chaining all three steps.
#   Step 1  LoRA-finetune Qwen3-VL-4B-Instruct on eFFAA_ext + merge  (-> runs/qwen_merged)
#   Step 2  generate MIDS train/eval jsons with the merged MLLM via vLLM (-> $WORK dir)
#   Step 3  train the MIDS 4-class head FROM SCRATCH on the Qwen-generated data
#
# Step 1 runs in the PROJECT VENV (transformers>=5 / peft); Step 2 in the venv (vLLM);
# Step 3 on the GLOBAL python3.12 / transformers==4.37.2 (T5+CLIP MIDS stack).
#
# Each step is gated:
#   RUN_STEP1=1 RUN_STEP2=1 RUN_STEP3=1 bash run_finetuning.sh     # full chain (default)
#   RUN_STEP1=0 RUN_STEP2=0 RUN_STEP3=1 bash run_finetuning.sh     # just retrain MIDS head
set -e

PROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root = this script's directory
cd "$PROOT"
VPY="$PROOT/venv/bin/python"

GPUS="${GPUS:-0,1,2,3}"
RUN_STEP1="${RUN_STEP1:-1}"; RUN_STEP2="${RUN_STEP2:-1}"; RUN_STEP3="${RUN_STEP3:-1}"

# --- training log: tee everything to a file ---
LOG_DIR="${LOG_DIR:-$PROOT/runs/finetune_logs}"
mkdir -p "$LOG_DIR"
LOG="${LOG:-$LOG_DIR/run_finetuning_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo "[run_finetuning] $(date '+%F %T') | logging to $LOG"

# ---- paths (override via env) ----
BASE_MODEL="${BASE_MODEL:-$PROOT/base_models/Qwen3-VL-8B-Instruct}"
EFFAA_TRAIN="${EFFAA_TRAIN:-/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json}"
EFFAA_EVAL="${EFFAA_EVAL:-/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/datasets/newout}"
WORK="${WORK:-/datasets/newout/vqa_info_2+13+4+3_fmt/temp_qwen}"           # MIDS working dir
MIDS_IMAGES="${MIDS_IMAGES:-/datasets/work/vLLM/temp/testset/testset_mids/mids_first_half.json}"
MIDS_EVAL_IMAGES="${MIDS_EVAL_IMAGES:-/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json}"
MERGED="${MERGED:-$PROOT/runs/qwen_merged}"
export PAAS_CLIP="$PROOT/base_models/clip-vit-large-patch14-336"
export PAAS_T5="$PROOT/base_models/t5-base"

IFS=',' read -ra G <<< "$GPUS"; NGPU="${#G[@]}"
mkdir -p "$WORK"

############################################
# STEP 1 -- LoRA finetune Qwen3-VL + merge
############################################
if [ "$RUN_STEP1" = "1" ]; then
  LORA_OUT="${LORA_OUT:-runs/qwen_lora}"
  echo "==== Step 1: LoRA-finetune Qwen3-VL ($BASE_MODEL, r${LORA_R:-32}) -> $LORA_OUT + merge ===="
  mkdir -p "$LORA_OUT"
  CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS="${OMP:-8}" \
    "$VPY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port "${PORT1:-29617}" \
      qwen/train_qwen_lora.py \
      --model "$BASE_MODEL" \
      --train "$EFFAA_TRAIN" --eval "$EFFAA_EVAL" --image-root "$IMAGE_ROOT" \
      --out "$LORA_OUT" \
      --epochs "${EPOCHS1:-3}" --bs "${BS1:-8}" --grad-accum "${ACC1:-4}" --lr "${LR1:-1e-4}" \
      --lora-r "${LORA_R:-64}" --lora-alpha "${LORA_A:-128}" \
      --train-merger --merger-lr "${MLR1:-2e-5}" --balance-real "${BALANCE_REAL:-2.44}"
  echo "---- merging LoRA -> $MERGED ----"
  CUDA_VISIBLE_DEVICES="${G[0]}" "$VPY" qwen/merge_lora.py \
      --base "$BASE_MODEL" --adapter "$LORA_OUT/adapter_final" --out "$MERGED"
else
  echo "---- Step 1 skipped (RUN_STEP1=0); using merged MLLM at $MERGED ----"
fi

############################################
# GATE -- functional check of the merged MLLM (after Step 1, before Step 2)
#   parse rate + verdict accuracy on eFFAA_ext_eval, sharded across ALL GPUs.
#   Aborts the chain if below GATE_MIN_PARSE / GATE_MIN_REAL (skip with RUN_GATE=0).
############################################
if [ "${RUN_GATE:-1}" = "1" ]; then
  echo "==== Gate: functional eval of $MERGED on $EFFAA_EVAL ($NGPU GPU shards) ===="
  GDIR="$PROOT/runs/gate"; mkdir -p "$GDIR"
  gpids=()
  for i in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES="${G[$i]}" VLLM_LOGGING_LEVEL=WARNING \
      "$VPY" qwen/eval_effaa.py --model "$MERGED" --eval "$EFFAA_EVAL" \
        --image-root "$IMAGE_ROOT" --limit "${GATE_N:-5000}" \
        --which-part "$i" --n-divided "$NGPU" --compliance \
        --json-out "$GDIR/gate_$i.json" > "$GDIR/gate_$i.log" 2>&1 &
    gpids+=("$!")
  done
  for i in "${!gpids[@]}"; do
    wait "${gpids[$i]}" || { echo "  gate shard $i FAILED (see $GDIR/gate_$i.log)"; exit 1; }
  done
  python3.12 - "$GDIR" "$NGPU" "${GATE_MIN_PARSE:-99}" "${GATE_MIN_REAL:-45}" "${GATE_MIN_COMPLY_REAL:-60}" "${GATE_MIN_COMPLY_FAKE:-80}" <<'PY'
import json, sys
gdir, n = sys.argv[1], int(sys.argv[2])
min_parse, min_real, min_cr, min_cf = (float(x) for x in sys.argv[3:7])
N = fmt = ok = 0
per = {"real": [0, 0], "fake": [0, 0]}; comp = {"real": [0, 0], "fake": [0, 0]}
for i in range(n):
    d = json.load(open(f"{gdir}/gate_{i}.json"))
    N += d["n"]; fmt += d["ok_fmt"]; ok += d["ok_verdict"]
    for k in per:
        per[k][0] += d["per"][k][0]; per[k][1] += d["per"][k][1]
        c = d.get("comply", {}).get(k, [0, 0]); comp[k][0] += c[0]; comp[k][1] += c[1]
parse = fmt / max(N, 1) * 100
real = per["real"][0] / max(per["real"][1], 1) * 100
fake = per["fake"][0] / max(per["fake"][1], 1) * 100
cr = comp["real"][0] / max(comp["real"][1], 1) * 100
cf = comp["fake"][0] / max(comp["fake"][1], 1) * 100
print(f"==== GATE RESULT: n={N:,} parse={parse:.2f}% verdict={ok/max(fmt,1)*100:.2f}% "
      f"real={real:.2f}% fake={fake:.2f}% | comply real={cr:.2f}% fake={cf:.2f}%")
print(f"     thresholds: parse>={min_parse} real>={min_real} comply_real>={min_cr} comply_fake>={min_cf}")
print("     (real bar = LLaVA parity 45; MIDS decides the verdict -- claim compliance is the")
print("      data-critical metric; LLaVA reference compliance was real 45.1 / fake 84.8)")
if parse < min_parse or real < min_real or cr < min_cr or cf < min_cf:
    print("==== GATE FAILED -- chain aborted before Step 2 (set RUN_GATE=0 to bypass) ====")
    sys.exit(1)
print("==== GATE PASSED ====")
PY
else
  echo "---- Gate skipped (RUN_GATE=0) ----"
fi

############################################
# STEP 2 -- MIDS data generation (vLLM, sharded across GPUs)
############################################
if [ "$RUN_STEP2" = "1" ]; then
  for TASK in train testset; do
    if [ "$TASK" = "train" ]; then SRC="$MIDS_IMAGES"; OUTBASE="$WORK/mids_qwen_train"; else SRC="$MIDS_EVAL_IMAGES"; OUTBASE="$WORK/mids_qwen_testset"; fi
    echo "==== Step 2: generate MIDS json ($TASK) from $SRC | $NGPU vLLM shards ===="
    pids=()
    for i in "${!G[@]}"; do
      CUDA_VISIBLE_DEVICES="${G[$i]}" VLLM_LOGGING_LEVEL=WARNING \
        "$VPY" qwen/gen_mids_vllm.py --model "$MERGED" \
          --images-json "$SRC" --out "${OUTBASE}_$i.json" \
          --which-part "$i" --n-divided "$NGPU" \
          --batch-size "${GEN_BS:-256}" > "$WORK/gen_${TASK}_$i.log" 2>&1 &
      pids+=("$!")
    done
    fail=0
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "  shard $i FAILED (see $WORK/gen_${TASK}_$i.log)"; fail=1; }
    done
    [ "$fail" = "0" ] || exit 1
    # ---- recovery pass: reprocess FORMAT-error images ONE-BY-ONE (not batched) with retries ----
    echo "---- Step 2 ($TASK): one-by-one recovery of format-error images ----"
    rpids=()
    for i in "${!G[@]}"; do
      if [ -s "${OUTBASE}_$i.json.errors.json" ]; then
        CUDA_VISIBLE_DEVICES="${G[$i]}" VLLM_LOGGING_LEVEL=WARNING \
          "$VPY" qwen/gen_mids_vllm.py --model "$MERGED" \
            --recover "${OUTBASE}_$i.json.errors.json" --out "${OUTBASE}_$i.json" \
            --retries "${RETRIES:-5}" >> "$WORK/gen_${TASK}_$i.log" 2>&1 &
        rpids+=("$!")
      fi
    done
    for j in "${!rpids[@]}"; do
      wait "${rpids[$j]}" || echo "  recovery shard $j failed (non-fatal; see logs)"
    done
    python3.12 - "$OUTBASE" "$NGPU" <<'PY'
import json, sys
base, n = sys.argv[1], int(sys.argv[2])
allr, seen = [], set()
for i in range(n):
    for r in json.load(open(f"{base}_{i}.json")):
        if r["image"] not in seen:
            seen.add(r["image"]); allr.append(r)
json.dump(allr, open(f"{base}.json", "w"))
from collections import Counter
print(f"  merged -> {base}.json: {len(allr):,} records | cls:", dict(Counter(r['cls_label'] for r in allr)))
PY
  done
else
  echo "---- Step 2 skipped (RUN_STEP2=0); using MIDS data at $WORK ----"
fi

############################################
# STEP 3 -- Train MIDS head (warm-start; global py3.12 / tf 4.37.2 stack)
############################################
if [ "$RUN_STEP3" = "1" ]; then
  INIT3="${INIT3-$PROOT/weights/mids_warmstart.pth}"    # warm-start head (v3 deployed 0.9784)
  echo "==== Step 3: train MIDS head (warm-start from $INIT3) on $WORK/mids_qwen_train.json ===="
  DEVICE_MAP="localhost:$GPUS"
  deepspeed --master_port "${PORT3:-25636}" --include "$DEVICE_MAP" \
      train_mids_new.py \
      --hidden_dim 768 --version v1 \
      --image_model_path "$PAAS_CLIP" --text_model_path "$PAAS_T5" \
      --init_model_path "$INIT3" \
      --data_path "$WORK/mids_qwen_train.json" --val_data_path "$WORK/mids_qwen_testset.json" \
      --output_dir "$PROOT/runs/mids_head" \
      --per_device_train_batch_size "${BS3:-24}" --per_device_val_batch_size 8 \
      --learning_rate "${LR3:-1e-5}" --unfreeze_vision_encoder_last_layers 2 \
      --num_train_epochs "${EPOCHS3:-5}" --warmup_ratio 0.03 --weight_decay 1e-5 \
      --eval_every_steps "${EVAL_STEPS3:-5000}" --select_metric "${SELECT_METRIC3:-acc}"
  echo "  trained MIDS head -> $PROOT/runs/mids_head/*.pth"
fi

echo "==== run_finetuning.sh DONE ===="
