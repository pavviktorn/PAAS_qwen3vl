#!/usr/bin/env python3.12
"""Baseline: pass-1 verdict accuracy of the OLD FFAA LLaVA-Mistral MLLM on the SAME
eFFAA_ext_eval subset used by eval_effaa.py. Runs on the GLOBAL python3.12 /
transformers==4.37.2 with PAAS_ensemble_v2's vendored llava stack.

  PYTHONPATH=/datasets/work/vLLM/temp/PAAS_ensemble_v2/ffaa CUDA_VISIBLE_DEVICES=0 \
    python3.12 qwen/eval_effaa_llava.py --model /datasets/work/vLLM/temp/PAAS_ensemble_v2/weights/ffaa_llava_mids
"""
import argparse
import json
import os
import re
import sys

from PIL import Image

# NOTE: resolve `utils` from the v2 ffaa tree on PYTHONPATH (it also provides file_utils);
# do NOT put this project's root on sys.path here or its utils/ package shadows llava's.
from utils.file_utils import decode_response          # noqa: E402

_ = sys  # keep import


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json")
    ap.add_argument("--image-root", default="/datasets/newout")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--compliance", action="store_true",
                    help="also measure conditioned-answer compliance (FFAA pass-2/3 prompts)")
    args = ap.parse_args()

    import models as ffaa_models                      # vendored v2 ffaa/models.py (via PYTHONPATH)

    data = json.load(open(args.eval))
    items = []
    for r in data:
        conv = r.get("conversations") or []
        if len(conv) < 2 or "image" not in r:
            continue
        m = re.search(r"Analysis result:\s*(\w+)", conv[1]["value"])
        gt = m.group(1).lower() if m else None
        if gt in ("real", "fake"):
            q = conv[0]["value"].replace("<image>", "").strip()
            items.append((os.path.join(args.image_root, r["image"]), q, gt))
    if args.limit and args.limit < len(items):
        stride = len(items) / args.limit
        items = [items[int(i * stride)] for i in range(args.limit)]
    print(f"eval items: {len(items):,}")

    model, image_processor, tokenizer = ffaa_models.load_llava(args.model, 0)

    n = ok_fmt = ok_verdict = 0
    per = {"real": [0, 0], "fake": [0, 0]}
    for s in range(0, len(items), args.batch_size):
        chunk = items[s:s + args.batch_size]
        imgs, prompts, gts = [], [], []
        for p, q, gt in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB")); prompts.append(q); gts.append(gt)
            except Exception:
                pass
        if not imgs:
            continue
        answers = ffaa_models.get_llava_answer_batch(
            model, tokenizer, image_processor, imgs, prompts,
            0.0, None, 1, 512, 1, "v1")               # temp 0, 1 answer (pass-1 only)
        for gt, ans in zip(gts, answers):
            a = ans[0] if isinstance(ans, (list, tuple)) else ans
            n += 1
            rj, _ = decode_response(a)
            if all(k in rj for k in ("Analysis result", "Image description", "Forgery reasoning")):
                ok_fmt += 1
                per[gt][1] += 1
                if rj["Analysis result"].lower() == gt:
                    ok_verdict += 1; per[gt][0] += 1
        print(f"{min(s + args.batch_size, len(items)):,}/{len(items):,} | "
              f"parse {ok_fmt / max(n, 1) * 100:.2f}% | verdict {ok_verdict / max(ok_fmt, 1) * 100:.2f}%",
              flush=True)

    comply = {"real": [0, 0], "fake": [0, 0]}
    if args.compliance:
        cond = "This is a _ human face. What evidence do you have?"
        for target in ("real", "fake"):
            q = cond.replace("_", target)
            for s in range(0, len(items), args.batch_size):
                chunk = items[s:s + args.batch_size]
                imgs = []
                for p, _, _ in chunk:
                    try:
                        imgs.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass
                if not imgs:
                    continue
                answers = ffaa_models.get_llava_answer_batch(
                    model, tokenizer, image_processor, imgs, [q] * len(imgs),
                    0.0, None, 1, 512, 1, "v1")
                for ans in answers:
                    a = ans[0] if isinstance(ans, (list, tuple)) else ans
                    rj, _ = decode_response(a)
                    if "Analysis result" in rj:
                        comply[target][1] += 1
                        if rj["Analysis result"].lower() == target:
                            comply[target][0] += 1
            print(f"comply[{target}]: {comply[target][0]}/{comply[target][1]}", flush=True)

    print("\n==== LLaVA (old FFAA) eFFAA_ext_eval functional metrics ====")
    print(f"n={n:,}  parse-rate={ok_fmt / max(n, 1) * 100:.2f}%  "
          f"verdict-acc={ok_verdict / max(ok_fmt, 1) * 100:.2f}%")
    for k, (c, t) in per.items():
        print(f"  {k}: {c}/{t} = {c / max(t, 1) * 100:.2f}%")
    if args.compliance:
        for k, (c, t) in comply.items():
            print(f"  comply[{k}]: {c}/{t} = {c / max(t, 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
