#!/usr/bin/env python3.12
"""Frame-level FFAA scoring from a MIDS answers-json (GLOBAL python3.12 / transformers==4.37.2).

Given a MIDS-format json ([{image, cls_label, answers:[{content,result,label} x3]}], produced by
qwen/gen_mids_vllm.py) and a trained MIDS head, this reproduces FFAA's make_decision path per
frame -> forgery_score, and writes a unified results.txt (same columns as PAAS test_video_image_batch)
plus prints AUC / ACC / real- and fake-recall and the frontier fake-recall@real{95,98}.

This is the CACHED-answer scorer: the MLLM answers are already generated, so no LLaVA/Qwen here --
only the T5+CLIP MIDS head runs. Ground truth = cls_label (0 real / 1 fake) from the record.

  python3.12 qwen/score_frames_mids.py --mids runs/mids_head/<run>/best.pth \
      --answers /datasets/newout/vqa_info_2+13+4+3_fmt/temp_qwen/mids_qwen_testset.json \
      --out runs/eval/qwen_ffaa_testset.txt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5Tokenizer, CLIPProcessor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from mids.mids_arch import MIDS                        # noqa: E402
from mids.selector import make_decision_batch          # noqa: E402
from utils.mids_utils import flatten_tuple_list        # noqa: E402

CLIP = os.environ.get("PAAS_CLIP", os.path.join(_ROOT, "base_models", "clip-vit-large-patch14-336"))
T5 = os.environ.get("PAAS_T5", os.path.join(_ROOT, "base_models", "t5-base"))


def load_mids(path, model):
    sd = model.state_dict()
    ft = {k.replace("module.", ""): v for k, v in torch.load(path, map_location="cpu").items()}
    sd.update(ft); model.load_state_dict(sd)
    return model


def rankdata(a):
    o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); sa = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return r


def auc(s, y):
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if not p or not n:
        return float("nan")
    return float((rankdata(s)[y == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def fr_at_real(s, y, R):
    real = np.sort(s[y == 0]); fake = s[y == 1]; nr = len(real)
    if not nr or not len(fake):
        return float("nan")
    k = min(max(int(np.ceil(R * nr)), 1), nr)
    t = np.nextafter(real[k - 1], np.inf)
    return float((fake >= t).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mids", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden-dim", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    data = json.load(open(args.answers))
    data = [r for r in data if len(r.get("answers", [])) >= 3]
    print(f"scoring {len(data):,} frames from {args.answers}", flush=True)

    clip = CLIPProcessor.from_pretrained(CLIP)
    tok = T5Tokenizer.from_pretrained(T5, use_fast=False, legacy=False)
    model = MIDS(args.hidden_dim, image_model_path=CLIP, text_model_path=T5)
    model = load_mids(args.mids, model).to(args.device).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fh = open(args.out, "w")
    fh.write("# columns: OK truth pred fake_score match_score image\n")
    y_all, s_all = [], []
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            chunk = data[i:i + args.batch_size]
            imgs, texts, results, truths, paths = [], [], [], [], []
            for r in chunk:
                try:
                    im = Image.open(r["image"]).convert("RGB")
                except Exception:
                    continue
                px = clip(images=im, return_tensors="pt")["pixel_values"]
                imgs.append(px)
                a3 = r["answers"][:3]
                texts.append(tuple(a["content"] for a in a3))
                results.append(tuple(a["result"] for a in a3))
                truths.append(int(r["cls_label"])); paths.append(r["image"])
            if not imgs:
                continue
            B = len(imgs)
            images = torch.cat(imgs, 0).to(args.device)
            flat = flatten_tuple_list(texts)
            res = flatten_tuple_list(results)
            ids = tok(flat, return_tensors="pt", padding="longest",
                      max_length=tok.model_max_length, truncation=True).to(args.device)
            logits = model(ids, images, None, B, 1, 1)["logits"]
            scores = F.softmax(logits, dim=2)
            _, preds, matches, forgeries = make_decision_batch(res, scores)
            for b in range(B):
                fake = float(forgeries[b]); truth = "fake" if truths[b] else "real"
                pred = "fake" if int(preds[b]) == 1 else "real"
                fh.write(f"OK truth={truth} pred={pred} fake={fake:.4f} "
                         f"match={float(matches[b]):.4f} {paths[b]}\n")
                y_all.append(truths[b]); s_all.append(fake)
            if (i // args.batch_size) % 20 == 0:
                print(f"  {min(i + args.batch_size, len(data)):,}/{len(data):,}", flush=True)
    fh.close()

    y = np.array(y_all); s = np.array(s_all)
    m = {
        "n": len(y), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "auc": auc(s, y), "acc@0.5": float(((s >= 0.5) == (y == 1)).mean()),
        "real_recall@0.5": float((s[y == 0] < 0.5).mean()) if (y == 0).any() else float("nan"),
        "fake_recall@0.5": float((s[y == 1] >= 0.5).mean()) if (y == 1).any() else float("nan"),
        "fr@real95": fr_at_real(s, y, 0.95), "fr@real98": fr_at_real(s, y, 0.98),
    }
    print("\n==== frame-level metrics ====")
    for k, v in m.items():
        print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k:18s}: {v:,}")
    json.dump(m, open(args.out + ".metrics.json", "w"), indent=1)
    print(f"wrote {args.out} (+ .metrics.json)")


if __name__ == "__main__":
    main()
