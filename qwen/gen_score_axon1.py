#!/usr/bin/env python
"""Qwen3-VL FFAA on axonlabs_data_1, aligned to a baseline results.txt frame-key list.

Two stages (different envs -- run each separately):

  STAGE gen  (PROJECT VENV: vLLM + tf5)
    Reads the baseline results_ffaa.txt, decodes EXACTLY the frames it kept (path#frame=N; images
    or video frames via cv2), generates 3 Qwen answers per frame (FFAA 3-pass conditional protocol),
    and writes a MIDS answers-json shard {frame_key, cls_label, answers:[...x3]}. Sharded/resumable.

  STAGE score  (GLOBAL python3.12: T5+CLIP MIDS head)
    Reads the merged answer-json, re-decodes each frame_key, runs the MIDS head -> forgery_score,
    writes a results.txt with the SAME frame-keys (so it aligns 1:1 with the baseline for comparison).

  # gen (venv), 4 GPU shards:
  CUDA_VISIBLE_DEVICES=g ./venv/bin/python qwen/gen_score_axon1.py gen \
      --baseline .../results_ffaa.txt --merged runs/qwen_merged \
      --out $WORK/axon1_ans_g.json --which-part g --n-divided 4
  # score (global), 1 GPU:
  python3.12 qwen/gen_score_axon1.py score --answers $WORK/axon1_ans.json \
      --mids runs/mids_head/<run>/best.pth --out runs/eval/qwen_ffaa_axon1.txt
"""
import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from get_label import get_label_all                    # noqa: E402
from utils.file_utils import decode_response           # noqa: E402

CONDITION = "This is a _ human face. What evidence do you have?"
# columns: <status> truth=.. pred=.. type=.. fake=.. match=.. <image>  -- the image path may contain
# spaces/commas, so capture EVERYTHING after the match= field (don't split on whitespace).
KEY = re.compile(r"truth=(\w+)\b.*?\bfake=([0-9.]+|-+)\b.*?\bmatch=(?:[0-9.]+|-+)\s+(.+?)\s*$")
FRAME = re.compile(r"^(.*)#frame=(\d+)$")


def parse_baseline(path, which_part=0, n_divided=1):
    """[(frame_key, truth01)] for the numeric-scored frames the baseline KEPT (drop SK/ER).

    Sharding is BY VIDEO (not strided frame): whole videos are round-robin assigned to shards, so
    each video is opened/decoded by exactly ONE shard -- no redundant cross-shard video reads."""
    import collections
    by_vid = collections.OrderedDict()
    for ln in open(path):
        if ln.startswith("#"):
            continue
        m = KEY.search(ln)
        if not m or m.group(2).startswith("-"):
            continue
        key = m.group(3); y = 1 if m.group(1) == "fake" else 0
        fm = FRAME.match(key)
        vid = fm.group(1) if fm else key
        by_vid.setdefault(vid, []).append((key, y))
    out = []
    for i, (vid, frames) in enumerate(by_vid.items()):
        if i % n_divided == which_part:
            out.extend(frames)
    return out


def decode_frame(frame_key):
    """Single-frame decode (used by the SCORE stage). '<path>' or '<video>#frame=N' -> RGB PIL."""
    m = FRAME.match(frame_key)
    if not m:
        img = cv2.imread(frame_key, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else None
    vid, idx = m.group(1), int(m.group(2))
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read(); cap.release()
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) if ok else None


def iter_decoded_prefetch(frames, workers=4, qsize=512):
    """Threaded wrapper around iter_decoded: decode videos on a small worker pool and hand PILs to
    the (GPU-bound) caller through a bounded queue, so decode/IO overlaps generation instead of
    stalling it. Videos are split across `workers` threads; each thread runs iter_decoded on its
    slice. Order is not preserved across videos (irrelevant -- each item carries its frame_key)."""
    import collections
    import queue
    import threading
    by_vid = collections.OrderedDict()
    for key, y in frames:
        m = FRAME.match(key)
        by_vid.setdefault(m.group(1) if m else key, []).append((key, y))
    vid_lists = list(by_vid.values())
    q = queue.Queue(maxsize=qsize)
    SENTINEL = object()

    def worker(slice_frames):
        for item in iter_decoded(slice_frames):
            q.put(item)
        q.put(SENTINEL)

    threads = []
    for w in range(workers):
        slc = [fr for vl in vid_lists[w::workers] for fr in vl]
        if slc:
            t = threading.Thread(target=worker, args=(slc,), daemon=True)
            t.start(); threads.append(t)
    live = len(threads)
    while live:
        item = q.get()
        if item is SENTINEL:
            live -= 1; continue
        yield item


def iter_decoded(frames):
    """Yield (frame_key, truth, PIL) for a list of (frame_key, truth), decoding each VIDEO ONCE via
    a sequential read (grab/retrieve) instead of a per-frame seek -- ~50-100x faster than random
    seeking. Plain images are read directly. Order within a video is by ascending frame index."""
    import collections
    vids = collections.OrderedDict()          # video_path -> [(idx, key, truth)]  ; None-key => image
    for key, y in frames:
        m = FRAME.match(key)
        if m:
            vids.setdefault(m.group(1), []).append((int(m.group(2)), key, y))
        else:
            vids.setdefault(None, []).append((0, key, y))
    for vid, want in vids.items():
        if vid is None:                       # standalone images
            for _, key, y in want:
                img = cv2.imread(key, cv2.IMREAD_COLOR)
                if img is not None:
                    yield key, y, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            continue
        want.sort()
        meta = {idx: (key, y) for idx, key, y in want}
        cap = cv2.VideoCapture(vid)
        grabbed = -1                          # index of the most recently grab()'d frame
        for nxt in sorted(meta):
            ok = True
            while grabbed < nxt:              # cheap sequential advance to the wanted index
                ok = cap.grab()
                if not ok:
                    break
                grabbed += 1
            if not ok or grabbed != nxt:
                break                         # ran past end of video
            dec_ok, fr = cap.retrieve()
            if dec_ok:
                key, y = meta[nxt]
                yield key, y, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()


# ----------------------------------------------------------------- STAGE gen
def stage_gen(args):
    import transformers
    from vllm import LLM, SamplingParams

    frames = parse_baseline(args.baseline, args.which_part, args.n_divided)   # sharded BY VIDEO
    done = set()
    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out)); done = {r["frame_key"] for r in results}
    frames = [(k, y) for k, y in frames if k not in done]
    if args.limit:
        frames = frames[:args.limit]
    print(f"[gen {args.which_part}/{args.n_divided}] {len(frames):,} frames ({len(done):,} already done)", flush=True)
    if not frames:
        json.dump(results, open(args.out, "w")); return

    proc = transformers.AutoProcessor.from_pretrained(args.merged)
    prompt_base = open(os.path.join(_ROOT, "playground", "prompts.txt")).readline().strip()

    def chat(q):
        return proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tb, tf, tr = chat(prompt_base), chat(CONDITION.replace("_", "fake")), chat(CONDITION.replace("_", "real"))

    llm = LLM(model=args.merged, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              max_model_len=4096, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": args.max_pixels}})
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    def gen(tmpls, imgs):
        return [o.outputs[0].text.strip() for o in
                llm.generate([{"prompt": t, "multi_modal_data": {"image": im}}
                              for t, im in zip(tmpls, imgs)], sp)]

    def flush_batch(imgs, meta):
        a1 = gen([tb] * len(imgs), imgs)
        p2, p3 = [], []
        for t in a1:
            rj, _ = decode_response(t)
            if len(rj) != 5 or rj.get("Analysis result", "").lower() == "real":
                p2.append(tf); p3.append(tr)
            else:
                p2.append(tr); p3.append(tf)
        a2, a3 = gen(p2, imgs), gen(p3, imgs)
        for (key, y), answers3 in zip(meta, zip(a1, a2, a3)):
            contents, res, claims, ok = [], [], [], True
            for a in answers3:
                rj, _ = decode_response(a)
                if all(k in rj for k in ("Analysis result", "Image description", "Forgery reasoning")):
                    r = rj["Analysis result"].lower()
                    contents.append("Image description: %s\nForgery reasoning: %s"
                                    % (rj["Image description"], rj["Forgery reasoning"]))
                    res.append(r); claims.append(0 if r == "real" else 1)
                else:
                    ok = False; break
            if not ok:
                # keep a 1-answer fallback so the frame still scores (rare with Qwen)
                contents = [contents[0]] * 3 if contents else ["Image description: n/a\nForgery reasoning: n/a"] * 3
                res = (res + ["fake"] * 3)[:3]; claims = (claims + [1] * 3)[:3]
            results.append({"frame_key": key, "cls_label": int(y),
                            "answers": [{"content": contents[k], "result": res[k],
                                         "label": 2 * int(y) + claims[k]} for k in range(3)]})

    # stream frames through video-grouped sequential decode; flush vLLM in fixed-size batches
    n_saved = len(results); seen = 0
    bimgs, bmeta = [], []
    for key, y, im in iter_decoded_prefetch(frames):
        bimgs.append(im); bmeta.append((key, y)); seen += 1
        if len(bimgs) >= args.batch_size:
            flush_batch(bimgs, bmeta); bimgs, bmeta = [], []
            if len(results) - n_saved >= args.save_every:
                n_saved = len(results); json.dump(results, open(args.out, "w"))
                print(f"[gen {args.which_part}] decoded {seen:,}/{len(frames):,} | kept {len(results):,}", flush=True)
    if bimgs:
        flush_batch(bimgs, bmeta)
    json.dump(results, open(args.out, "w"))
    print(f"[gen {args.which_part}] DONE: kept {len(results):,} / {len(frames):,} frames -> {args.out}")


# ----------------------------------------------------------------- STAGE score
def stage_score(args):
    import torch
    import torch.nn.functional as F
    from transformers import T5Tokenizer, CLIPProcessor
    from mids.mids_arch import MIDS
    from mids.selector import make_decision_batch
    from utils.mids_utils import flatten_tuple_list

    CLIP = os.environ.get("PAAS_CLIP", os.path.join(_ROOT, "base_models", "clip-vit-large-patch14-336"))
    T5 = os.environ.get("PAAS_T5", os.path.join(_ROOT, "base_models", "t5-base"))
    data = json.load(open(args.answers))
    print(f"scoring {len(data):,} frames", flush=True)

    clip = CLIPProcessor.from_pretrained(CLIP)
    tok = T5Tokenizer.from_pretrained(T5, use_fast=False, legacy=False)
    model = MIDS(768, image_model_path=CLIP, text_model_path=T5)
    sd = model.state_dict()
    ft = {k.replace("module.", ""): v for k, v in torch.load(args.mids, map_location="cpu").items()}
    sd.update(ft); model.load_state_dict(sd)
    model = model.to(args.device).eval()

    ans = {r["frame_key"]: r for r in data}               # frame_key -> record
    frames = [(r["frame_key"], int(r["cls_label"])) for r in data]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fh = open(args.out, "w")
    fh.write("# columns: OK truth pred fake_score match_score frame_key\n")
    done = 0

    def flush(bimgs, btexts, bres, bmeta):
        B = len(bimgs)
        images = torch.cat(bimgs, 0).to(args.device)
        ids = tok(flatten_tuple_list(btexts), return_tensors="pt", padding="longest",
                  max_length=tok.model_max_length, truncation=True).to(args.device)
        logits = model(ids, images, None, B, 1, 1)["logits"]
        scores = F.softmax(logits, dim=2)
        _, preds, matches, forgeries = make_decision_batch(flatten_tuple_list(bres), scores)
        for b in range(B):
            key, y = bmeta[b]
            fh.write(f"OK truth={'fake' if y else 'real'} pred={'fake' if int(preds[b]) else 'real'} "
                     f"fake={float(forgeries[b]):.4f} match={float(matches[b]):.4f} {key}\n")

    with torch.no_grad():
        bimgs, btexts, bres, bmeta = [], [], [], []
        for key, y, im in iter_decoded_prefetch(frames):           # fast video-grouped sequential decode
            a3 = ans[key]["answers"][:3]
            bimgs.append(clip(images=im, return_tensors="pt")["pixel_values"])
            btexts.append(tuple(a["content"] for a in a3))
            bres.append(tuple(a["result"] for a in a3))
            bmeta.append((key, y))
            if len(bimgs) >= args.batch_size:
                flush(bimgs, btexts, bres, bmeta); done += len(bimgs)
                bimgs, btexts, bres, bmeta = [], [], [], []
                if (done // args.batch_size) % 50 == 0:
                    print(f"  scored {done:,}/{len(data):,}", flush=True)
        if bimgs:
            flush(bimgs, btexts, bres, bmeta); done += len(bimgs)
    fh.close()
    print(f"wrote {args.out} ({done:,} frames)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--baseline", required=True); g.add_argument("--merged", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--which-part", type=int, default=0); g.add_argument("--n-divided", type=int, default=1)
    g.add_argument("--batch-size", type=int, default=128); g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--max-pixels", type=int, default=451584); g.add_argument("--gpu-mem", type=float, default=0.9)
    g.add_argument("--save-every", type=int, default=5000); g.add_argument("--limit", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--answers", required=True); sc.add_argument("--mids", required=True)
    sc.add_argument("--out", required=True); sc.add_argument("--batch-size", type=int, default=64)
    sc.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    (stage_gen if args.stage == "gen" else stage_score)(args)


if __name__ == "__main__":
    main()
