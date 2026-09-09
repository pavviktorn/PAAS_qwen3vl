#!/usr/bin/env python
"""Inference-SPEED benchmark for the FFAA MLLM swap: LLaVA-Mistral-7B / Qwen3-VL-4B / Qwen3-VL-8B.

Same condition for all three:
  * IDENTICAL fixed frame subset (bench_frames.json, built by make_bench_frames.py)
  * SINGLE GPU (set CUDA_VISIBLE_DEVICES to one device)
  * Native deployed backend: LLaVA -> HF transformers generate ; Qwen -> vLLM
  * IDENTICAL FFAA 3-pass conditional protocol (base + 2 conditioned), temp 0, max_new_tokens 512, 1 img
  * Images PRE-LOADED before the timer (decode is shared/identical; we time MLLM compute, not disk)

Two stages (different envs):

  STAGE gen  (produces answers + MLLM timing)
    # Qwen (PROJECT VENV: vLLM):
    CUDA_VISIBLE_DEVICES=0 ./venv/bin/python qwen/bench_speed.py gen --backend qwen \
        --model runs/qwen8b_merged --frames qwen/bench/bench_frames.json \
        --out qwen/bench/ans_qwen8b.json --timing qwen/bench/time_qwen8b_gen.json
    # LLaVA (GLOBAL python3.12, PYTHONPATH=<v2>/ffaa):
    PYTHONPATH=/datasets/work/vLLM/temp/PAAS_ensemble_v2/ffaa CUDA_VISIBLE_DEVICES=0 \
      python3.12 qwen/bench_speed.py gen --backend llava \
        --model /datasets/work/vLLM/temp/PAAS_ensemble_v2/weights/ffaa_llava_mids \
        --frames qwen/bench/bench_frames.json --out qwen/bench/ans_llava.json \
        --timing qwen/bench/time_llava_gen.json --batch-size 16

  STAGE mids (T5+CLIP head timing; model-independent, run once per head env -- GLOBAL python)
    CUDA_VISIBLE_DEVICES=0 python3.12 qwen/bench_speed.py mids --answers qwen/bench/ans_qwen8b.json \
        --mids runs/mids_head/<run>/best.pth --timing qwen/bench/time_mids.json
"""
import argparse, json, os, sys, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

CONDITION = "This is a _ human face. What evidence do you have?"


def load_frames(path):
    data = json.load(open(path))
    from PIL import Image
    frames = []
    for r in data:
        p = r["frame_key"].split("#frame=")[0]              # bench uses plain images
        frames.append((r["frame_key"], int(r["truth"]), Image.open(p).convert("RGB")))
    return frames


# --------------------------------------------------------------------- QWEN (vLLM)
def gen_qwen(args, frames):
    import transformers
    from vllm import LLM, SamplingParams
    from utils.file_utils import decode_response

    proc = transformers.AutoProcessor.from_pretrained(args.model)
    prompt_base = open(os.path.join(_ROOT, "playground", "prompts.txt")).readline().strip()

    def chat(q):
        return proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tb, tf, tr = chat(prompt_base), chat(CONDITION.replace("_", "fake")), chat(CONDITION.replace("_", "real"))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              max_model_len=4096, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": args.max_pixels}})
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    def gen(tmpls, imgs):
        return [o.outputs[0].text.strip() for o in
                llm.generate([{"prompt": t, "multi_modal_data": {"image": im}}
                              for t, im in zip(tmpls, imgs)], sp)]

    imgs = [im for _, _, im in frames]

    # warmup (excluded from timing): compile / cache graphs
    _ = gen([tb] * min(4, len(imgs)), imgs[:min(4, len(imgs))])

    t = {}
    t0 = time.time()
    a1 = gen([tb] * len(imgs), imgs); t["pass1"] = time.time() - t0
    p2, p3 = [], []
    for x in a1:
        rj, _ = decode_response(x)
        if len(rj) != 5 or rj.get("Analysis result", "").lower() == "real":
            p2.append(tf); p3.append(tr)
        else:
            p2.append(tr); p3.append(tf)
    t1 = time.time(); a2 = gen(p2, imgs); t["pass2"] = time.time() - t1
    t2 = time.time(); a3 = gen(p3, imgs); t["pass3"] = time.time() - t2
    t["gen_total"] = time.time() - t0
    return a1, a2, a3, t


# --------------------------------------------------------------------- LLAVA (HF)
def gen_llava(args, frames):
    from make_mids_dataset_from_folder_batch import load_llava, generate_batch_with_conditionals

    model, image_processor, tokenizer = load_llava(args.model, 0)
    prompt_base = open(os.path.join(_ROOT, "playground", "prompts.txt")).readline().strip()
    imgs = [im for _, _, im in frames]

    def run_all(image_list):
        out = [None] * len(image_list)
        B = args.batch_size
        for s in range(0, len(image_list), B):
            chunk = image_list[s:s + B]
            res = generate_batch_with_conditionals(
                model, tokenizer, image_processor, chunk, [prompt_base] * len(chunk),
                conv_mode="v1", temperature=0.0, top_p=None, num_beams=1,
                max_new_tokens=args.max_tokens, per_sample_generate_num=3)
            for i, r in enumerate(res):
                out[s + i] = r                              # r = [a1, a2, a3]
        return out

    import torch
    # warmup
    _ = run_all(imgs[:min(args.batch_size, len(imgs))])
    torch.cuda.synchronize()

    t = {}
    t0 = time.time()
    triples = run_all(imgs)
    torch.cuda.synchronize()
    t["gen_total"] = time.time() - t0                       # HF path: per-pass split not isolated (sub-batched)
    a1 = [x[0] for x in triples]; a2 = [x[1] for x in triples]; a3 = [x[2] for x in triples]
    return a1, a2, a3, t


# --------------------------------------------------------------------- shared: pack answers
def pack_answers(frames, a1, a2, a3):
    from utils.file_utils import decode_response
    recs = []
    for (key, y, _), triple in zip(frames, zip(a1, a2, a3)):
        contents, res, claims, ok = [], [], [], True
        for a in triple:
            rj, _ = decode_response(a)
            if all(k in rj for k in ("Analysis result", "Image description", "Forgery reasoning")):
                r = rj["Analysis result"].lower()
                contents.append("Image description: %s\nForgery reasoning: %s"
                                % (rj["Image description"], rj["Forgery reasoning"]))
                res.append(r); claims.append(0 if r == "real" else 1)
            else:
                ok = False; break
        if not ok:
            contents = [contents[0]] * 3 if contents else ["Image description: n/a\nForgery reasoning: n/a"] * 3
            res = (res + ["fake"] * 3)[:3]; claims = (claims + [1] * 3)[:3]
        recs.append({"frame_key": key, "cls_label": int(y),
                     "answers": [{"content": contents[k], "result": res[k],
                                  "label": 2 * int(y) + claims[k]} for k in range(3)]})
    return recs


def stage_gen(args):
    frames = load_frames(args.frames)
    print(f"[gen {args.backend}] {len(frames)} frames, single GPU", flush=True)
    if args.backend == "qwen":
        a1, a2, a3, t = gen_qwen(args, frames)
    else:
        a1, a2, a3, t = gen_llava(args, frames)
    recs = pack_answers(frames, a1, a2, a3)
    json.dump(recs, open(args.out, "w"))
    n = len(frames)
    t["n"] = n; t["backend"] = args.backend; t["model"] = args.model
    t["frames_per_s"] = n / t["gen_total"]
    t["passes"] = 3
    json.dump(t, open(args.timing, "w"), indent=2)
    print(f"[gen {args.backend}] n={n} gen_total={t['gen_total']:.2f}s "
          f"=> {t['frames_per_s']:.3f} frames/s (3-pass) -> {args.timing}", flush=True)


# --------------------------------------------------------------------- MIDS head timing
def stage_mids(args):
    import torch, torch.nn.functional as F
    from transformers import T5Tokenizer, CLIPProcessor
    from PIL import Image
    from mids.mids_arch import MIDS
    from mids.selector import make_decision_batch
    from utils.mids_utils import flatten_tuple_list

    CLIP = os.environ.get("PAAS_CLIP", os.path.join(_ROOT, "base_models", "clip-vit-large-patch14-336"))
    T5 = os.environ.get("PAAS_T5", os.path.join(_ROOT, "base_models", "t5-base"))
    data = json.load(open(args.answers))
    clip = CLIPProcessor.from_pretrained(CLIP)
    tok = T5Tokenizer.from_pretrained(T5, use_fast=False, legacy=False)
    model = MIDS(768, image_model_path=CLIP, text_model_path=T5)
    sd = model.state_dict()
    ft = {k.replace("module.", ""): v for k, v in torch.load(args.mids, map_location="cpu").items()}
    sd.update(ft); model.load_state_dict(sd); model = model.to(args.device).eval()

    # pre-load images (shared, not timed)
    items = []
    for r in data:
        p = r["frame_key"].split("#frame=")[0]
        a3 = r["answers"][:3]
        items.append((Image.open(p).convert("RGB"),
                      tuple(a["content"] for a in a3), tuple(a["result"] for a in a3)))

    def run(batch):
        B = len(batch)
        images = torch.cat([clip(images=im, return_tensors="pt")["pixel_values"] for im, _, _ in batch], 0).to(args.device)
        ids = tok(flatten_tuple_list([tx for _, tx, _ in batch]), return_tensors="pt", padding="longest",
                  max_length=tok.model_max_length, truncation=True).to(args.device)
        logits = model(ids, images, None, B, 1, 1)["logits"]
        scores = F.softmax(logits, dim=2)
        make_decision_batch(flatten_tuple_list([rs for _, _, rs in batch]), scores)

    B = args.batch_size
    with torch.no_grad():
        run(items[:min(B, len(items))])                     # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        for s in range(0, len(items), B):
            run(items[s:s + B])
        torch.cuda.synchronize()
        dt = time.time() - t0
    n = len(items)
    out = {"n": n, "mids_total": dt, "frames_per_s": n / dt, "mids": args.mids}
    json.dump(out, open(args.timing, "w"), indent=2)
    print(f"[mids] n={n} mids_total={dt:.2f}s => {n/dt:.3f} frames/s -> {args.timing}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--backend", choices=["qwen", "llava"], required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--frames", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--timing", required=True)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--max-pixels", type=int, default=451584)
    g.add_argument("--gpu-mem", type=float, default=0.9)
    g.add_argument("--batch-size", type=int, default=16, help="LLaVA HF sub-batch (vLLM ignores)")
    m = sub.add_parser("mids")
    m.add_argument("--answers", required=True)
    m.add_argument("--mids", required=True)
    m.add_argument("--timing", required=True)
    m.add_argument("--batch-size", type=int, default=64)
    m.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.cmd == "gen":
        stage_gen(args)
    else:
        stage_mids(args)


if __name__ == "__main__":
    main()
