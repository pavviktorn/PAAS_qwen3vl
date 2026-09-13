#!/usr/bin/env python3
"""Build a FIXED, balanced frame subset for the inference-speed benchmark.

Deterministic (stride-sampled, no RNG) so every model sees the IDENTICAL frames. Uses the testset
baseline (plain .jpg images -> zero video-seek variance) and writes bench_frames.json:
    [{"frame_key": <path>, "truth": 0|1}]
All three models (LLaVA-7B / Qwen-4B / Qwen-8B) read this same list.

  python3 qwen/make_bench_frames.py --baseline <results_ffaa.txt> --n 320 --out qwen/bench/bench_frames.json
"""
import argparse, json, os, re

KEY = re.compile(r"truth=(\w+)\b.*?\bfake=([0-9.]+|-+)\b.*?\bmatch=(?:[0-9.]+|-+)\s+(.+?)\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--n", type=int, default=320, help="total frames (split evenly real/fake)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reals, fakes = [], []
    for ln in open(args.baseline):
        if ln.startswith("#"):
            continue
        m = KEY.search(ln)
        if not m or m.group(2).startswith("-"):
            continue
        key = m.group(3)
        (reals if m.group(1) == "real" else fakes).append(key)

    half = args.n // 2
    def stride_pick(lst, k):
        if k >= len(lst):
            return lst
        step = len(lst) / k
        return [lst[int(i * step)] for i in range(k)]

    picked = [(k, 0) for k in stride_pick(reals, half)] + [(k, 1) for k in stride_pick(fakes, half)]
    # keep only frames that actually exist on disk (plain images)
    picked = [(k, y) for k, y in picked if os.path.exists(k.split("#frame=")[0])]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump([{"frame_key": k, "truth": y} for k, y in picked], open(args.out, "w"))
    nr = sum(1 for _, y in picked if y == 0); nf = len(picked) - nr
    print(f"wrote {args.out}: {len(picked)} frames (real {nr} / fake {nf})")


if __name__ == "__main__":
    main()
