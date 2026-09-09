#!/usr/bin/env python3.12
"""Frame-level N-way comparison across FFAA variants, aligned on the COMMON frame-key set.

Reads several results.txt (unified columns: OK truth=.. pred=.. [type=..] fake=.. match=.. <path[#frame=N]>),
intersects their frame-keys so every model is scored on the SAME frames, and recomputes per-model
metrics (same conventions as score_frames_mids.py: Mann-Whitney AUC, acc@0.5, real/fake recall,
frontier fake-recall @ real-floor). Writes a compare json keyed by model name.

  python3.12 qwen/build_compare4.py --out runs/eval/compare4_testset.json --floors 95,98 \
      llava=<...>/results_ffaa.txt qwen4b=runs/eval/qwen_ffaa_testset_axon0ws.txt \
      qwen8b=runs/eval/qwen8b_ffaa_testset.txt
"""
import argparse, json, re, sys
import numpy as np
from scipy.stats import rankdata

KEY = re.compile(r"truth=(\w+)\b.*?\bfake=([0-9.]+|-+)\b.*?\bmatch=(?:[0-9.]+|-+)\s+(.+?)\s*$")


def parse(path):
    """path -> {frame_key: (truth01, fake_score)}; drops non-numeric (SK/ER) rows."""
    d = {}
    for ln in open(path):
        if ln.startswith("#"):
            continue
        m = KEY.search(ln)
        if not m or m.group(2).startswith("-"):
            continue
        d[m.group(3)] = (1 if m.group(1) == "fake" else 0, float(m.group(2)))
    return d


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


def metrics(s, y, floors):
    m = {"auc": auc(s, y), "acc": float(((s >= 0.5) == (y == 1)).mean()),
         "rr": float((s[y == 0] < 0.5).mean()) if (y == 0).any() else float("nan"),
         "fkr": float((s[y == 1] >= 0.5).mean()) if (y == 1).any() else float("nan")}
    for f in floors:
        m[f"fr{f}"] = fr_at_real(s, y, f / 100.0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--floors", default="95,98", help="comma real-floors, e.g. 90,95,98,99")
    ap.add_argument("models", nargs="+", help="name=results.txt ...")
    args = ap.parse_args()
    floors = [int(x) for x in args.floors.split(",")]

    parsed = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        parsed[name] = parse(path)
        print(f"  {name}: {len(parsed[name]):,} scored frames <- {path}", flush=True)

    common = set.intersection(*[set(d) for d in parsed.values()])
    keys = sorted(common)
    print(f"  common (aligned) frames: {len(keys):,}", flush=True)
    if not keys:
        sys.exit("no common frame-keys across models")

    # truth is model-independent; take from the first model
    ref = parsed[list(parsed)[0]]
    y = np.array([ref[k][0] for k in keys])
    out = {"n": len(keys), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    for name, d in parsed.items():
        s = np.array([d[k][1] for k in keys])
        out[name] = metrics(s, y, floors)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    # console table
    cols = ["auc", "acc", "rr", "fkr"] + [f"fr{f}" for f in floors]
    print("\nmodel        " + "  ".join(f"{c:>7}" for c in cols))
    for name in parsed:
        print(f"{name:12s} " + "  ".join(f"{out[name][c]:7.4f}" for c in cols))


if __name__ == "__main__":
    main()
