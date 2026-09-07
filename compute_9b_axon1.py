#!/usr/bin/env python3
"""Compute axon1 metrics for the 9B FFAA from a results.txt (OK truth=.. pred=.. fake=.. match=.. <key>):
AUC, real-recall@0.5, fake-recall at real-floors (FR90/95/98/99), and per-identity real-recall.
  python3 compute_9b_axon1.py runs/eval/axon1/qwen35_9b_axon1.txt
"""
import re, sys
from sklearn.metrics import roc_auc_score
KEY = re.compile(r"truth=(\w+)\b.*?\bfake=([0-9.]+|-+)\b.*?\bmatch=(?:[0-9.]+|-+)\s+(.+?)\s*$")
def load(path):
    ys, ss, keys = [], [], []
    for ln in open(path):
        if ln.startswith("#"): continue
        m = KEY.search(ln)
        if not m: continue
        t, f, k = m.group(1), m.group(2), m.group(3)
        if f.strip("-") == "": continue
        ys.append(1 if t == "fake" else 0); ss.append(float(f)); keys.append(k)
    return ys, ss, keys
def fr_at_floor(ys, ss, floor):
    reals = sorted(s for y, s in zip(ys, ss) if y == 0)
    if not reals: return None
    idx = min(len(reals)-1, int(round(floor*len(reals))))  # threshold s.t. real-recall ~= floor
    t = reals[idx]
    fakes = [s for y, s in zip(ys, ss) if y == 1]
    fr = sum(1 for s in fakes if s >= t)/len(fakes) if fakes else 0
    return fr, t
def main():
    ys, ss, keys = load(sys.argv[1]); n = len(ys)
    nr = sum(1 for y in ys if y == 0); nf = n-nr
    auc = roc_auc_score(ys, ss)
    real_rec05 = sum(1 for y, s in zip(ys, ss) if y == 0 and s < 0.5)/max(nr, 1)
    print(f"n={n:,} (real={nr:,} fake={nf:,})  AUC={auc:.4f}  real-rec@0.5={real_rec05*100:.2f}%")
    for fl in (0.90, 0.95, 0.98, 0.99):
        fr, t = fr_at_floor(ys, ss, fl); print(f"  FR{int(fl*100)} (real-floor {int(fl*100)}%): fake-recall={fr*100:.2f}%  (thr={t:.4f})")
    # per-identity real-recall @0.5 (reals only), identity from frame_key
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    for y, s, k in zip(ys, ss, keys):
        if y != 0: continue
        mid = re.search(r"(R_\d+)", k); idn = mid.group(1) if mid else "R_?"
        per[idn][1] += 1
        if s < 0.5: per[idn][0] += 1
    print("per-identity REAL-recall@0.5:")
    for idn in sorted(per):
        c, tot = per[idn]; print(f"  {idn}: {c/max(tot,1)*100:.1f}%  ({c:,}/{tot:,})")
if __name__ == "__main__": main()
