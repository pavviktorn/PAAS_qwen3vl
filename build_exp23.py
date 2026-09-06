#!/usr/bin/env python3
"""Experiment 23: three-seed replication of the detector stack (seeds 0 / 1981723 / 42).

Measures the noise floor of this pipeline -- how much the result moves when ONLY the seed changes --
and records the pipeline fixes that made a controlled seed comparison possible in the first place
(MIDS-4c was never seeded at all). Appends to EXPERIMENTS.pdf."""
import os, shutil
from fpdf import FPDF, XPos, YPos
from pypdf import PdfReader, PdfWriter
EXP = "/datasets/work/vLLM/temp/EXPERIMENTS.pdf"
pdf = FPDF(unit="pt", format=(612, 792)); pdf.set_auto_page_break(True, margin=40); pdf.set_margins(40, 40, 40)
CW = 532; NX, NY = XPos.LMARGIN, YPos.NEXT
def h1(t): pdf.set_font("Helvetica", "B", 13); pdf.multi_cell(CW, 16, t, new_x=NX, new_y=NY); pdf.ln(3)
def h2(t): pdf.ln(4); pdf.set_font("Helvetica", "B", 10); pdf.multi_cell(CW, 13, t, new_x=NX, new_y=NY)
def para(t, fs=9): pdf.set_font("Helvetica", "", fs); pdf.multi_cell(CW, 12.5, t, new_x=NX, new_y=NY)
def table(headers, rows, widths, fs=7.5, align0="L", boldrows=()):
    pdf.set_font("Helvetica", "B", fs); pdf.set_fill_color(228, 228, 228)
    for h, w in zip(headers, widths): pdf.cell(w, 13, h, border=1, align="C", fill=True)
    pdf.ln()
    for ri, r in enumerate(rows):
        pdf.set_font("Helvetica", "B" if ri in boldrows else "", fs)
        n = 1
        for c, w in zip(r, widths):
            n = max(n, len(pdf.multi_cell(w - 3, 9, str(c), dry_run=True, output="LINES")))
        hgt = 9 * n + 3
        if pdf.get_y() + hgt > pdf.h - pdf.b_margin:
            pdf.add_page(); pdf.set_font("Helvetica", "B", fs); pdf.set_fill_color(228, 228, 228)
            for h, w in zip(headers, widths): pdf.cell(w, 13, h, border=1, align="C", fill=True)
            pdf.ln(); pdf.set_font("Helvetica", "B" if ri in boldrows else "", fs)
        y0 = pdf.get_y(); x = pdf.l_margin
        for i, (c, w) in enumerate(zip(r, widths)):
            pdf.rect(x, y0, w, hgt); pdf.set_xy(x + 1.5, y0 + 1.5)
            pdf.multi_cell(w - 3, 9, str(c), border=0, align=(align0 if i == 0 else "C")); x += w
        pdf.set_xy(pdf.l_margin, y0 + hgt)

pdf.add_page()
h1("Experiment 23  -  three-seed replication of the detector stack   [2026-08-18 .. 08-20]")
para("Question: how much of a measured difference in this pipeline is real, and how much is seed "
     "noise? Three runs of the four detector branches on the SAME 1,366,146-image trainset, the SAME "
     "reused Qwen3.5-4B and its 1,366,114 generated answers, differing ONLY in the seed: 0 (Exp 22), "
     "1981723, 42. The MLLM chain (59h LoRA + 7h distill + merge + gate) was trained once in Exp 22 "
     "and reused verbatim, so the MLLM contributes zero variance here. Answer, up front: the seed "
     "moves nothing this validation set can resolve -- but the SAFE THRESHOLD BAND does move, which "
     "matters more than the accuracy numbers (Table D).")

h2("Table A - what varied, what was held fixed")
table(["", "seed 0 (Exp 22)", "seed 1981723", "seed 42"],
      [["run dir", "finetune_20260814_061131", "finetune_seed1981723_20260818_061214", "finetune_seed42_20260819_033852"],
       ["seed applied to", "9c=0, GSD=42, SeLop=42, MIDS=NONE", "all four = 1981723", "all four = 42"],
       ["GPUs", "1,2,3 (three)", "0,1,2,3", "0,1,2,3"],
       ["MLLM", "trained here (66 h)", "reused from Exp 22", "reused from Exp 22"],
       ["gen answers", "generated here (1,366,114)", "reused", "reused"],
       ["9c / GSD / SeLop / MIDS", "3h27 / 5h04 / 1h50 / 8h08", "2h57 / 5h07 / 1h22 / 8h08", "2h56 / 5h05 / 1h22 / 8h10"],
       ["calibration", "PARTIAL (stopped, reals only)", "2h45 full (4-GPU)", "2h45 full (4-GPU)"]],
      [92, 140, 150, 150])
para("Note the seed row: before this round the four branches used three DIFFERENT seeds and MIDS-4c "
     "used NONE (Table E), so 'seed 0' is really 'the seeds that happened to be in the configs'. Only "
     "the last two runs are controlled single-seed replications.", 8)

h2("Table B - per-branch validation results (mids_testset.json, n=30,197)")
table(["branch", "metric", "seed 0", "seed 1981723", "seed 42", "spread"],
      [["A2_9c", "val acc", "0.9995", "0.9995", "0.9995", "0.0000"],
       ["A2_9c", "val auc", "1.0000", "1.0000", "1.0000", "0.0000"],
       ["GSD", "bin_auc", "0.9999", "0.9999", "0.9999", "0.0000"],
       ["GSD", "acc", "0.9934", "0.9944", "0.9927", "0.0017"],
       ["GSD", "real recall", "0.994", "0.992", "0.996", "0.004"],
       ["GSD", "deepfake recall", "0.988", "0.995", "0.986", "0.009"],
       ["SeLop", "bin_auc", "1.0000", "0.9999", "1.0000", "0.0001"],
       ["MIDS-4c", "val acc", "0.9998", "0.9998", "0.9997", "0.0001"],
       ["MIDS-4c", "val auc", "1.0000", "1.0000", "1.0000", "0.0000"]],
      [66, 96, 90, 100, 90, 90], boldrows=(0, 1))
para("A2_9c is IDENTICAL to four decimals across three seeds. MIDS-4c and SeLop differ by one part in "
     "ten thousand -- three frames of 30,197. Only GSD moves at all, and it is the one branch whose "
     "selection metric never saturated (Exp 22 Table F); even there no seed dominates: seed 42 has the "
     "best real recall (0.996), seed 1981723 the best deepfake recall (0.995). CONSEQUENCE: any future "
     "change claiming an improvement on this validation set must clear ~0.002 accuracy on GSD and "
     "essentially ANY margin on the other three -- where three seeds cannot be distinguished at all, "
     "a difference in the fourth decimal is not evidence.", 8)

pdf.add_page()
h2("Table C - fused calibration, full testset media (n=30,218; real 10,126 / fake 20,092)")
para("Both controlled runs were calibrated end-to-end through all five members (MLLM + MIDS-4c + "
     "A2_9c + GSD + SeLop, plain mean). Seed 0's calibration was stopped early and covered reals "
     "only, so it cannot be compared here.", 8)
table(["quantity", "seed 1981723", "seed 42"],
      [["fused AUC", "1.0000", "1.0000"],
       ["fake recall (all floors)", "100.00%", "100.00%"],
       ["real fake-score p99", "0.1418", "0.1069"],
       ["real fake-score max", "0.9761", "0.8570"],
       ["LOWEST fake score", "0.4807", "0.3043"],
       ["max tau with 0 fake misses", "0.4806  -> real 99.89%", "0.3042  -> real 99.81%"],
       ["real FPs @ tau=0.2192", "58 of 10,126 (99.43%)", "39 of 10,126 (99.61%)"],
       ["auto-fitted tau (0.90 floor)", "0.0070  -> real 90.01%", "0.0036  -> real 90.13%"]],
      [160, 186, 186], boldrows=(4, 5))

h2("Table D - THE finding: the models are seed-insensitive, the safe threshold band is NOT")
para("Fake recall is 100% at every floor in both runs, so the DEPLOY_FLOOR=0.90 rule (smallest tau "
     "meeting the floor) writes a threshold that throws away ~10% of real frames for nothing. That "
     "was already noted in Exp 22. The new result is that the CEILING differs by seed:", 8)
table(["tau", "seed 1981723", "seed 42", "verdict"],
      [["0.2192 (old production)", "real 99.43%, 0 fake missed", "real 99.61%, 0 fake missed", "safe for both"],
       ["0.35", "real 99.79%, 0 fake missed (margin 0.13 to the lowest fake)",
        "real 99.84%, 1 FAKE MISSED (lowest fake is 0.3043)", "safe for 1981723 ONLY"],
       ["max safe", "0.4806", "0.3042", "differs by 58%"]],
      [96, 152, 152, 132], boldrows=(1,))
para("So two runs that are statistically indistinguishable on every accuracy metric have safe "
     "threshold ceilings of 0.4806 and 0.3042 -- the lowest-scoring fake frame is a single-sample "
     "statistic and it moves a long way between runs. A threshold chosen near the ceiling of one run "
     "is NOT transferable to another set of weights, even weights trained identically bar the seed. "
     "0.35 was set on the promoted seed-1981723 build (safe, 0.13 margin); on seed 42 the same value "
     "would have missed a fake. Practical rule, stronger than the existing one: re-fit per weights, "
     "and place tau by the REAL distribution (e.g. p99 + margin) rather than near the observed fake "
     "minimum, which is the least stable quantity in the calibration.", 8)

pdf.add_page()
h2("Table E - pipeline fixes that made this comparison possible (and one that made it honest)")
table(["#", "problem", "why it mattered", "fix"],
      [["1", "MIDS-4c was NEVER SEEDED",
        "it drives its own DeepSpeed loop instead of transformers.Trainer, so nothing called set_seed and TrainingArguments.seed was inert -- every MIDS run, including the deployed Exp-18 head, drew an uncontrolled torch RNG. A seed comparison was impossible for this branch",
        "transformers.set_seed(training_args.seed) + the value PRINTED at startup ('[mids] seed=42 (set_seed applied)')"],
       ["2", "seeds were per-branch and inconsistent (9c 0, GSD 42, SeLop 42)",
        "'changing the seed' meant editing three configs, and SeLop had no CLI override at all so its seed could only be changed by editing the file",
        "one SEED env var in run_finetuning.sh plumbed to all four; 'seed' added to SeLop's int-override list"],
       ["3", "no record of which seed produced which weights",
        "a seed you cannot find later is not reproducible",
        "$OUT_ROOT/run_config.json written at startup (seed + trainset + MLLM + every hyperparameter), copied into deploy/ so a promoted build carries its own provenance"],
       ["4", "unreadable images were silently replaced with a BLACK FRAME",
        "when the testset's deepfake folder was briefly moved aside mid-run, 9,725 of 30,197 eval images (32%) became black squares and GSD still reported 'anchor U(1024,16) from 30197 refs' plus a full val split -- an anchor one-third constant, embedded in every checkpoint, with nothing in the log",
        "guarded loader in gsd/data.py and selop/data.py: every substitution logged, ABORT past IMG_MISS_MAX (100/worker) or IMG_MISS_RATE (1%). Tested both directions: 1-in-300 tolerated, 33% aborts after 68 misses. All three later stages reported 0 substitutions"],
       ["5", "calibration was single-GPU (~44 frames/min)",
        "a full testset pass needed ~11 h while three GPUs idled; the Exp-22 calibration was abandoned part-way and its threshold could not be fitted",
        "--which-part/--n-divided added to scripts/run_dataset.py (strides MEDIA FILES, never splitting a video) + scripts/run_dataset.sh fans N shards over N GPUs and merges. 2h45m for 30,219 frames, ~4x. It also prints the LABEL MIX at merge, which is what makes a reals-only sample impossible to miss"]],
      [16, 118, 190, 208])

h2("Table F - promoted build")
table(["item", "value"],
      [["promoted", "seed 1981723 -> PAAS_ensemble_v4_inf/weights/ (10 GB)"],
       ["contents", "qwen35_4b_merged 8.5G (the MLLM this MIDS head was TRAINED against), ffaa_qwen35_mids/best.pth, gsd/best.pt (gstep 22000, bin_auc 0.999920), selop/best.pt, ensemble9/A2_svdgend_9c.pt, ensemble9.json, run_config.json"],
       ["threshold", "0.35 (operator decision) in both config/experiments/paas4_qwen.json and paas_seed1981723.json"],
       ["env fix", "env.GSD_CKPT was 'weights/gsd/best_lastN_ep0_auc0.9930.pt' -- a filename with a metric in it, so EVERY promotion silently broke that one default while the other four resolved. Now weights/gsd/best.pt, the stable name make_deploy_config.py writes"],
       ["archive", "PAAS_ensemble_v4_inf.zip -- 9.4 GB, 254 files, weights00 excluded"]],
      [70, 462])

h2("Verdict")
para("The pipeline replicates. Three seeds produce detector weights that this validation set cannot "
     "tell apart (A2_9c identical, MIDS/SeLop within 1e-4, GSD within 0.002), which is the useful "
     "negative result: it sets the noise floor for every future comparison and means seed-averaging "
     "is not required to trust a measurement at this resolution. It also means the validation set has "
     "no headroom left to demonstrate an improvement -- combined with Exp 22 Table E (axon1 is inside "
     "the trainset) and Table F (three of four selection metrics saturate), the honest summary is that "
     "this project can no longer MEASURE progress on the sets it currently uses. The next real step is "
     "not another training run; it is an evaluation set drawn from a different capture session than "
     "the trainset, with checkpoint selection on a metric that still has resolution (GSD's bin_auc "
     "moved through 0.9957 -> 0.9999 and is the existence proof that such a metric exists).", 8)

NEW = "/tmp/claude-1001/exp23_new.pdf"; os.makedirs(os.path.dirname(NEW), exist_ok=True); pdf.output(NEW)
shutil.copy(EXP, EXP + ".bak23")
w = PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP, "wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages)")
