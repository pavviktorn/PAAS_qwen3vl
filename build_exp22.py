#!/usr/bin/env python3
"""Experiment 22: the full five-branch retrain on the 1.37M trainset (run 20260814_061131).

Records what was trained, what the numbers were, and -- the part that matters for the next run --
two findings that make the headline validation numbers unusable as evidence of quality:
axon1 contamination of the trainset, and metric saturation of the validation split.
Appends to EXPERIMENTS.pdf."""
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
        nlines = 1
        for c, w in zip(r, widths):
            nlines = max(nlines, len(pdf.multi_cell(w - 3, 9, str(c), dry_run=True, output="LINES")))
        hgt = 9 * nlines + 3
        if pdf.get_y() + hgt > pdf.h - pdf.b_margin:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", fs); pdf.set_fill_color(228, 228, 228)
            for h, w in zip(headers, widths): pdf.cell(w, 13, h, border=1, align="C", fill=True)
            pdf.ln(); pdf.set_font("Helvetica", "B" if ri in boldrows else "", fs)
        y0 = pdf.get_y(); x = pdf.l_margin
        for i, (c, w) in enumerate(zip(r, widths)):
            pdf.rect(x, y0, w, hgt); pdf.set_xy(x + 1.5, y0 + 1.5)
            pdf.multi_cell(w - 3, 9, str(c), border=0, align=(align0 if i == 0 else "C")); x += w
        pdf.set_xy(pdf.l_margin, y0 + hgt)

pdf.add_page()
h1("Experiment 22  -  full five-branch retrain on the 1.37M trainset   [2026-08-14 .. 08-18]")
para("run: PAAS_ensemble_v4/runs/finetune_20260814_061131. All five branches retrained FROM SCRATCH "
     "on one trainset (/datasets/work/vLLM/data/no_delete_mids_train, 1,366,146 images; get_label_all; "
     "33.2% real / 41.6% PAD / 25.1% deepfake), whole-frame letterbox everywhere, on a self-contained "
     "venv. Total ~104 GPU-hours of training plus ~7.5 h of MLLM generation. Every stage completed; "
     "calibration was stopped early by the user, so NO fusion threshold was fitted and the deploy "
     "build is NOT servable as it stands. Read Tables E and F before using any number in Table B: two "
     "independent problems make the near-perfect validation figures unusable as evidence of quality.")

h2("Table A - what ran, and how long")
table(["stage", "wall-clock", "outcome"],
      [["manifests", "54 s", "1,366,146 train / 30,197 val; 9-class histogram flat across labels 0..8"],
       ["A2_9c (SVD+GenD)", "3 h 27 m", "ep1 val acc 0.9990 auc 1.0000; ep2 acc 0.9995 auc 1.0000; CE 0.0746 -> 0.0011"],
       ["GSD (dual-stream)", "5 h 04 m", "best bin_auc 0.9999 @ gstep 28000, acc 0.9934, real recall 0.994"],
       ["SeLop (LROR)", "1 h 50 m", "bin_auc 1.0000 from step 3000 onward"],
       ["Qwen3.5-4B LoRA", "59 h 11 m", "59,874 steps; train 0.1505, eval 0.1296; 12 consecutive eval gains, no reversal"],
       ["distill continuation", "7 h 01 m", "7,292 steps over 700k conditioned records; train 0.1146, eval 0.1307"],
       ["merge", "20 s", "qwen_merged 8.5 GB"],
       ["GATE", "~6 m", "PASSED: parse 100.00, verdict 99.94, cond compliance real 99.98 / fake 99.92"],
       ["gen(train)", "~7 h 30 m", "1,366,114 records = 99.9977% coverage; 32 unrecoverable parse errors"],
       ["gen(testset)", "~10 m", "30,197 / 30,197 = 100.00%"],
       ["MIDS-4c", "8 h 08 m", "71,155 steps; final ACC 0.9998 AUC 1.0000; best.pth @ step 50000"],
       ["deploy", "6 s", "4 ckpts + configs; qwen_dir PINNED to this run; whole_frame=true"],
       ["calibrate", "2 h 47 m", "STOPPED by user at 7,249 frames -- threshold NOT fitted"]],
      [96, 62, 374])

h2("Table B - validation numbers (see Tables E/F before trusting these)")
table(["branch", "selection metric", "value", "note"],
      [["A2_9c", "AUC (production marginal)", "1.0000", "acc 0.9995; saturated from epoch 1"],
       ["GSD", "bin_auc", "0.9999", "the ONLY branch whose metric moved (0.9983 -> 0.9999)"],
       ["SeLop", "bin_auc", "1.0000", "saturated at step 3,000 of 10,671"],
       ["MIDS-4c", "AUC", "1.0000", "acc 0.9998; AUC saturated from step 10,000 of 71,155"],
       ["MLLM gate", "verdict acc", "99.94%", "n=5,000; real 99.93 / fake 99.94"]],
      [72, 122, 52, 286])

h2("Table C - the only genuinely informative training signal: GSD per-class recall")
para("GSD was the one branch whose validation metric had resolution left, and its per-class recalls "
     "show the real story of the run -- real recall is the axis that moves, and it recovers:", 8)
table(["gstep", "bin_auc", "acc", "real", "pad", "deepfake"],
      [["4000", "0.9983", "0.9742", "0.953", "0.992", "0.977"],
       ["12000", "0.9991", "0.9704", "0.932  (dip)", "0.999", "0.980"],
       ["18000", "0.9999", "0.9897", "0.982", "0.998", "0.988"],
       ["20000", "0.9999", "0.9924", "0.993", "0.999", "0.986"],
       ["28000 (best.pt)", "0.9999", "0.9934", "0.994", "0.998", "0.988"],
       ["32000 (final)", "0.9999", "0.9933", "0.996", "0.998", "0.986"]],
      [100, 66, 66, 100, 100, 100], boldrows=(4,))

pdf.add_page()
h2("Table D - partial calibration: the real-frame result, and why the threshold MUST be re-fitted")
para("Calibration scored 7,249 frames before being stopped. Every one is truth=real (run_dataset walks "
     "the media tree in order and had not reached the fake material), so there is NO AUC, NO fake "
     "recall, and fit_threshold correctly REFUSED to write a threshold (AUC=nan, one class present). "
     "What the 7,249 reals do give is a clean picture of the false-positive axis:", 8)
table(["quantity", "value"],
      [["real frames scored", "7,249 (100% of them truth=real)"],
       ["false positives", "5  ->  real-frame accuracy 99.93%"],
       ["real fake-score distribution", "p50 0.0004 | p90 0.0052 | p95 0.0125 | p99 0.0550 | max 0.3011"],
       ["the 5 FPs", "R_10 0.2250 | R_13 0.3011 | R_15 0.2267 | R_16 0.2499 | R_16 0.2291"],
       ["inherited threshold", "0.2192 (fitted on the OLD weights)"]],
      [150, 382])
para("All five false positives are the known-hard identities -- R_13 and R_15 are exactly the frames "
     "the axon0 warm-start used to over-fire on -- and every one lands in 0.225..0.301 while the real "
     "p99 is 0.055. Against the INHERITED 0.2192 they are all false positives; against any threshold "
     "above ~0.31 all 7,249 reals are correct. This is direct evidence for the standing rule that a "
     "threshold is specific to the weights it was fitted on: keeping 0.2192 would import 5 avoidable "
     "false positives on the easiest slice of the testset. NOTE the converse risk too -- a threshold "
     "fitted on reals alone is unvalidated against fakes and must not be deployed.", 8)

h2("Table E - FINDING 1: axon1 is no longer a held-out set for these models")
para("The trainset specified for this run contains the axon1 evaluation data. Counted directly from "
     "runs/manifests/images_train.json:", 8)
table(["source", "images in the trainset"],
      [["axonlabs_data_1  (axon1 -- the held-out benchmark)", "43,894"],
       ["axonlabs_data    (axon0)", "36,617"],
       ["other axon*", "833"]],
      [300, 232], boldrows=(0,))
para("Consequence: scoring these retrained detectors on axon1 and comparing against the recorded "
     "baselines (docs/COMBINATION_FINDINGS_axon1.md -- GSD 0.9985, SeLop 0.9980, FFAA 0.9965, "
     "ensemble 0.9982 AUC over 613,415 frames) compares a TRAINED-ON set against a HELD-OUT one. The "
     "new numbers would improve whether or not the models did. The historical baselines remain valid "
     "(none of those models ever saw axon1); it is only the new models that are contaminated. Fair "
     "comparison requires scoring axon1 with the 43,894 training images EXCLUDED -- the remainder is "
     "still genuinely held out for the new models and still held out for the old ones.", 8)

h2("Table F - FINDING 2: the validation split no longer discriminates")
para("mids_testset.json is a random split of the same pool as the trainset, so near-duplicate frames "
     "span both sides. Three of the four branches saturate their selection metric long before "
     "training ends, at which point checkpoint selection stops choosing and starts tracking recency "
     "(the comparison is >=, so ties overwrite):", 8)
table(["branch", "metric saturates at", "steps remaining after that", "effect on best.pt"],
      [["A2_9c", "AUC 1.0000 @ epoch 1", "18,974 of 37,948", "selection is by recency"],
       ["SeLop", "bin_auc 1.0000 @ step 3,000", "7,671 of 10,671", "selection is by recency"],
       ["MIDS-4c", "AUC 1.0000 @ step 10,000", "61,155 of 71,155", "acc varied 0.9989..0.9998 but AUC could not see it"],
       ["GSD", "did NOT saturate (0.9983 -> 0.9999)", "-", "genuine selection; best.pt = gstep 28000"]],
      [66, 150, 130, 186], boldrows=(3,))
para("Actionable fix for the next run: break AUC ties on a second criterion (balanced accuracy, or "
     "real recall at a fixed operating point), and/or select on a validation set drawn from a "
     "DIFFERENT capture session than the trainset. GSD is the existence proof that a harder metric "
     "still discriminates at this accuracy level.", 8)

pdf.add_page()
h2("Table G - pipeline defects found and fixed during this run (all silent -- none raised an error)")
table(["#", "defect", "how it would have shown up", "fix"],
      [["1", "flash_attn absent after the venv was made self-contained",
        "Step 2a aborted 10 s in -- but only AFTER 10 h of detector training had completed",
        "copied the already-compiled 2.8.3 build (matches torch 2.11.0+cu130/cp312) into the venv; pinned in requirements.txt; is_flash_attn_2_available() added to the isolation probe"],
       ["2", "SMOKE wrote its manifests to the SHARED runs/manifests/",
        "a smoke run silently replaced the production manifests (1,366,146 -> 40 records); the next real run with RUN_MANIFEST=0 would have trained every branch on 40 images and reported OK",
        "smoke manifests are per-run ($OUT_ROOT/manifests), with IMG_TRAIN/M9_* re-derived"],
       ["3", "GSD and SeLop training loops NEVER ran under SMOKE",
        "production batch (>=40 / 128) exceeded the 40/16-image smoke slice, so drop_last yielded ZERO steps: smoke printed 'gsd OK' for 3 epochs with train_loss=0.0000 and gstep=0",
        "SMOKE now passes batch_size=8 to both; their loops are genuinely exercised"],
       ["4", "'pip check' clean did not mean the stack worked",
        "'from vllm import LLM' died on a missing pycountry -- an undeclared runtime import hidden behind vLLM's lazy submodules; a bare 'import vllm' passed",
        "isolation probe now runs DEEP imports (the real entry points), not bare module imports"],
       ["5", "3-shard -> 4-shard resume would have corrupted the generated set",
        "sharding is a strided i::N split and resume only skips images already in ITS OWN shard file; moving 3->4 shards reassigns every image, so ~92% would be regenerated and old records would collide across shards",
        "the 230,399 completed records were merged and re-seeded into all four shard files; each shard then skipped them and generated only its own stride; verified 230,399 + 1,135,747 = 1,366,146 exactly"]],
      [16, 118, 190, 208])

h2("Table H - infrastructure changes made alongside the run")
table(["change", "detail"],
      [["venv is now self-contained", "was created with --system-site-packages and owned only 11 dists; 367 came from ~/.local (160) and /usr/local (207). ~/.local still holds a COMPLETE legacy stack (transformers 4.37.2, peft 0.7.1, tokenizers 0.15.2) that was inert only because the venv sorted earlier on sys.path. Now include-system-site-packages=false + PYTHONNOUSERSITE=1 + guards in train/_bootstrap.py and paas/env.py"],
       ["requirements.txt", "32 pins verified against the live venv; documents the three places where plain resolution CANNOT work (opencv declares numpy>=2 while the pipeline runs 1.26.4; albumentations requires opencv and re-triggers it; vllm's closure is unresolvable -- mistral-common backtracks to nothing)"],
       ["per-step logging", "9c / GSD / SeLop / MIDS-4c now emit one greppable line per TRAIN_LOG_EVERY steps with loss, acc and lr. A bare print() lost ~30% of lines to the tqdm bar; both tqdm-using trainers now use tqdm.write()"],
       ["PAAS_ensemble_v4_inf", "standalone inference-only project (54 py files, 13 GB, zero symlinks), built from the static import closure of the 4 inference entry points. Only one weight format per model is shipped (the upstream dirs carried the same tensors up to 5x). Verified by loading each model; since serving live on GPU 0"]],
      [110, 422])

h2("Verdict, and what to do next")
para("The pipeline is sound and every branch trained to completion with healthy curves -- the MLLM in "
     "particular improved its held-out eval loss 12 times consecutively with no reversal, and the gate "
     "cleared every threshold by a wide margin (conditioned compliance 99.98% on reals, against a 60% "
     "floor). The real-frame false-positive behaviour is the best it has been: 5 FPs in 7,249 reals, "
     "all on the known-hard identities, all comfortably separable by a correctly fitted threshold.", 8)
para("But this experiment CANNOT yet claim an accuracy improvement, for two independent reasons that "
     "are both about measurement rather than training: axon1 is inside the trainset (Table E), and the "
     "validation split is saturated (Table F). The next three actions, in order:", 8)
table(["#", "action", "why"],
      [["1", "Finish calibration on a BALANCED sample (reals and fakes), then fit the threshold",
        "the deploy build currently has threshold=None and cannot serve; a threshold fitted on reals alone is unvalidated against fakes"],
       ["2", "Score axon1 with the 43,894 training images excluded, and compare to the recorded baselines",
        "the only fair comparison available; the old baselines stay valid on that subset"],
       ["3", "Give the next run a validation set from a different capture session, and break AUC ties on real recall",
        "restores checkpoint selection as an actual selector for 9c / SeLop / MIDS-4c"]],
      [16, 250, 266])

NEW = "/tmp/claude-1001/exp22_new.pdf"; os.makedirs(os.path.dirname(NEW), exist_ok=True); pdf.output(NEW)
shutil.copy(EXP, EXP + ".bak22")
w = PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP, "wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages)")
