#!/usr/bin/env python3
"""Experiment 21: PAAS_ensemble_v4 pipeline hardening + WHOLE-FRAME preprocessing.

Documents the code/config corrections made before the full 1.37M-image retrain: 17 defects found by
an independent code review (gpt-5.6-sol) plus the whole-frame preprocessing change requested by the
user. CODE ONLY -- no training has run, so this entry records mechanisms and expected effects, not
measured accuracy. Appends to EXPERIMENTS.pdf."""
import os, shutil
from fpdf import FPDF, XPos, YPos
from pypdf import PdfReader, PdfWriter
EXP = "/datasets/work/vLLM/temp/EXPERIMENTS.pdf"
pdf = FPDF(unit="pt", format=(612, 792)); pdf.set_auto_page_break(True, margin=40); pdf.set_margins(40, 40, 40)
CW = 532; NX, NY = XPos.LMARGIN, YPos.NEXT
def h1(t): pdf.set_font("Helvetica", "B", 13); pdf.multi_cell(CW, 16, t, new_x=NX, new_y=NY); pdf.ln(3)
def h2(t): pdf.ln(3); pdf.set_font("Helvetica", "B", 10); pdf.multi_cell(CW, 13, t, new_x=NX, new_y=NY)
def para(t, fs=9): pdf.set_font("Helvetica", "", fs); pdf.multi_cell(CW, 12.5, t, new_x=NX, new_y=NY)
def table(headers, rows, widths, fs=7.5, align0="L", boldrows=()):
    """Wrapped-cell table: measures each cell's wrapped height so long text is never truncated."""
    pdf.set_font("Helvetica", "B", fs); pdf.set_fill_color(228, 228, 228)
    for h, w in zip(headers, widths): pdf.cell(w, 13, h, border=1, align="C", fill=True)
    pdf.ln()
    for ri, r in enumerate(rows):
        pdf.set_font("Helvetica", "B" if ri in boldrows else "", fs)
        nlines = 1
        for c, w in zip(r, widths):
            nlines = max(nlines, len(pdf.multi_cell(w - 3, 9, str(c), dry_run=True, output="LINES")))
        hgt = 9 * nlines + 3
        if pdf.get_y() + hgt > pdf.h - pdf.b_margin:      # keep rows whole across pages
            pdf.add_page()
            pdf.set_font("Helvetica", "B", fs); pdf.set_fill_color(228, 228, 228)
            for h, w in zip(headers, widths): pdf.cell(w, 13, h, border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "B" if ri in boldrows else "", fs)
        y0 = pdf.get_y(); x = pdf.l_margin
        for i, (c, w) in enumerate(zip(r, widths)):
            pdf.rect(x, y0, w, hgt)
            pdf.set_xy(x + 1.5, y0 + 1.5)
            pdf.multi_cell(w - 3, 9, str(c), border=0, align=(align0 if i == 0 else "C"))
            x += w
        pdf.set_xy(pdf.l_margin, y0 + hgt)


pdf.add_page()
h1("Experiment 21  -  PAAS_ensemble_v4 pipeline hardening + WHOLE-FRAME preprocessing   "
   "[CODE ONLY, 2026-08-13]")
para("Prepares v4 for a full retrain of all five branches (Qwen3.5-4B MLLM + MIDS-4c + A2_9c + GSD + "
     "SeLop) on the enlarged trainset /datasets/work/vLLM/data/no_delete_mids_train (1,366,146 images; "
     "get_label_all: 33.2% real / 41.6% PAD / 25.1% deepfake = 2.01:1 fake:real, essentially the same "
     "balance as the data that produced the Exp-18 best head). Two inputs drove the changes: (a) the "
     "user's directive that BOTH training and inference must use the WHOLE FRAME, and (b) an "
     "independent code review (gpt-5.6-sol) run repeatedly over the diff, which found 17 real defects "
     "-- most of them silent (no crash, just degraded or invalid results). IMPORTANT: this entry is "
     "code only. No training has been run, so nothing here is a measured accuracy claim; the effects "
     "described are mechanisms, to be confirmed by the post-training axon1 evaluation.")

h2("Table A - WHOLE-FRAME preprocessing (train == inference for every branch)")
para("Cropping discards the image border, which for presentation attacks is exactly where the evidence "
     "lives (screen edges, bezels, hands, moire). Three separate train/serve mismatches existed:", 8)
table(["branch", "was (train)", "was (inference)", "now (both)"],
      [["A2_9c", "CLIP resize+CENTER-CROP", "letterbox (whole frame)", "letterbox"],
       ["MIDS-4c", "CLIPProcessor crop (+RandomSizedCenterCrop p=0.5)", "CLIPProcessor crop", "letterbox"],
       ["GSD", "Resize+CenterCrop", "same fn (crop)", "letterbox"],
       ["SeLop", "RandomResizedCrop", "Resize-squash", "letterbox"]],
      [70, 175, 145, 142])
para("Canonical implementation: paas/preprocess.py (pad to square, resize, CLIP-normalise). GSD and "
     "SeLop record whole_frame=true INSIDE their checkpoint config, so inference self-selects the "
     "matching transform and legacy checkpoints keep their original behaviour (the running server was "
     "never disturbed); MIDS-4c uses ffaa.whole_frame, which make_deploy_config.py sets. A crop hidden "
     "in the MIDS albumentations pipeline (RandomSizedCenterCrop, p=0.5) still violated the rule for "
     "half the samples and was removed.", 8)

h2("Table B - A2_9c: the branch was training the WRONG MODEL (verified against the deployed ckpt)")
table(["property", "deployed A2_svdgend_9c.pt", "config in use (mids_pp.yaml)", "now (mids_a2_9c.yaml)"],
      [["num_classes", "9", "4  -> cannot load", "9"],
       ["artifact_enabled", "False", "true (that is A3's recipe)", "False"],
       ["clip_adapt / svd", "svd / last6 / rank16", "svd / last6 / rank16", "svd / last6 / rank16"],
       ["training data", "3 FIXED claim anchors, label=3*true+claim", "Qwen gen: label=2*cls+claim (0..3)", "build_manifests --mode mids9"]],
      [80, 150, 150, 152])
para("ensemble9/mids9lib/ensemble.py does logits.view(n,3,3), so a 4-class head cannot be loaded at "
     "all. The 9-class set is also MLLM-FREE: its three claim anchors are fixed texts (taken from "
     "config/ensemble9.json so they cannot drift), and true/claim are in {0 real,1 pad,2 deepfake} with "
     "MAKEUP folded into PAD and UNKNOWN dropped -- reproducing mids_plus/scripts/build_mids9.py, which "
     "produced the deployed member. The Qwen generator writes binary cls_label and 4-class labels and "
     "structurally cannot produce this. Verified: a checkpoint trained from the corrected config "
     "matches the deployed one field-for-field.", 8)

h2("Table C - accuracy-affecting defects fixed")
table(["#", "defect", "why it mattered", "fix"],
      [["1", "A2 selected by ACC of the winning-answer SELECTOR score; production ranks by mean(1-P(real)) over all answers",
        "not monotonic -- 5 ranking inversions in 30 random pairs; selection optimised a different ordering",
        "evaluate_split computes the production marginal (bit-exact vs ensemble.py, diff 0.0); select by its AUC"],
       ["2", "GSD scored against a batch-derived semantic basis whenever batch>=2",
        "same image scored differently depending on which frames shared its batch; the embedded anchor was effectively never used, and a threshold calibrated at one batch composition did not transfer",
        "explicit use_fixed_anchor(): fixed anchor at inference+validation, per-batch during training"],
       ["3", "GSD anchor choice keyed off self.training; engine._set_train_mode restores only child modules",
        "after the first validation, OPTIMIZATION silently used the TESTSET-derived anchor -> train/serve drift AND test-set leakage",
        "explicit _infer_anchor flag; _set_train_mode also restores the parent flag"],
       ["4", "MIDS-4c weight_decay never reached the optimizer (client optimizer voids ds_config's optimizer block)",
        "torch AdamW default 0.01 applied instead of the configured 1e-5 (1000x). The deployed Exp-18 head also trained at 0.01",
        "passed explicitly; default kept at the evidence-backed 0.01 (MIDS_WD), effective value printed"],
       ["5", "JPEG augmentation disabled by an albumentations 2.x API rename",
        "quality_lower/upper silently ignored -> effective range 99-100 instead of 60-100 (pre-dates the venv move; both envs are 2.0.8)",
        "version-agnostic _jpeg_kwargs -> verified (60,100)"],
       ["6", "threshold fit could violate its own real-recall floor",
        "scores are rounded, so a tau EQUAL to a tied group excluded it: a requested 90% floor returned 50% real-recall (reproduced)",
        "walk distinct scores, place tau strictly above the qualifying group, verify the WRITTEN value still meets the floor"],
       ["7", "calibration read 4dp scores while serving compares unrounded ones",
        "borderline frames could flip after calibration",
        "results written at 6dp (error <=5e-7)"],
       ["8", "MIDS progress metrics recomputed acc/AUC/AP over all accumulated samples every batch",
        "~9.7e9 element-ops per epoch at 1.37M images, on the training hot path, purely to fill the progress bar",
        "bounded rolling window on a stride + guarded AUC/AP: ~1700x less work"]],
      [16, 130, 200, 186])

h2("Table D - silent-failure guards added (all failure paths tested, not just the happy path)")
table(["guard", "what it prevents"],
      [["preflight config parse (train/validate_configs.py)",
        "a key a parser rejects killing a branch after the MLLM chain already burned a day (MidsPlusConfig rejects unknown keys); also asserts 9c num_classes==9"],
       ["gen coverage floor (GEN_MIN_COV, 0.95)",
        "training on a truncated set after format errors / failed recovery shards; recovery failures are now counted"],
       ["per-run isolation (WORK, MERGED under OUT_ROOT)",
        "a later run overwriting the gen data or the merged MLLM that an older deployment pins, invalidating its head and threshold"],
       ["MLLM fingerprint stamp ($WORK/.mllm)",
        "gen_mids_vllm resuming unconditionally and silently reusing a DIFFERENT model's answers"],
       ["deploy fail-fast (ALLOW_PARTIAL)",
        "a failed branch producing a deployment that mixes retrained and previously deployed weights; a missing gen set now records a FAILURE instead of skipping silently"],
       ["qwen_dir pinned by make_deploy_config",
        "the MIDS head being served answers from a different MLLM than it was trained on (config fields fall back to env defaults)"],
       ["gate: fake-verdict accuracy + conditioned parse-rate",
        "a model failing on fakes, or one whose conditioned answers mostly fail to parse, passing the gate (compliance excludes unparsable outputs from its denominator)"],
       ["one-venv guard (train/_bootstrap.py, paas/env.py)",
        "running on the legacy transformers==4.37 interpreter, which breaks the tf5 CLIP ports mid-load"]],
      [175, 357])

h2("Deliberately NOT changed")
para("(1) MIDS rank_margin is accepted but unused -- the loss is a smooth logistic ranking "
     "(softplus, no margin term). Documented as inert rather than implemented: adding a margin would "
     "change the loss that produced the proven head. (2) weight_decay default stays 0.01, the value the "
     "deployed Exp-18 head actually trained at, rather than the 1e-5 that was never in effect. In both "
     "cases the config line is decorative, and the measured result outranks the stated intent.", 8)

h2("Conclusion")
para("(1) The v4 retrain would have produced an UNUSABLE A2 member (4-class, wrong architecture, on data "
     "that cannot express 9-class targets) and would have deployed a stale threshold with the old MLLM "
     "pinned -- all without a single error message. (2) The dominant failure mode throughout was SILENT "
     "degradation: every one of the 17 defects either produced a wrong-but-plausible number or quietly "
     "skipped work. (3) The recurring root cause on the implementation side was validating a DIFFERENT "
     "path than production -- a results filename that was never written, a smoke that trained a "
     "substitute config, guard tests that set the variable whose default was broken, a metric computed "
     "from the wrong score. The durable lesson is to assert equivalence against the production code "
     "path numerically (comparing the new validation score to ensemble.py and getting 0.0 was worth more "
     "than any amount of reading), and to exercise each guard's FAILURE path. (4) Deviations from the "
     "Exp-18/19 recipe are now deliberate and listed: whole-frame pixels, AUC-on-production-score "
     "selection, active JPEG augmentation, deterministic GSD. They are expected to help but are "
     "UNMEASURED -- the honest test is the post-training axon1 frontier (FR98/FR99 and hard-real "
     "recall on R_13/R_15) against the incumbent Qwen3.5-4B-from-scratch (testset AUC 0.9996, axon1 AUC "
     "0.9993, FR99 0.9927). (5) Calibration guidance: fit the threshold on the hardest representative "
     "set (axon1, 614k, which contains the hard reals) rather than the identity-sharing testset, and "
     "consider DEPLOY_FLOOR=0.99 -- on an easy slice the smallest-tau-meeting-the-floor rule selects a "
     "threshold with no margin (a 240-frame rehearsal fitted 0.0621 where the battle-tested value is "
     "0.2192).")

NEW = "/tmp/claude-1001/exp21_new.pdf"; os.makedirs(os.path.dirname(NEW), exist_ok=True); pdf.output(NEW)
shutil.copy(EXP, EXP + ".bak21")
w = PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP, "wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages)")
