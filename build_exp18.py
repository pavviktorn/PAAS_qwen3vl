#!/usr/bin/env python3
"""Experiment 18: Qwen3.5-4B MIDS head init ablation -- axon0 WARM-START vs FROM-SCRATCH.
Root-causes the Exp-17 axon1 real-FP collapse (testset & axon1 share the 6 real identities, so it was
an anomaly) to the axon0 warm-start's inherited CLIP calibration, and shows from-scratch init fixes it,
making Qwen3.5-4B the best FFAA of the project on BOTH sets. Appends to EXPERIMENTS.pdf."""
import os, shutil
from fpdf import FPDF, XPos, YPos
from pypdf import PdfReader, PdfWriter
EXP="/datasets/work/vLLM/temp/EXPERIMENTS.pdf"
pdf=FPDF(unit="pt",format=(612,792)); pdf.set_auto_page_break(True,margin=40); pdf.set_margins(40,40,40)
CW=532; NX,NY=XPos.LMARGIN,YPos.NEXT
def h1(t): pdf.set_font("Helvetica","B",13); pdf.multi_cell(CW,16,t,new_x=NX,new_y=NY); pdf.ln(3)
def h2(t): pdf.ln(3); pdf.set_font("Helvetica","B",10); pdf.multi_cell(CW,13,t,new_x=NX,new_y=NY)
def para(t,fs=9): pdf.set_font("Helvetica","",fs); pdf.multi_cell(CW,12.5,t,new_x=NX,new_y=NY)
def table(headers,rows,widths,fs=8,align0="L",boldrows=()):
    pdf.set_font("Helvetica","B",fs); pdf.set_fill_color(228,228,228)
    for h,w in zip(headers,widths): pdf.cell(w,14,h,border=1,align="C",fill=True)
    pdf.ln()
    for ri,r in enumerate(rows):
        pdf.set_font("Helvetica","B" if ri in boldrows else "",fs)
        for i,(c,w) in enumerate(zip(r,widths)): pdf.cell(w,12,str(c),border=1,align=(align0 if i==0 else "C"))
        pdf.ln()
pdf.add_page()
h1("Experiment 18  -  Qwen3.5-4B MIDS head init: axon0 warm-start vs FROM-SCRATCH   [DONE, 2026-08-04]")
para("Exp 17 found Qwen3.5-4B best on the testset but collapsing on axonlabs_1 (AUC 0.9919, FR99 "
     "0.7464). KEY REALIZATION (per review): the testset was SAMPLED FROM axonlabs_1 -- both share the "
     "same 6 real identities (R_10/12/13/15/16/17) -- so the collapse was an ANOMALY, not a "
     "generalization limit. Diagnosis: it is localized to hard/blurry REAL frames of R_13 & R_15; on "
     "those frames Qwen3.5-4B's MLLM answers are CORRECT ('real, natural depth') yet the MIDS head "
     "scores them fake. So the fault is the HEAD, not the MLLM -- specifically the axon0 (LLaVA-era) "
     "warm-start's CLIP calibration, which over-fires 'fake' on blurry reals and which testset-ACC "
     "selection masks. FIX: train the MIDS head FROM SCRATCH (random init, no warm-start) on "
     "Qwen3.5-4B's own answer distribution. Same recipe otherwise (LR 1e-5, 5 ep, best-by-acc). No "
     "MLLM change and no re-generation -- the cached testset/axon1 answers are re-scored with the new head.")

h2("Head-vs-head, per-identity REAL-recall @0.5 (clean testset, 30k frames)")
table(["MIDS head","AUC","real-rec","R_10","R_12","R_13","R_15","R_16","R_17"],
      [["axon0 warm-start","0.9972","97.54","99.5","99.3","89.7","95.8","99.6","99.9"],
       ["FROM-SCRATCH","0.9996","99.39","99.9","99.9","98.8","99.0","99.5","100.0"]],
      [110,50,52,44,44,44,44,44,46],fs=8,boldrows=(1,))
para("From-scratch recovers exactly the collapsing identities: R_13 +9.0, R_15 +3.2; AUC 0.9972->0.9996.",8)

h2("Held-out axonlabs_1 (614,029 frames, identical scoring pipeline for both heads)")
table(["MIDS head","AUC","real-rec","FR90","FR95","FR98","FR99"],
      [["axon0 warm-start","0.9919","94.01","99.83","99.33","93.46","74.64"],
       ["FROM-SCRATCH","0.9993","99.02","99.85","99.69","99.47","99.27"]],
      [130,55,58,55,55,55,55],fs=8,boldrows=(1,))
para("The frontier collapse is ELIMINATED: FR99 74.64 -> 99.27 (+24.6), FR98 93.46 -> 99.47, "
     "AUC 0.9919 -> 0.9993; R_13 84.6->99.4, R_15 81.5->98.5 real-recall.",8)

h2("Project standing -- Qwen3.5-4B (from-scratch head) is now the best FFAA on BOTH sets")
table(["MLLM (best head)","testset AUC","axon1 AUC","axon1 FR99"],
      [["LLaVA-7B","0.9922","0.9965","0.9687"],
       ["Qwen3-VL-4B","0.9952","0.9960","0.9526"],
       ["Qwen3-VL-8B","0.9970","0.9971","0.9700"],
       ["Qwen3.5-4B (warm-start)","0.9972","0.9919","0.7464"],
       ["Qwen3.5-4B (FROM-SCRATCH)","0.9996","0.9993","0.9927"]],
      [200,100,90,90],fs=8,boldrows=(4,))

h2("Conclusion")
para("(1) The Exp-17 'held-out weakness' of Qwen3.5-4B was NOT a real generalization failure -- it was "
     "the axon0 MIDS-head warm-start importing LLaVA/CLIP calibration that over-fires on hard reals, "
     "hidden by testset-ACC selection. (2) FROM-SCRATCH head init fixes it completely: Qwen3.5-4B "
     "becomes the best FFAA of the project on BOTH the testset (AUC 0.9996) AND held-out axonlabs_1 "
     "(AUC 0.9993, FR99 0.9927) -- beating the 8B at half the params. (3) LESSON: warm-starting the "
     "MIDS head from a different-MLLM-era head can bias it against hard reals; for a strong new MLLM, "
     "train the head from scratch on its own answer distribution, and do NOT rely on ACC selection "
     "when the val set under-samples hard identities. (Note: axon1 video-frame scoring carries ~3pp "
     "decode noise, applied equally to both heads; the 24-point FR99 gain is far above it.)")

NEW="/tmp/claude-1001/exp18_new.pdf"; os.makedirs(os.path.dirname(NEW),exist_ok=True); pdf.output(NEW)
shutil.copy(EXP,EXP+".bak18")
w=PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP,"wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages)")
