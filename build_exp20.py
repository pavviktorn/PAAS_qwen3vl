#!/usr/bin/env python3
"""Experiment 20: Qwen3.5-9B MIDS head -- FROM-SCRATCH vs warm-start from the 4B-from-scratch head.
Exp 18 showed warm-starting from a DIFFERENT-MLLM-era (axon0/LLaVA) head hurt. This tests a SAME-family
warm-start: init the 9B head from the deployed 4B-from-scratch head, same 9B answer data / recipe
(AUC-selected). Result: the warm-start IMPROVES the testset but DEGRADES held-out axon1 -- a textbook
testset-overfit, since it inherits the 4B head's calibration on the identity-shared testset. From-scratch
generalizes better on axon1; neither beats the 4B-from-scratch. Appends to EXPERIMENTS.pdf."""
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
h1("Experiment 20  -  Qwen3.5-9B MIDS head: from-scratch vs 4B-head warm-start   [DONE, 2026-08-11]")
para("Exp 18 showed warm-starting the MIDS head from a DIFFERENT-MLLM-era head (axon0/LLaVA) hurt -- it "
     "imported a CLIP calibration that over-fired on hard reals. This asks whether a SAME-family warm-start "
     "helps instead: initialize the 9B head from the deployed 4B-from-scratch head, then train on the 9B's "
     "OWN answer data with the identical recipe (LR 1e-5, 5 ep, AUC-selected). No MLLM change; the cached "
     "9B answers are re-scored with each head. The warm-start converged FASTER (val-AUC 0.9979 @ step5000 vs "
     "the from-scratch 0.9944) and to a higher best val-AUC (0.9993 vs 0.9989) -- but val-AUC is measured on "
     "the testset, which shares the 6 real identities with the head's calibration.")

h2("Table A - clean testset (full 30,194 frames), per-identity REAL-recall @0.5")
table(["9B MIDS head","AUC","real-rec","R_10","R_12","R_13","R_15","R_16","R_17"],
      [["FROM-SCRATCH","0.9986","98.90","99.7","100.0","96.7","97.7","99.8","100.0"],
       ["warm-start (4B head)","0.9994","99.80","100.0","99.9","99.9","99.7","99.9","100.0"]],
      [140,44,50,40,40,40,40,40,42],fs=8,boldrows=(1,))
para("On the testset the WARM-START wins -- AUC 0.9994 vs 0.9986, and it lifts exactly the hard identities "
     "the from-scratch head trails on: R_13 96.7->99.9, R_15 97.7->99.7. It inherits the 4B head's calibration "
     "on these shared identities.",8)

h2("Table B - held-out axonlabs_1 (614,029 frames), frontier fake-recall @ real-floor")
table(["9B MIDS head","AUC","real-rec","FR90","FR95","FR98","FR99"],
      [["FROM-SCRATCH","0.9992","99.54","99.83","99.69","99.47","99.22"],
       ["warm-start (4B head)","0.9985","99.17","99.72","99.49","99.13","98.75"]],
      [150,52,54,50,50,50,50],fs=8,boldrows=(0,))
para("On held-out axon1 the direction REVERSES: the from-scratch head is better on EVERY frontier metric -- "
     "AUC 0.9992 vs 0.9985, FR99 99.22 vs 98.75, FR98 99.47 vs 99.13, FR95/FR90 also higher. The warm-start's "
     "testset gain does NOT transfer; it trades held-out generalization for testset fit.",8)

h2("Conclusion")
para("(1) Warm-starting the 9B head from the same-family 4B-from-scratch head is a TEXTBOOK testset-overfit: "
     "it improves the testset (AUC 0.9994, R_13/R_15 recovered) but DEGRADES the truly-held-out axon1 frontier "
     "(FR99 99.22 -> 98.75; every FR floor and AUC lower). The head memorizes the 4B's identity-shared "
     "calibration -- which is exactly what the testset rewards and axon1 does not. (2) FROM-SCRATCH, learning "
     "only from the 9B's own answer distribution, generalizes better on axon1 -- the deployment metric -- so it "
     "remains the correct head recipe (reinforcing Exp 18 from the opposite direction: whether the warm-start "
     "source is a WORSE or a BETTER head, from-scratch still wins the held-out frontier). (3) Neither 9B head "
     "beats the 4B-from-scratch on axon1 (AUC 0.9993 / FR99 0.9927), and the 9B is ~1.4x slower -- so Qwen3.5-4B "
     "(from-scratch) stays the project best and the deployment choice. (axon1 carries ~3pp video-frame decode "
     "noise applied equally; the from-scratch vs warm-start axon1 gaps are small but uniformly one-directional.)")

NEW="/tmp/claude-1001/exp20_new.pdf"; os.makedirs(os.path.dirname(NEW),exist_ok=True); pdf.output(NEW)
shutil.copy(EXP,EXP+".bak20")
w=PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP,"wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages)")
