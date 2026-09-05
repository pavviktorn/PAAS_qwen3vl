#!/usr/bin/env python3
"""Experiment 19: Qwen3.5-9B FFAA -- does a bigger MLLM help? Full 4B-best recipe (r64/a128 base LoRA +
conditioning-distillation + FROM-SCRATCH, AUC-selected MIDS head). Result: the 9B has the LOWEST MLLM
eval-loss of the project yet gives NO downstream FFAA gain -- it ties the 4B-from-scratch on held-out
axon1 and is slightly behind on the testset, at ~1.4x slower inference. 4B-from-scratch stays best.
Appends to EXPERIMENTS.pdf."""
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
h1("Experiment 19  -  Qwen3.5-9B FFAA: does a bigger MLLM help?   [DONE, 2026-08-10]")
para("Tests whether scaling the FFAA MLLM from Qwen3.5-4B to Qwen3.5-9B improves the deployed detector, "
     "using the SAME 4B-best recipe end-to-end: base LoRA (r=64/alpha=128, 3 ep, reals x2.44, merger "
     "full-FT) + 700k conditioning-distillation continuation (gentle LRs) -> merge, then regenerate the "
     "MIDS answer data with the 9B and train the MIDS head FROM SCRATCH (LR 1e-5, 5 ep) selected by AUC "
     "(the Exp-17/18 lesson). enable_thinking=False throughout. The question: the 9B has a much lower "
     "MLLM eval-loss -- does that convert into better axon1 FR99 / hard-real recall, and is it worth the "
     "extra inference cost?")

h2("MLLM training / gate")
para("Qwen3.5-9B eval_loss 0.1166 (base) / 0.1158 (distilled) -- the LOWEST of any MLLM in this log "
     "(vs 4B 0.133, 8B 0.147, Qwen3-VL-4B 0.177): best answer quality by LM loss. Gate (5k, "
     "enable_thinking=False): parse 100.00%, verdict 99.98%, compliance real 99.94 / fake 100.00 -- "
     "matches the 4B's perfect gate. From-scratch MIDS head best testset-val AUC 0.9989 (AUC-selected).")

h2("Table A - clean testset (per-identity REAL-recall @0.5), 9B vs the 4B-from-scratch incumbent")
table(["MLLM (from-scratch head)","AUC","real-rec","R_10","R_12","R_13","R_15","R_16","R_17"],
      [["Qwen3.5-4B","0.9996","99.39","99.9","99.9","98.8","99.0","99.5","100.0"],
       ["Qwen3.5-9B","0.9985","98.86","99.7","100.0","95.8","97.8","99.9","100.0"]],
      [128,46,50,42,42,42,42,42,44],fs=8,boldrows=(0,))
para("On the clean testset the 9B is slightly BEHIND: AUC 0.9985 vs 0.9996, and it loses ground on the "
     "hard identities -- R_13 95.8 vs 98.8 (-3.0), R_15 97.8 vs 99.0 (-1.2).",8)

h2("Table B - held-out axonlabs_1 (614,029 frames), frontier fake-recall @ real-floor")
table(["MLLM (from-scratch head)","AUC","real-rec","FR90","FR95","FR98","FR99"],
      [["Qwen3.5-4B","0.9993","99.02","99.85","99.69","99.47","99.27"],
       ["Qwen3.5-9B","0.9992","99.54","99.83","99.69","99.47","99.22"]],
      [150,52,54,50,50,50,50],fs=8,boldrows=(0,))
para("On held-out axon1 the two are a statistical TIE: AUC 0.9992 vs 0.9993, FR99 99.22 vs 99.27, FR90/95/98 "
     "identical to 2 dp; the 9B is marginally higher on overall real-recall (99.54 vs 99.02). All well "
     "within axon1's ~3pp video-frame decode noise.",8)

h2("Table C - project standing (best head each) + inference speed")
table(["MLLM","testset AUC","axon1 AUC","axon1 FR99","e2e frames/s"],
      [["LLaVA-7B","0.9922","0.9965","0.9687","2.1"],
       ["Qwen3-VL-4B","0.9952","0.9960","0.9526","14.3"],
       ["Qwen3-VL-8B","0.9970","0.9971","0.9700","11.3"],
       ["Qwen3.5-4B (FROM-SCRATCH)","0.9996","0.9993","0.9927","9.3"],
       ["Qwen3.5-9B (FROM-SCRATCH)","0.9985","0.9992","0.9922","6.8"]],
      [190,92,86,84,80],fs=8,boldrows=(3,))
para("9B 3-pass gen 8.94 frames/s -> e2e ~6.8 (slowest capable MLLM; GDN linear-attention + 2.25x params). "
     "4B-from-scratch e2e 9.3 -- ~1.4x faster.",8)

h2("Conclusion")
para("(1) The Qwen3.5-9B is the BEST MLLM by LM metrics -- lowest eval-loss of the project (0.116) and a "
     "perfect gate. (2) But this does NOT convert into a better FFAA detector: on the clean testset it is "
     "slightly BEHIND the 4B-from-scratch (AUC 0.9985 vs 0.9996; hard-real R_13 95.8 vs 98.8, R_15 97.8 vs "
     "99.0), and on held-out axonlabs_1 it TIES within noise (AUC 0.9992 vs 0.9993; FR99 99.22 vs 99.27). "
     "(3) And it is the SLOWEST capable MLLM (e2e 6.8 frames/s, ~1.4x slower than the 4B-from-scratch). "
     "(4) NET: Qwen3.5-4B (from-scratch) REMAINS the project best and the deployment choice -- equal held-out "
     "quality at 1.4x the speed and half the params. LESSON: past the 4B, MLLM scaling gives no FFAA "
     "benefit on these (near-saturated) sets -- the from-scratch head + recipe dominate MLLM size, and a "
     "lower MLLM eval-loss is NOT a downstream FFAA signal. (Same recipe/eval pipeline as Exp 18; axon1 "
     "carries ~3pp decode noise applied equally.)")

NEW="/tmp/claude-1001/exp19_new.pdf"; os.makedirs(os.path.dirname(NEW),exist_ok=True); pdf.output(NEW)
shutil.copy(EXP,EXP+".bak19")
w=PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP,"wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages)")
