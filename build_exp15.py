#!/usr/bin/env python3.12
"""Experiment 15 (comprehensive): FFAA MLLM swap LLaVA-Mistral-7B -> Qwen3-VL-4B, with a MIDS-head
warm-start ablation (from-scratch vs axon0-only warm-start), all contamination-free. Reads the 3-way
comparison jsons. Existing pages preserved (backup first)."""
import json, os, shutil
from fpdf import FPDF, XPos, YPos
from pypdf import PdfReader, PdfWriter

EXP = "/datasets/work/vLLM/temp/EXPERIMENTS.pdf"
RUN = "/datasets/work/vLLM/temp/PAAS_qwen3vl/runs/eval"
TS = json.load(open(f"{RUN}/compare3_testset.json"))
AX = json.load(open(f"{RUN}/compare3_axon1.json"))

pdf = FPDF(unit="pt", format=(612, 792)); pdf.set_auto_page_break(True, margin=40); pdf.set_margins(40, 40, 40)
CW = 532; NX, NY = XPos.LMARGIN, YPos.NEXT
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
def p(x): return f"{x*100:.2f}"

pdf.add_page()
h1("Experiment 15  -  FFAA MLLM swap: LLaVA-Mistral-7B -> Qwen3-VL-4B   [DONE, 2026-07-22]")
para("Rebuilds the FFAA (MLLM + MIDS) branch on Qwen/Qwen3-VL-4B-Instruct (standalone project "
     "PAAS_qwen3vl) and compares it frame-level to the deployed FFAA whose MLLM is LLaVA-Mistral-7B "
     "trained on axonlabs_0. Includes a MIDS-head-init ABLATION: (a) from scratch, (b) warm-start "
     "from the axon0-only v2 MIDS head. All variants are contamination-free -- the Qwen MIDS data "
     "(mids_first_half) and both warm-start heads never touch axonlabs_1 (an earlier variant "
     "warm-started from the axon1-exposed v3 head; it was discarded as contaminated).")

h2("Pipeline (run_finetuning.sh): 3 chained steps, 2 environments")
para("Step 1 (venv, tf5): LoRA-finetune Qwen3-VL-4B on eFFAA_ext (vision FROZEN, decoder LoRA r32/a48 "
     "@1e-4, merger FULLY fine-tuned @2e-5 own LR, reals x2.44, 3 ep) + conditioning-distillation "
     "continuation (LLaVA-compliant targets) to fix pass-2/3 compliance. Step 2 (venv, vLLM): generate "
     "984,538 MIDS records. Step 3 (global, tf4.37.2): train the T5+CLIP MIDS head (from-scratch OR "
     "warm-started).")

h2("Functional gate (eFFAA_ext_eval, 5k) -- MLLM answer quality")
table(["MLLM","parse","verdict","comply(real)","comply(fake)","gen speed"],
      [["LLaVA-7B","93.90","91.86","83.26","99.94","1.6 img/s (HF)"],
       ["Qwen3-VL-4B","100.00","85.66","96.62","99.90","5.5 img/s (vLLM)"]],
      [100,55,58,80,80,101],fs=8,boldrows=(1,))
para("MLLM stage favours Qwen: 100% parse (no error-recovery pass), higher conditioned compliance, "
     "~3.4x faster generation, at 4B vs 7B params.",8)

h2(f"Table A - testset ({TS['n']:,} aligned frames), frame-level")
table(["MIDS head / model","AUC","ACC","real","fake","FR95","FR98"],
      [["axon0-FFAA (LLaVA-7B)",f"{TS['llava']['auc']:.4f}",p(TS['llava']['acc']),p(TS['llava']['rr']),p(TS['llava']['fkr']),p(TS['llava']['fr95']),p(TS['llava']['fr98'])],
       ["Qwen-4B  from-scratch",f"{TS['scratch']['auc']:.4f}",p(TS['scratch']['acc']),p(TS['scratch']['rr']),p(TS['scratch']['fkr']),p(TS['scratch']['fr95']),p(TS['scratch']['fr98'])],
       ["Qwen-4B  axon0-warmstart",f"{TS['axon0ws']['auc']:.4f}",p(TS['axon0ws']['acc']),p(TS['axon0ws']['rr']),p(TS['axon0ws']['fkr']),p(TS['axon0ws']['fr95']),p(TS['axon0ws']['fr98'])]],
      [150,54,52,52,52,54,54],fs=8,boldrows=(2,))

h2(f"Table B - axonlabs_data_1 ({AX['n']:,} aligned frames), frontier fake-recall @ real-floor")
table(["MIDS head / model","AUC","90%","95%","98%","99%"],
      [["axon0-FFAA (LLaVA-7B)",f"{AX['llava']['auc']:.4f}",p(AX['llava']['fr90']),p(AX['llava']['fr95']),p(AX['llava']['fr98']),p(AX['llava']['fr99'])],
       ["Qwen-4B  from-scratch",f"{AX['scratch']['auc']:.4f}",p(AX['scratch']['fr90']),p(AX['scratch']['fr95']),p(AX['scratch']['fr98']),p(AX['scratch']['fr99'])],
       ["Qwen-4B  axon0-warmstart",f"{AX['axon0ws']['auc']:.4f}",p(AX['axon0ws']['fr90']),p(AX['axon0ws']['fr95']),p(AX['axon0ws']['fr98']),p(AX['axon0ws']['fr99'])]],
      [150,58,56,56,56,56],fs=8)

h2("Analysis / conclusion")
para(f"(1) With a MATCHED warm-start regime (both starting from the axon0 head), Qwen-4B EDGES the "
     f"LLaVA-7B baseline on the testset (AUC {TS['axon0ws']['auc']:.4f} vs {TS['llava']['auc']:.4f}, "
     f"FR95 {p(TS['axon0ws']['fr95'])} vs {p(TS['llava']['fr95'])}) and is effectively TIED on axon1 "
     f"(AUC {AX['axon0ws']['auc']:.4f} vs {AX['llava']['auc']:.4f}; LLaVA slightly better in the "
     "extreme tail FR98/99). (2) The MIDS-head init matters a lot: from-scratch trails on the testset "
     f"(AUC {TS['scratch']['auc']:.4f}, FR98 {p(TS['scratch']['fr98'])}) though it is competitive on "
     "axon1 -- so the head's starting point, not just the MLLM, drives the headline number. (3) NET: "
     "Qwen-4B is on par with / marginally better than LLaVA-7B on detection accuracy, at ~half the "
     "params with a faster, 100%-parse MLLM stage -- an attractive efficiency swap, not a large "
     "accuracy jump. (4) Caveat: testset is the head's val-selection set (mild coupling for the Qwen "
     "variants); axon1 is the cleaner held-out comparison, where all three are within ~0.001 AUC. "
     "Project: PAAS_qwen3vl. The discarded v3-warmstart variant scored a contaminated flat ~0.9998 on "
     "axon1 -- excluded.")

NEW="/tmp/claude-1001/exp15_new.pdf"; os.makedirs(os.path.dirname(NEW),exist_ok=True); pdf.output(NEW)
shutil.copy(EXP, EXP+".bak15")
w=PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP,"wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages); backup {EXP}.bak15")
