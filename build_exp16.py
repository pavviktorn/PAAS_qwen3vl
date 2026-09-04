#!/usr/bin/env python3
"""Experiment 16: FFAA MLLM scale-up Qwen3-VL-4B -> Qwen3-VL-8B-Instruct, highest-accuracy build.
Renders a 3-model frame-level accuracy comparison (LLaVA-7B / Qwen-4B / Qwen-8B, all axon0-only
warm-start, contamination-free) on testset + axonlabs_1, PLUS a single-GPU inference-speed
comparison (native backend, identical FFAA 3-pass protocol, fixed 320-frame subset).
Appends to EXPERIMENTS.pdf (backup first)."""
import json, os, shutil
from fpdf import FPDF, XPos, YPos
from pypdf import PdfReader, PdfWriter

EXP = "/datasets/work/vLLM/temp/EXPERIMENTS.pdf"
R = "/datasets/work/vLLM/temp/PAAS_qwen3vl/runs/eval"
B = "/datasets/work/vLLM/temp/PAAS_qwen3vl/qwen/bench"
TS = json.load(open(f"{R}/compare4_testset.json"))
AX = json.load(open(f"{R}/compare4_axon1.json"))

# speed timings
def T(f): return json.load(open(f"{B}/{f}"))
g8, g4, gl, gm = T("time_qwen8b_gen.json"), T("time_qwen4b_gen.json"), T("time_llava_gen.json"), T("time_mids.json")
def e2e(g): return g["n"] / (g["gen_total"] + gm["mids_total"])
SP = {
 "llava": {"mllm": gl["frames_per_s"], "e2e": e2e(gl)},
 "qwen4b": {"mllm": g4["frames_per_s"], "e2e": e2e(g4)},
 "qwen8b": {"mllm": g8["frames_per_s"], "e2e": e2e(g8)},
}
MIDS_FPS = gm["frames_per_s"]

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
h1("Experiment 16  -  FFAA MLLM scale-up: Qwen3-VL-4B -> Qwen3-VL-8B-Instruct   [DONE, 2026-07-28]")
para("Goal: highest-accuracy FFAA. Rebuilds the MLLM branch on Qwen/Qwen3-VL-8B-Instruct (project "
     "PAAS_qwen3vl) with the proven-best recipe from Exp 15 -- vision tower FROZEN, decoder LoRA "
     "(r64/a128 @2e-5), vision->language merger FULLY fine-tuned (@1e-5 own LR), reals x2.44, 3 ep, "
     "then conditioning-distillation from LLaVA-compliant targets; MIDS T5+CLIP head warm-started "
     "from the axon0-ONLY v2 head (contamination-free) and trained 5 ep, best-by-acc selection. "
     "Compared frame-level to LLaVA-Mistral-7B (deployed FFAA) and the Exp-15 Qwen-4B, all on the "
     "axon0-only warm-start regime. Adds a single-GPU inference-SPEED comparison.")

h2("MLLM training / gate")
para("8B eval_loss 0.1473 (< 4B 0.1771 < LLaVA target ~0.15): the larger MLLM answers better. "
     "Functional gate (eFFAA_ext_eval, 5k): parse 100%, autonomous real-verdict 91.9% (vs 4B 85.7%, "
     "LLaVA 91.9%), conditioning compliance real 99.82 / fake 100.00 after distillation. MIDS head "
     "peaked at step 25000 (val ACC 0.9756 / AUC 0.9972), an epoch-3 LR-decay rebound past the "
     "epoch-1 peak; later epochs overfit (best-by-acc keeps step 25000).")

h2(f"Table A - testset ({TS['n']:,} aligned frames), frame-level")
table(["MLLM (all axon0-ws MIDS)","AUC","ACC","real","fake","FR95","FR98"],
      [["LLaVA-7B (deployed)",f"{TS['llava']['auc']:.4f}",p(TS['llava']['acc']),p(TS['llava']['rr']),p(TS['llava']['fkr']),p(TS['llava']['fr95']),p(TS['llava']['fr98'])],
       ["Qwen3-VL-4B",f"{TS['qwen4b']['auc']:.4f}",p(TS['qwen4b']['acc']),p(TS['qwen4b']['rr']),p(TS['qwen4b']['fkr']),p(TS['qwen4b']['fr95']),p(TS['qwen4b']['fr98'])],
       ["Qwen3-VL-8B",f"{TS['qwen8b']['auc']:.4f}",p(TS['qwen8b']['acc']),p(TS['qwen8b']['rr']),p(TS['qwen8b']['fkr']),p(TS['qwen8b']['fr95']),p(TS['qwen8b']['fr98'])]],
      [152,52,52,52,52,56,56],fs=8,boldrows=(2,))

h2(f"Table B - axonlabs_data_1 ({AX['n']:,} aligned frames), frontier fake-recall @ real-floor")
table(["MLLM (all axon0-ws MIDS)","AUC","90%","95%","98%","99%"],
      [["LLaVA-7B (deployed)",f"{AX['llava']['auc']:.4f}",p(AX['llava']['fr90']),p(AX['llava']['fr95']),p(AX['llava']['fr98']),p(AX['llava']['fr99'])],
       ["Qwen3-VL-4B",f"{AX['qwen4b']['auc']:.4f}",p(AX['qwen4b']['fr90']),p(AX['qwen4b']['fr95']),p(AX['qwen4b']['fr98']),p(AX['qwen4b']['fr99'])],
       ["Qwen3-VL-8B",f"{AX['qwen8b']['auc']:.4f}",p(AX['qwen8b']['fr90']),p(AX['qwen8b']['fr95']),p(AX['qwen8b']['fr98']),p(AX['qwen8b']['fr99'])]],
      [152,58,55,55,55,55],fs=8,boldrows=(2,))

h2("Table C - inference SPEED (single GPU, native backend, identical FFAA 3-pass protocol, 320-frame subset)")
para("Same condition: 1 GPU (RTX PRO 6000), temp 0, max_new_tokens 512, 1 image/prompt, images "
     "pre-loaded (times MLLM compute, not disk). LLaVA -> HF transformers; Qwen -> vLLM (as deployed). "
     "MIDS T5+CLIP head is model-independent (%.1f frames/s). e2e = 3-pass MLLM gen + MIDS scoring." % MIDS_FPS)
table(["MLLM","backend","MLLM frames/s","e2e frames/s","speedup vs LLaVA"],
      [["LLaVA-7B","HF",f"{SP['llava']['mllm']:.2f}",f"{SP['llava']['e2e']:.2f}","1.0x"],
       ["Qwen3-VL-4B","vLLM",f"{SP['qwen4b']['mllm']:.2f}",f"{SP['qwen4b']['e2e']:.2f}",f"{SP['qwen4b']['e2e']/SP['llava']['e2e']:.1f}x"],
       ["Qwen3-VL-8B","vLLM",f"{SP['qwen8b']['mllm']:.2f}",f"{SP['qwen8b']['e2e']:.2f}",f"{SP['qwen8b']['e2e']/SP['llava']['e2e']:.1f}x"]],
      [110,70,110,110,120,],fs=8,boldrows=(2,))

h2("Analysis / conclusion")
best_ts = "8B" if TS['qwen8b']['auc'] >= max(TS['llava']['auc'],TS['qwen4b']['auc']) else "?"
para(f"(1) ACCURACY: Qwen-8B is the best of the three on the testset on EVERY metric "
     f"(AUC {TS['qwen8b']['auc']:.4f} vs 4B {TS['qwen4b']['auc']:.4f} vs LLaVA {TS['llava']['auc']:.4f}; "
     f"FR95 {p(TS['qwen8b']['fr95'])} vs {p(TS['qwen4b']['fr95'])} vs {p(TS['llava']['fr95'])}). On the "
     f"cleaner held-out axonlabs_1, 8B AUC {AX['qwen8b']['auc']:.4f} vs 4B {AX['qwen4b']['auc']:.4f} vs "
     f"LLaVA {AX['llava']['auc']:.4f}. (2) SPEED: despite ~2x the params of 4B and ~similar to LLaVA-7B, "
     f"Qwen-8B runs {SP['qwen8b']['e2e']/SP['llava']['e2e']:.1f}x faster end-to-end than LLaVA-7B "
     f"({SP['qwen8b']['mllm']/SP['llava']['mllm']:.1f}x on the MLLM stage) -- the vLLM backend + Qwen "
     "architecture more than offset the size. (3) NET: the 8B is the highest-accuracy FFAA built and is "
     "still multiples faster than the deployed LLaVA-7B; the 4B remains the throughput champion if speed "
     "dominates. All variants contamination-free (axon0-only MIDS warm-start; axon1 never in training).")

NEW="/tmp/claude-1001/exp16_new.pdf"; os.makedirs(os.path.dirname(NEW),exist_ok=True); pdf.output(NEW)
shutil.copy(EXP, EXP+".bak16")
w=PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP,"wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages); backup {EXP}.bak16")
