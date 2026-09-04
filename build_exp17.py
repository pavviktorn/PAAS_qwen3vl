#!/usr/bin/env python3
"""Experiment 17: FFAA MLLM swap to Qwen3.5-4B (hybrid linear-attention), 4-way comparison.
Renders frame-level accuracy (testset + axonlabs_1) and single-GPU inference speed for
LLaVA-7B / Qwen3-VL-4B / Qwen3-VL-8B / Qwen3.5-4B (all axon0-only warm-start MIDS head).
Appends to EXPERIMENTS.pdf (backup first)."""
import json, os, shutil
from fpdf import FPDF, XPos, YPos
from pypdf import PdfReader, PdfWriter

EXP = "/datasets/work/vLLM/temp/EXPERIMENTS.pdf"
R = "/datasets/work/vLLM/temp/PAAS_qwen3vl/runs/eval"
B = "/datasets/work/vLLM/temp/PAAS_qwen3vl/qwen/bench"
TS = json.load(open(f"{R}/compare4_testset_v2.json"))
AX = json.load(open(f"{R}/compare4_axon1_v2.json"))
def T(f): return json.load(open(f"{B}/{f}"))
g35,gm35 = T("time_qwen35_4b_gen.json"), T("time_mids_qwen35.json")
g8,g4,gl,gm = T("time_qwen8b_gen.json"),T("time_qwen4b_gen.json"),T("time_llava_gen.json"),T("time_mids.json")
def e2e(g,m): return g["n"]/(g["gen_total"]+m["mids_total"])
SP = {"llava":(gl["frames_per_s"],e2e(gl,gm)), "qwen4b":(g4["frames_per_s"],e2e(g4,gm)),
      "qwen8b":(g8["frames_per_s"],e2e(g8,gm)), "qwen35_4b":(g35["frames_per_s"],e2e(g35,gm35))}
LL=SP["llava"][1]

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
ORD=[("LLaVA-7B","llava"),("Qwen3-VL-4B","qwen4b"),("Qwen3-VL-8B","qwen8b"),("Qwen3.5-4B","qwen35_4b")]

pdf.add_page()
h1("Experiment 17  -  FFAA MLLM swap: -> Qwen3.5-4B (hybrid linear-attention)   [DONE, 2026-08-01]")
para("Swaps the FFAA MLLM to Qwen/Qwen3.5-4B -- a HYBRID model (24 Mamba/GDN linear-attention layers "
     "+ 8 full-attention layers), unlike the pure-transformer Qwen3-VL. Adaptations: LoRA extended to "
     "the linear-attention token-mixing (in_proj_qkv/z + out_proj) across all 32 layers (200 modules, "
     "88.2M trainable = HALF the Qwen3-VL-4B's 175M); dynamic merger discovery (visual.merger only, no "
     "deepstack); fla GDN kernel installed (2x train speedup). CRITICAL FIX: Qwen3.5 is a reasoning "
     "model whose thinking mode broke the FFAA JSON (gate parse 66%); enable_thinking=False at "
     "inference (matching the empty-<think></think> training targets) restored parse to 100% -- no "
     "retraining. Same recipe otherwise: reals x2.44, 3ep base + 700k conditioning-distillation, MIDS "
     "head warm-started from the axon0-only v2 head, 5ep, best-by-acc.")

h2("MLLM training / gate")
para("Qwen3.5-4B eval_loss 0.1326 (base) / 0.1336 (distilled) -- the LOWEST of any MLLM in this log "
     "(vs 8B 0.147, Qwen3-VL-4B 0.177): best answer quality at half the trainable params. Gate (5k, "
     "enable_thinking=False): parse 100.00%, verdict 99.98%, compliance real 99.90 / fake 100.00 -- "
     "the best gate of the project. MIDS head val-ACC peaked 0.9808 @ step35000 (epoch-4 rebound).")

h2(f"Table A - testset ({TS['n']:,} aligned frames), frame-level")
table(["MLLM (all axon0-ws MIDS)","AUC","ACC","real","fake","FR95","FR98"],
      [[nm, f"{TS[k]['auc']:.4f}", p(TS[k]['acc']), p(TS[k]['rr']), p(TS[k]['fkr']), p(TS[k]['fr95']), p(TS[k]['fr98'])] for nm,k in ORD],
      [150,54,52,52,52,54,54],fs=8,boldrows=(3,))

h2(f"Table B - axonlabs_data_1 ({AX['n']:,} aligned frames), frontier fake-recall @ real-floor")
table(["MLLM (all axon0-ws MIDS)","AUC","90%","95%","98%","99%"],
      [[nm, f"{AX[k]['auc']:.4f}", p(AX[k]['fr90']), p(AX[k]['fr95']), p(AX[k]['fr98']), p(AX[k]['fr99'])] for nm,k in ORD],
      [150,58,55,55,55,55],fs=8,boldrows=(2,))  # 8B is the axon1 frontier winner

h2("Table C - inference SPEED (single GPU, native backend, identical FFAA 3-pass protocol, 320 frames)")
para("LLaVA -> HF transformers; Qwen -> vLLM (as deployed, enable_thinking=False). MIDS head ~%.0f "
     "frames/s (model-independent). e2e = 3-pass MLLM gen + MIDS scoring." % gm["frames_per_s"])
table(["MLLM","backend","MLLM frames/s","e2e frames/s","speedup vs LLaVA"],
      [[nm, ("HF" if k=="llava" else "vLLM"), f"{SP[k][0]:.2f}", f"{SP[k][1]:.2f}", f"{SP[k][1]/LL:.1f}x"] for nm,k in ORD],
      [110,70,110,110,120],fs=8,boldrows=(3,))

h2("Analysis / conclusion")
para(f"(1) IN-DISTRIBUTION (testset): Qwen3.5-4B is the BEST -- AUC {TS['qwen35_4b']['auc']:.4f} / "
     f"ACC {p(TS['qwen35_4b']['acc'])} / FR98 {p(TS['qwen35_4b']['fr98'])}, edging the 8B "
     f"(AUC {TS['qwen8b']['auc']:.4f} / FR98 {p(TS['qwen8b']['fr98'])}) at HALF the params, plus the "
     "lowest MLLM eval_loss (0.133) and a perfect gate. (2) HELD-OUT (axonlabs_1) -- THE IMPORTANT "
     f"CAVEAT: Qwen3.5-4B does NOT generalize as well. It leads on ACC@0.5 ({p(AX['qwen35_4b']['acc'])}), "
     f"fake-recall ({p(AX['qwen35_4b']['fkr'])}) and FR90 ({p(AX['qwen35_4b']['fr90'])}), but has the "
     f"LOWEST AUC ({AX['qwen35_4b']['auc']:.4f}) and its frontier COLLAPSES at strict real-floors "
     f"(FR98 {p(AX['qwen35_4b']['fr98'])}, FR99 {p(AX['qwen35_4b']['fr99'])} vs the 8B's "
     f"{p(AX['qwen8b']['fr98'])}/{p(AX['qwen8b']['fr99'])}). Cause: real-recall is only "
     f"{p(AX['qwen35_4b']['rr'])} (vs 8B {p(AX['qwen8b']['rr'])}) -- the model is biased toward 'fake', "
     "so meeting a 98-99% real floor forces the threshold down and fake-recall craters (the real-FP "
     "wall). The 8B is the MOST ROBUST across both sets. Likely contributors: testset is the head's "
     "val-selection set (mild overfit), and best-by-ACC picked a testset-tuned checkpoint (its step-20k "
     "checkpoint had higher testset AUC 0.9983) -- an AUC-selected or calibrated head may recover the "
     f"axon1 tail. (3) SPEED: Qwen3.5-4B is the SLOWEST Qwen (e2e {SP['qwen35_4b'][1]:.1f} vs 4B "
     f"{SP['qwen4b'][1]:.1f} vs 8B {SP['qwen8b'][1]:.1f} frames/s; GDN linear-attention overhead), still "
     f"{SP['qwen35_4b'][1]/LL:.1f}x faster than LLaVA-7B. (4) NET: Qwen3.5-4B wins in-distribution and on "
     "raw ACC/fake-recall but shows a held-out real-FP weakness; Qwen3-VL-8B is the most robust on the "
     "held-out frontier; Qwen3-VL-4B is the fastest. All contamination-free (axon0-only MIDS warm-start; "
     "axon1 never in training).")

NEW="/tmp/claude-1001/exp17_new.pdf"; os.makedirs(os.path.dirname(NEW),exist_ok=True); pdf.output(NEW)
shutil.copy(EXP, EXP+".bak17")
w=PdfWriter()
for pg in PdfReader(EXP).pages: w.add_page(pg)
for pg in PdfReader(NEW).pages: w.add_page(pg)
with open(EXP,"wb") as fh: w.write(fh)
print(f"appended -> {EXP} (now {len(PdfReader(EXP).pages)} pages); backup {EXP}.bak17")
