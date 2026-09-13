#!/usr/bin/env python3.12
"""Re-score CACHED axon1 answers (Qwen3.5-4B) with MULTIPLE MIDS-head checkpoints in ONE decode pass,
to find whether a different head checkpoint recovers the hard-real (R_13/R_15) frontier that the
testset-ACC-selected best.pth (step 35000) loses. No MLLM re-generation -- only the T5+CLIP head runs.

Scores ALL real frames + a bounded fake sample; reports per-head AUC / real-recall / FR@real-floors
and per-identity real-recall. GLOBAL python3.12 / transformers 4.37.2 stack.

  CUDA_VISIBLE_DEVICES=0 python3.12 qwen/multi_head_rescore.py --answers-glob 'runs/eval/axon1/qwen35_4b_axon1_*.json' \
     --heads best,1,2,3,4,5 --head-dir runs/mids_head/mids_v1_20260731_064638 --fake-cap 60000
"""
import argparse, glob, json, os, re, sys, collections
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import rankdata
from transformers import T5Tokenizer, CLIPProcessor
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _ROOT)
from mids.mids_arch import MIDS
from mids.selector import make_decision_batch
from utils.mids_utils import flatten_tuple_list
from qwen.gen_score_axon1 import iter_decoded_prefetch      # fast video-grouped decode

def auc(s,y):
    p,n=int((y==1).sum()),int((y==0).sum())
    return float((rankdata(s)[y==1].sum()-p*(p+1)/2.0)/(p*n)) if p and n else float("nan")
def fr_at_real(s,y,R):
    real=np.sort(s[y==0]); fake=s[y==1]; nr=len(real)
    if not nr or not len(fake): return float("nan")
    k=min(max(int(np.ceil(R*nr)),1),nr); t=np.nextafter(real[k-1],np.inf)
    return float((fake>=t).mean())
def realid(p):
    m=re.search(r'/id_(R_\d+)/',p); return m.group(1) if m else "?"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--answers-glob",required=True); ap.add_argument("--head-dir",required=True)
    ap.add_argument("--heads",default="best,1,2,3,4,5"); ap.add_argument("--fake-cap",type=int,default=60000)
    ap.add_argument("--device",default="cuda:0"); ap.add_argument("--batch-size",type=int,default=64)
    a=ap.parse_args()
    CLIP=os.environ.get("PAAS_CLIP",os.path.join(_ROOT,"base_models","clip-vit-large-patch14-336"))
    T5=os.environ.get("PAAS_T5",os.path.join(_ROOT,"base_models","t5-base"))

    # load cached answers -> frame_key -> record; select all reals + capped fakes
    ans={}
    for f in sorted(glob.glob(a.answers_glob)):
        for r in json.load(open(f)): ans[r["frame_key"]]=r
    reals=[k for k,r in ans.items() if int(r["cls_label"])==0]
    fakes=[k for k,r in ans.items() if int(r["cls_label"])==1]
    if a.fake_cap and len(fakes)>a.fake_cap:
        fakes=fakes[::max(1,len(fakes)//a.fake_cap)][:a.fake_cap]   # deterministic stride sample
    keys=reals+fakes
    frames=[(k,int(ans[k]["cls_label"])) for k in keys]
    print(f"scoring {len(reals):,} reals + {len(fakes):,} fakes = {len(frames):,} frames with heads {a.heads}",flush=True)

    clip=CLIPProcessor.from_pretrained(CLIP); tok=T5Tokenizer.from_pretrained(T5,use_fast=False,legacy=False)
    heads={}
    for h in a.heads.split(","):
        p=os.path.join(a.head_dir,f"{h}.pth"); m=MIDS(768,image_model_path=CLIP,text_model_path=T5)
        sd=m.state_dict(); ft={k.replace("module.",""):v for k,v in torch.load(p,map_location="cpu").items()}
        sd.update(ft); m.load_state_dict(sd); heads[h]=m.to(a.device).eval()
    print("heads loaded:",list(heads),flush=True)

    # accumulate per-head scores aligned to `keys`
    out={h:[] for h in heads}; ys=[]; ids=[]
    bimg,btxt,bres,bmeta=[],[],[],[]
    def flush():
        if not bimg: return
        images=torch.cat(bimg,0).to(a.device)
        idsr=tok(flatten_tuple_list(btxt),return_tensors="pt",padding="longest",max_length=tok.model_max_length,truncation=True).to(a.device)
        res=flatten_tuple_list(bres)
        with torch.no_grad():
            for h,m in heads.items():
                logits=m(idsr,images,None,len(bimg),1,1)["logits"]; sc=F.softmax(logits,dim=2)
                _,_,_,forg=make_decision_batch(res,sc)
                out[h].extend(float(x) for x in forg)
        for (k,y) in bmeta: ys.append(y); ids.append(realid(k))
        bimg.clear();btxt.clear();bres.clear();bmeta.clear()
    done=0
    for key,y,im in iter_decoded_prefetch(frames):
        r=ans[key]["answers"][:3]
        bimg.append(clip(images=im,return_tensors="pt")["pixel_values"])
        btxt.append(tuple(x["content"] for x in r)); bres.append(tuple(x["result"] for x in r)); bmeta.append((key,y))
        if len(bimg)>=a.batch_size:
            flush(); done+=a.batch_size
            if (done//a.batch_size)%100==0: print(f"  {done:,}/{len(frames):,}",flush=True)
    flush()
    y=np.array(ys); ids=np.array(ids)
    print(f"\n{'head':>6} {'AUC':>8} {'real@.5':>8} {'fake@.5':>8} {'FR95':>7} {'FR98':>7} {'FR99':>7}  | R_13  R_15")
    for h in heads:
        s=np.array(out[h])
        rr=float((s[y==0]<0.5).mean()); fk=float((s[y==1]>=0.5).mean())
        def idrec(name):
            mask=(y==0)&(ids==name); return (s[mask]<0.5).mean()*100 if mask.any() else float('nan')
        print(f"{h:>6} {auc(s,y):8.4f} {rr*100:8.2f} {fk*100:8.2f} {fr_at_real(s,y,.95)*100:7.2f} "
              f"{fr_at_real(s,y,.98)*100:7.2f} {fr_at_real(s,y,.99)*100:7.2f}  | {idrec('R_13'):5.1f} {idrec('R_15'):5.1f}")

if __name__=="__main__": main()
