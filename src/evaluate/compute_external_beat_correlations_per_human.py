#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluate.compute_curve_correlations import _score_info,_curve,_human_rows,_finite_corr
from src.evaluate.compute_external_measure_correlations import notes,align_pairs

def human_curves(work,onset,score_raw,bb):
 span=float(np.sum(score_raw[1:,0])); out=[]
 for rows in _human_rows(work):
  a=np.asarray(rows,float); ps=float(np.sum(a[1:,0])); out.append(_curve(onset,score_raw,rows,bb,span/ps if ps>0 else 1.0))
 return out

def external_curve(score_m,score,perf_m,perf,pairs,bb=1.0):
 valid=[(pi,si) for pi,si in pairs if 0<=pi<len(perf) and 0<=si<len(score)]
 if len(valid)<10:return None
 by_score=defaultdict(list)
 for pi,si in valid:by_score[si].append(pi)
 sis=np.array(sorted(by_score),int)
 q=np.array([score[i].start/score_m.ticks_per_beat for i in sis],float)
 t=np.array([np.median([perf[j].start/perf_m.ticks_per_beat for j in by_score[i]]) for i in sis],float)
 v=np.array([np.mean([perf[j].velocity for j in by_score[i]]) for i in sis],float)
 # Use the same beat-bin construction as the internal beat evaluator.
 bins=np.floor(q/bb).astype(int); tempo={}; vel={}
 prev=np.r_[q[0],q[:-1]]; dq=q-prev; dt=np.diff(np.r_[t[0],t])
 for i,b in enumerate(bins):
  if i>0 and dq[i]>1e-8 and dt[i]>1e-8:tempo.setdefault(b,[]).append(dq[i]/dt[i])
  vel.setdefault(b,[]).append(v[i])
 keys=sorted(set(tempo)&set(vel))
 return np.asarray(keys),np.asarray([np.mean(tempo[k]) for k in keys]),np.asarray([np.mean(vel[k]) for k in keys])

def run(model,manifest,root):
 d=json.loads(Path(manifest).read_text()); rows=[]; errors=[]
 for item in d['items']:
  source=item['score_source']; work,onset,score_raw=_score_info(source); sp=ROOT/'data/ASAP_processed'/source
  sm,sn=notes(sp); hs=human_curves(work,onset,score_raw,1.0); mt=[];mv=[]
  for pp in item.get('prediction_paths',[]):
   pred=Path(root)/'midis'/Path(pp).name; pm,pn=notes(pred); pairs,err=align_pairs(sp,pred)
   if pairs is None:errors.append({'score':source,'path':str(pred),'error':err});continue
   c=external_curve(sm,sn,pm,pn,pairs)
   if c is None:errors.append({'score':source,'path':str(pred),'error':'too_few_matches'});continue
   pt=[];pv=[]
   for h in hs:
    keys=sorted(set(c[0].tolist())&set(h[0].tolist()))
    if len(keys)<3:continue
    a={int(k):(t,v) for k,t,v in zip(*c)};b={int(k):(t,v) for k,t,v in zip(*h)}
    A=np.asarray([a[k] for k in keys]);B=np.asarray([b[k] for k in keys])
    pt.append(_finite_corr(A[:,0],B[:,0],'pearson'));pv.append(_finite_corr(A[:,1],B[:,1],'pearson'))
   if pt:mt.append(float(np.nanmean(pt)));mv.append(float(np.nanmean(pv)))
  if mt:rows.append({'score_source':source,'n_human':len(hs),'n_samples':len(mt),'n_pairs':len(mt)*len(hs),'tempo':float(np.nanmean(mt)),'velocity':float(np.nanmean(mv))})
 out={'model':model,'unit':'beat','human_aggregation':'mean over individual human performers','correlation':'pearson','n_scores':len(rows),'n_samples':sum(x['n_samples'] for x in rows),'n_pairs':sum(x['n_pairs'] for x in rows),'tempo':float(np.nanmean([x['tempo'] for x in rows])) if rows else np.nan,'velocity':float(np.nanmean([x['velocity'] for x in rows])) if rows else np.nan,'score_rows':rows,'errors':errors}
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=run(a.model,a.manifest,a.root);Path(a.output).write_text(json.dumps(out,indent=2));print(a.model,out['n_scores'],out['n_samples'],out['n_pairs'],out['tempo'],out['velocity'],'errors',len(out['errors']))
if __name__=='__main__':main()
