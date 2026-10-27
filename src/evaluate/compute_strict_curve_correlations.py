#!/usr/bin/env python
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from miditoolkit import MidiFile
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays

def corr(a,b,kind):
 a=np.asarray(a,float);b=np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
 if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return np.nan
 return float((pearsonr if kind=='pearson' else spearmanr)(a,b).statistic)
def resolve(path,root):
 if path is None:return Path('__missing__')
 p=Path(path)
 if p.exists():return p
 for q in [root/p.parent.name/p.name,root/p.name,root/'midis'/p.name,root/'raw_outputs'/p.name]:
  if q.exists():return q
 return p
def score_info(source):
 rel=Path(source); work=ROOT/'data/ASAP_processed'/rel.with_suffix('.json'); midi=ROOT/'data/ASAP_processed'/rel
 d=json.loads(work.read_text());m=MidiFile(str(midi));notes=sorted([n for i in m.instruments if not i.is_drum for n in i.notes],key=lambda n:(n.start,n.pitch,n.end,n.velocity));
 if len(notes)!=len(d['score']['score_raw']):raise ValueError(f'mismatch {source}')
 q=np.asarray([n.start/m.ticks_per_beat for n in notes],float);ts=sorted(m.time_signature_changes,key=lambda x:x.time)
 return d,q,notes,m,ts
def unique_curve(q,rows,mode='beat'):
 rows=np.asarray(rows,float);n=min(len(q),len(rows));q=q[:n];rows=rows[:n]; perf=np.cumsum(np.maximum(rows[:,0],0.0));
 uq,inv=np.unique(q,return_inverse=True);ut=np.zeros(len(uq));uv=np.zeros(len(uq));
 for j in range(len(uq)):
  sel=inv==j;ut[j]=np.median(perf[sel]);uv[j]=np.mean(rows[sel,2])
 if len(uq)<3:return np.array([]),np.array([]),np.array([])
 lo=int(np.ceil(uq[0])); hi=int(np.floor(uq[-1])); beats=np.arange(lo,hi+1,dtype=float)
 if len(beats)<4:return np.array([]),np.array([]),np.array([])
 bt=np.interp(beats,uq,ut); period=np.diff(bt);tempo=1.0/np.maximum(period,1e-9)
 # velocity is assigned to the beat containing each score onset; empty beats omitted later
 bv=np.full(len(beats),np.nan)
 if len(beats):
  bidx=np.floor(q-beats[0]+1e-9).astype(int)
  valid=(bidx>=0)&(bidx<len(beats))&np.isfinite(rows[:,2])
  sums=np.bincount(bidx[valid],weights=rows[valid,2],minlength=len(beats))
  counts=np.bincount(bidx[valid],minlength=len(beats))
  occupied=counts>0;bv[occupied]=sums[occupied]/counts[occupied]
 return beats[:-1],tempo,bv[:-1]
def measure_ids(q,m,ts):
 maxq=float(np.max(q))+1
 changes=[]
 for x in ts:
  changes.append((x.time/m.ticks_per_beat,int(x.numerator),int(x.denominator)))
 if not changes:changes=[(0.0,4,4)]
 changes=sorted(changes); bounds=[]; idx=0
 for j,(start,num,den) in enumerate(changes):
  end=changes[j+1][0] if j+1<len(changes) else maxq+10
  length=num*4.0/den; cur=start
  while cur<end+1e-9:
   bounds.append((cur,min(cur+length,end),idx));idx+=1;cur+=length
 # Bounds are ordered and non-overlapping; binary search avoids scanning all
 # measures for every note (which becomes costly for long scores).
 starts=np.asarray([z[0] for z in bounds],float)
 ends=np.asarray([z[1] for z in bounds],float)
 ids=np.asarray([z[2] for z in bounds],int)
 pos=np.searchsorted(starts,np.asarray(q,float)+1e-8,side='right')-1
 pos=np.clip(pos,0,len(bounds)-1)
 # A time-signature change can truncate a measure; assign the boundary onset
 # to the following measure when it falls outside the selected interval.
 bad=np.asarray(q,float)>=ends[pos]-1e-8
 pos[bad]=np.minimum(pos[bad]+1,len(bounds)-1)
 return ids[pos]
def measure_curve(q,rows,m,ts):
 rows=np.asarray(rows,float);n=min(len(q),len(rows));q=q[:n];rows=rows[:n];mid=measure_ids(q,m,ts);beat,tempo,bv=unique_curve(q,rows)
 # assign beat-level tempo to the measure containing its beat start; use note measure ids via q
 mids=measure_ids(beat,m,ts) if len(beat) else np.array([],int)
 ids=sorted(set(mid.tolist()) & set(mids.tolist()));T=[];V=[]
 for k in ids:
  a=tempo[mids==k]; vv=rows[mid==k,2];
  if len(a)>=1 and len(vv)>=1:T.append(np.mean(a));V.append(np.mean(vv))
 return np.asarray(ids),np.asarray(T),np.asarray(V)
def human_rows(work):return [p['label_shared_raw'] for p in work.get('performances',[]) if p.get('split')=='test' and p.get('performance_dataset')=='ASAP' and p.get('label_shared_raw') is not None]
def run_item(item,root,mode,kind):
 work,q,notes,m,ts=score_info(item['score_source']);hs=human_rows(work)
 def get(rows):return unique_curve(q,rows) if mode=='beat_period' else measure_curve(q,rows,m,ts)
 hc=[get(x) for x in hs];
 if not hc:return None
 common=set(hc[0][0].tolist())
 for x in hc[1:]:common &= set(x[0].tolist())
 common=sorted(common)
 if len(common)<3:return None
 def align(x):
  d={int(k): (t,v) for k,t,v in zip(x[0],x[1],x[2])};return np.asarray([d[k] for k in common])
 H=np.asarray([align(x) for x in hc]);ref=np.nanmedian(H,axis=0)
 # only scores with >=2 human contribute ceiling
 ht=[];hv=[]
 for i in range(len(H)):
  if len(H)>1:
   O=np.delete(H,i,0);ht.append(corr(H[i,:,0],np.nanmedian(O,axis=0)[:,0],kind));hv.append(corr(H[i,:,1],np.nanmedian(O,axis=0)[:,1],kind))
 mt=[];mv=[];rps=item.get('raw_output_paths',[]);pps=item.get('prediction_paths',[])
 for j in range(max(len(rps),len(pps))):
  rows=None;p=resolve(rps[j] if j<len(rps) else None,root)
  if p.exists(): rows=json.loads(p.read_text()).get('reconstructed_raw7')
  if rows is None and j<len(pps):
   p=resolve(pps[j],root)
   if p.exists():
    a=extract_note_arrays(p);rows=np.column_stack([a['ioi'],a['duration'],a['velocity']]).tolist()
  if rows is None:continue
  x=get(rows);keys=[int(z) for z in x[0] if int(z) in set(common)]
  if len(keys)<3:continue
  d={int(k):(t,v) for k,t,v in zip(x[0],x[1],x[2])};A=np.asarray([d[k] for k in keys]);R=np.asarray([ref[common.index(k)] for k in keys]);
  mt.append(corr(A[:,0],R[:,0],kind));mv.append(corr(A[:,1],R[:,1],kind))
 return {'score_source':item['score_source'],'n_human':len(H),'n_samples':len(mt),'n_bins':len(common),'tempo':float(np.nanmean(mt)) if mt else np.nan,'velocity':float(np.nanmean(mv)) if mv else np.nan,'human_tempo':float(np.nanmean(ht)) if ht else np.nan,'human_velocity':float(np.nanmean(hv)) if hv else np.nan}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('manifests',nargs='+');ap.add_argument('--output',required=True);ap.add_argument('--mode',choices=['beat_period','measure'],required=True);ap.add_argument('--correlation',choices=['pearson','spearman'],default='pearson');a=ap.parse_args();out=[]
 for mp in a.manifests:
  p=Path(mp);d=json.loads(p.read_text());rows=[x for i in d.get('items',[]) if (x:=run_item(i,p.parent,a.mode,a.correlation))]
  mean=lambda k:float(np.nanmean([x[k] for x in rows])) if rows else np.nan
  out.append({'manifest':str(p),'n_scores':len(rows),'n_samples':sum(x['n_samples'] for x in rows),'tempo':mean('tempo'),'velocity':mean('velocity'),'human_tempo':mean('human_tempo'),'human_velocity':mean('human_velocity'),'score_rows':rows})
 Path(a.output).write_text(json.dumps({'mode':a.mode,'correlation':a.correlation,'results':out},indent=2))
 for x in out:print(Path(x['manifest']).parent,x['n_scores'],x['n_samples'],f"tempo={x['tempo']:.4f}",f"velocity={x['velocity']:.4f}",f"human=({x['human_tempo']:.4f},{x['human_velocity']:.4f})")
if __name__=='__main__':main()
