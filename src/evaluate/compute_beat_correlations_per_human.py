#!/usr/bin/env python3
"""Beat-level Pearson correlations averaged over individual human performers."""
import argparse, json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import pearsonr
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluate.compute_curve_correlations import (
    _score_info, _curve, _human_rows, _resolve, _finite_corr,
)
from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays

def human_curves(work, onset, score_raw, bin_beats):
    out=[]
    score_span=float(np.sum(score_raw[1:,0]))
    for rows in _human_rows(work):
        a=np.asarray(rows,float); span=float(np.sum(a[1:,0]))
        scale=score_span/span if span>0 else 1.0
        out.append(_curve(onset,score_raw,rows,bin_beats,scale))
    return out

def load_prediction(item, root, idx):
    raw_paths=item.get('raw_output_paths',[]); pred_paths=item.get('prediction_paths',[])
    if idx < len(raw_paths):
        p=_resolve(raw_paths[idx],root)
        if p.exists():
            rows=json.loads(p.read_text()).get('reconstructed_raw7')
            if rows is not None:return rows
    if idx < len(pred_paths):
        p=_resolve(pred_paths[idx],root)
        if p.exists():
            a=extract_note_arrays(p)
            return np.column_stack([a['ioi'],a['duration'],a['velocity']]).tolist()
    return None

def align_curve(c, other):
    d={int(k):(t,v) for k,t,v in zip(c[0],c[1],c[2])}
    keys=sorted(set(c[0].tolist()) & set(other[0].tolist()))
    if len(keys)<3:return None
    e={int(k):(t,v) for k,t,v in zip(other[0],other[1],other[2])}
    A=np.asarray([d[k] for k in keys]); B=np.asarray([e[k] for k in keys])
    return A,B

def run_item(item, root, bin_beats):
    work,onset,score_raw=_score_info(item['score_source'])
    hs=human_curves(work,onset,score_raw,bin_beats)
    if not hs:return None
    mt=[];mv=[]
    for j in range(max(len(item.get('raw_output_paths',[])),len(item.get('prediction_paths',[])))):
        rows=load_prediction(item,root,j)
        if rows is None:continue
        pc=_curve(onset,score_raw,rows,bin_beats,1.0)
        pair_t=[];pair_v=[]
        for hc in hs:
            z=align_curve(pc,hc)
            if z is None:continue
            A,B=z
            pair_t.append(_finite_corr(A[:,0],B[:,0],'pearson'))
            pair_v.append(_finite_corr(A[:,1],B[:,1],'pearson'))
        if pair_t:
            mt.append(float(np.nanmean(pair_t)));mv.append(float(np.nanmean(pair_v)))
    # Human ceiling: each performer compared with every other performer.
    ht=[];hv=[]
    for i,a in enumerate(hs):
        for j,b in enumerate(hs):
            if i>=j:continue
            z=align_curve(a,b)
            if z is None:continue
            A,B=z
            ht.append(_finite_corr(A[:,0],B[:,0],'pearson'))
            hv.append(_finite_corr(A[:,1],B[:,1],'pearson'))
    return {'score_source':item['score_source'],'n_human':len(hs),'n_samples':len(mt),
            'n_pairs':len(mt)*len(hs),'tempo':float(np.nanmean(mt)) if mt else np.nan,
            'velocity':float(np.nanmean(mv)) if mv else np.nan,
            'human_tempo':float(np.nanmean(ht)) if ht else np.nan,
            'human_velocity':float(np.nanmean(hv)) if hv else np.nan}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifests',nargs='+');ap.add_argument('--output',required=True);ap.add_argument('--bin-beats',type=float,default=1.0);a=ap.parse_args()
    outs=[]
    for mp in a.manifests:
        p=Path(mp);d=json.loads(p.read_text()); rows=[]
        for item in d.get('items',[]):
            x=run_item(item,p.parent,a.bin_beats)
            if x:rows.append(x)
        mean=lambda k:float(np.nanmean([x[k] for x in rows])) if rows else np.nan
        outs.append({'manifest':str(p),'n_scores':len(rows),'n_samples':sum(x['n_samples'] for x in rows),'n_pairs':sum(x['n_pairs'] for x in rows),'tempo':mean('tempo'),'velocity':mean('velocity'),'human_tempo':mean('human_tempo'),'human_velocity':mean('human_velocity'),'score_rows':rows})
    Path(a.output).write_text(json.dumps({'unit':'beat','human_aggregation':'mean over individual human performers','bin_beats':a.bin_beats,'correlation':'pearson','results':outs},indent=2))
    for x in outs: print(Path(x['manifest']).parent,x['n_scores'],x['n_samples'],x['n_pairs'],f"tempo={x['tempo']:.4f}",f"velocity={x['velocity']:.4f}",f"human=({x['human_tempo']:.4f},{x['human_velocity']:.4f})")
if __name__=='__main__':main()
