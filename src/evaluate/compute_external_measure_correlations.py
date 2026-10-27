#!/usr/bin/env python3
"""Measure-level tempo/velocity correlations for externally rendered MIDI.

External renderers may insert/delete notes, so this first runs the repository's
MIDIToMIDIAlign tool and computes curves only on score notes with a reliable
matched performance note.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[2]
ALIGN_ROOT = ROOT / "tools" / "AlignmentTool"
import sys
sys.path.insert(0, str(ROOT))
from src.evaluate.compute_strict_curve_correlations import score_info, human_rows, measure_curve, corr
from src.utils.midi import read_corresp

def notes(path):
    m = MidiFile(str(path))
    ns = sorted([n for i in m.instruments if not i.is_drum for n in i.notes],
                key=lambda n: (n.start, n.pitch, n.end, n.velocity))
    return m, ns

def align_pairs(score_path, perf_path):
    work = Path(tempfile.mkdtemp(prefix="ext_align_", dir=str(ALIGN_ROOT)))
    try:
        (work / "Programs").symlink_to(ALIGN_ROOT / "Programs", target_is_directory=True)
        (work / "MIDIToMIDIAlign.sh").symlink_to(ALIGN_ROOT / "MIDIToMIDIAlign.sh")
        (work / "ref.mid").symlink_to(Path(score_path).resolve())
        (work / "perf.mid").symlink_to(Path(perf_path).resolve())
        p = subprocess.run(["bash", "MIDIToMIDIAlign.sh", "ref", "perf"], cwd=work,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=180, check=False, text=True)
        cp = work / "perf_corresp.txt"
        if p.returncode != 0 or not cp.exists():
            return None, f"align_failed rc={p.returncode}: {p.stderr[-300:]}"
        # With ref=score and perf=generated, read_corresp returns (perf_idx, score_idx).
        return read_corresp(cp), None
    except Exception as e:
        return None, repr(e)
    finally:
        shutil.rmtree(work, ignore_errors=True)

def matched_measure_curve(score_m, score, perf_m, perf, pairs):
    if not pairs:
        return None
    # read_corresp returns (first-file index, second-file index) here: perf, score.
    valid = [(pi, si) for pi, si in pairs if 0 <= pi < len(perf) and 0 <= si < len(score)]
    if len(valid) < 10:
        return None
    by_score = {}
    for pi, si in valid:
        by_score.setdefault(si, []).append(pi)
    sis = np.array(sorted(by_score), int)
    # A score onset may contain a chord; use the median matched performance onset.
    sq = np.array([score[i].start / score_m.ticks_per_beat for i in sis], float)
    pt = np.array([np.median([perf[j].start / perf_m.ticks_per_beat for j in by_score[i]]) for i in sis], float)
    vv = np.array([np.mean([perf[j].velocity for j in by_score[i]]) for i in sis], float)
    uq, inv = np.unique(sq, return_inverse=True)
    ut = np.array([np.median(pt[inv == k]) for k in range(len(uq))])
    uv = np.array([np.mean(vv[inv == k]) for k in range(len(uq))])
    tempo = np.diff(uq) / np.maximum(np.diff(ut), 1e-9)
    tq = uq[1:]
    # Score measure boundaries from the score MIDI.
    ts = sorted(score_m.time_signature_changes, key=lambda x: x.time)
    maxq = float(np.max(uq)) + 1.0
    bounds = []
    changes = [(x.time / score_m.ticks_per_beat, int(x.numerator), int(x.denominator)) for x in ts]
    if not changes: changes = [(0.0, 4, 4)]
    for j, (st, num, den) in enumerate(changes):
        en = changes[j + 1][0] if j + 1 < len(changes) else maxq + 10
        L = num * 4.0 / den; cur = st
        while cur < en + 1e-9:
            bounds.append((cur, min(cur + L, en))); cur += L
    starts = np.array([x[0] for x in bounds]); ends = np.array([x[1] for x in bounds])
    ids = np.arange(len(bounds));
    def mid(x):
        p = np.clip(np.searchsorted(starts, x + 1e-8, side='right') - 1, 0, len(bounds)-1)
        bad = x >= ends[p] - 1e-8; p[bad] = np.minimum(p[bad] + 1, len(bounds)-1)
        return ids[p]
    mt, mv = mid(tq), mid(uq)
    common = sorted(set(mt.tolist()) & set(mv.tolist()))
    # Tempo is defined on inter-onset intervals, so average unique-onset
    # periods.  Velocity must use the same note-weighted aggregation as the
    # internal evaluator: average all matched note velocities in a measure,
    # rather than averaging onset means (which overweights sparse chords).
    score_note_measure = mid(sq)
    return (
        np.array(common),
        np.array([np.mean(tempo[mt == k]) for k in common]),
        np.array([np.mean(vv[score_note_measure == k]) for k in common]),
    )

def external_curve_ref(source):
    work, q, score_notes, score_m, ts = score_info(source)
    hs = [measure_curve(q, rows, score_m, ts) for rows in human_rows(work)]
    common = set(hs[0][0].tolist())
    for h in hs[1:]: common &= set(h[0].tolist())
    common = sorted(common)
    H = []
    for h in hs:
        d = {int(k): (t, v) for k, t, v in zip(*h)}; H.append([d[k] for k in common])
    return score_m, score_notes, common, np.nanmedian(np.asarray(H), axis=0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifests", nargs='+', required=True)
    ap.add_argument("--model-roots", nargs='+', required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--correlation", choices=['pearson','spearman'], default='pearson')
    a = ap.parse_args(); outputs=[]
    for manifest, model_root in zip(a.manifests, a.model_roots):
        d = json.loads(Path(manifest).read_text()); rows=[]; errors=[]
        for item in d['items']:
            source=item['score_source']; score_path=ROOT/'data/ASAP_processed'/source
            sm,sn,common,ref=external_curve_ref(source); vals=[]
            for pp in item.get('prediction_paths',[]):
                pred=Path(pp); pred = Path(model_root)/'midis'/pred.name
                if not pred.exists(): errors.append({'score':source,'path':str(pred),'error':'missing'}); continue
                pm,pn=notes(pred); pairs,err=align_pairs(score_path,pred)
                if pairs is None: errors.append({'score':source,'path':str(pred),'error':err}); continue
                c=matched_measure_curve(sm,sn,pm,pn,pairs)
                if c is None: errors.append({'score':source,'path':str(pred),'error':'too_few_matches'}); continue
                dd={int(k):(t,v) for k,t,v in zip(*c)}; keys=[k for k in common if k in dd]
                if len(keys)<3: continue
                A=np.array([dd[k] for k in keys]); R=np.array([ref[common.index(k)] for k in keys])
                vals.append((corr(A[:,0],R[:,0],a.correlation),corr(A[:,1],R[:,1],a.correlation)))
            if vals:
                z=np.nanmean(vals,axis=0); rows.append({'score_source':source,'n_samples':len(vals),'tempo':float(z[0]),'velocity':float(z[1])})
        outputs.append({'manifest':manifest,'model_root':model_root,'n_scores':len(rows),'n_samples':sum(x['n_samples'] for x in rows),'tempo':float(np.nanmean([x['tempo'] for x in rows])) if rows else np.nan,'velocity':float(np.nanmean([x['velocity'] for x in rows])) if rows else np.nan,'score_rows':rows,'errors':errors})
    Path(a.output).write_text(json.dumps({'correlation':a.correlation,'results':outputs},indent=2))
    for x in outputs: print(x['model_root'],x['n_scores'],x['n_samples'],x['tempo'],x['velocity'],'errors',len(x['errors']))
if __name__=='__main__': main()
