#!/usr/bin/env python
"""Score-aligned beat-level tempo/velocity correlations.

The input is one or more INR prediction manifests.  Paths in archived
manifests are allowed to be stale: raw outputs are resolved relative to the
manifest's sampling directory when the recorded absolute path is absent.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays

def _finite_corr(a, b, method="pearson"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float((pearsonr if method == "pearson" else spearmanr)(a, b).statistic)

def _resolve(path, sampling_dir):
    if path is None:
        return Path("__missing__")
    p = Path(path)
    if p.exists(): return p
    q = sampling_dir / p.parent.name / p.name  # raw_outputs or midis
    if q.exists(): return q
    q = sampling_dir / p.name
    if q.exists(): return q
    return p

def _score_info(source):
    rel = Path(source)
    work = ROOT / "data" / "ASAP_processed" / rel.with_suffix(".json")
    midi = ROOT / "data" / "ASAP_processed" / rel
    d = json.loads(work.read_text())
    sm = MidiFile(str(midi))
    notes = sorted([n for ins in sm.instruments if not ins.is_drum for n in ins.notes],
                   key=lambda n: (n.start, n.pitch, n.end, n.velocity))
    score_raw = np.asarray(d["score"]["score_raw"], float)
    if len(notes) != len(score_raw):
        raise ValueError(f"score note mismatch for {source}: MIDI={len(notes)} JSON={len(score_raw)}")
    onset = np.asarray([n.start / float(sm.ticks_per_beat) for n in notes])
    return d, onset, score_raw

def _curve(onset, score_raw, rows, bin_beats=1.0, scale=1.0):
    rows = np.asarray(rows, float)
    n = min(len(onset), len(score_raw), len(rows))
    onset, score_raw, rows = onset[:n], score_raw[:n], rows[:n]
    bins = np.floor(onset / bin_beats).astype(int)
    tempo, vel = {}, {}
    # score_raw[0] is the score-relative IOI for the first note; the first
    # note has no preceding onset and is excluded from local-tempo estimates.
    prev = np.r_[onset[0], onset[:-1]]
    score_delta = onset - prev
    perf_ioi = rows[:, 0] * scale
    for i, b in enumerate(bins):
        if i > 0 and score_delta[i] > 1e-8 and perf_ioi[i] > 1e-8:
            tempo.setdefault(b, []).append(score_delta[i] / perf_ioi[i])
        if np.isfinite(rows[i, 2]):
            vel.setdefault(b, []).append(rows[i, 2])
    keys = sorted(set(tempo) & set(vel))
    return np.asarray(keys), np.asarray([np.mean(tempo[k]) for k in keys]), np.asarray([np.mean(vel[k]) for k in keys])

def _human_rows(work):
    return [p["label_shared_raw"] for p in work.get("performances", [])
            if p.get("split") == "test" and p.get("performance_dataset") == "ASAP"
            and p.get("label_shared_raw") is not None]

def score_metrics(item, sampling_dir, bin_beats, method):
    work, onset, score_raw = _score_info(item["score_source"])
    humans = _human_rows(work)
    if not humans: return None
    # The existing evaluator rescales human IOIs to the score onset span. This
    # is a per-performance global factor and is retained for consistency.
    human_curves = []
    score_span = float(np.sum(score_raw[1:, 0]))
    for rows in humans:
        perf_span = float(np.sum(np.asarray(rows, float)[1:, 0]))
        sc = score_span / perf_span if perf_span > 0 else 1.0
        human_curves.append(_curve(onset, score_raw, rows, bin_beats, sc))
    # Intersect the beat bins across all human performances, then use the
    # pointwise human median as the reference curve.
    common = set(human_curves[0][0].tolist())
    for c in human_curves[1:]: common &= set(c[0].tolist())
    common = sorted(common)
    if len(common) < 3: return None
    def align(c):
        m = dict(zip(c[0].tolist(), zip(c[1].tolist(), c[2].tolist())))
        return np.asarray([m[k] for k in common])
    H = np.asarray([align(c) for c in human_curves])
    href = np.median(H, axis=0)
    # Human ceiling: each human against leave-one-out median.
    htempo, hvel = [], []
    for i in range(len(H)):
        other = np.delete(H, i, axis=0)
        if len(other):
            htempo.append(_finite_corr(H[i,:,0], np.median(other, axis=0)[:,0], method))
            hvel.append(_finite_corr(H[i,:,1], np.median(other, axis=0)[:,1], method))
    mt, mv = [], []
    raw_paths = item.get("raw_output_paths", [])
    pred_paths = item.get("prediction_paths", [])
    for idx in range(max(len(raw_paths), len(pred_paths))):
        raw_path = raw_paths[idx] if idx < len(raw_paths) else None
        p = _resolve(raw_path, sampling_dir)
        rows = None
        if p.exists():
            payload = json.loads(p.read_text())
            rows = payload.get("reconstructed_raw7")
        if rows is None and idx < len(pred_paths):
            pred = _resolve(pred_paths[idx], sampling_dir)
            if pred.exists():
                a = extract_note_arrays(pred)
                rows = np.column_stack([a["ioi"], a["duration"], a["velocity"]]).tolist()
        if rows is None: continue
        c = _curve(onset, score_raw, rows, bin_beats, 1.0)
        if len(c[0]) < 3: continue
        pred_common = [k for k in common if k in set(c[0].tolist())]
        if len(pred_common) < 3:
            continue
        m = dict(zip(c[0].tolist(), zip(c[1].tolist(), c[2].tolist())))
        A = np.asarray([m[k] for k in pred_common])
        href_local = np.asarray([href[common.index(k)] for k in pred_common])
        mt.append(_finite_corr(A[:,0], href_local[:,0], method))
        mv.append(_finite_corr(A[:,1], href_local[:,1], method))
    return {"score_source": item["score_source"], "n_human": len(H), "n_samples": len(mt),
            "tempo": float(np.nanmean(mt)) if mt else np.nan, "velocity": float(np.nanmean(mv)) if mv else np.nan,
            "human_tempo": float(np.nanmean(htempo)) if htempo else np.nan,
            "human_velocity": float(np.nanmean(hvel)) if hvel else np.nan,
            "n_bins": len(common)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("manifests", nargs="+"); ap.add_argument("--output", required=True); ap.add_argument("--bin-beats", type=float, default=1.0); ap.add_argument("--correlation", choices=["pearson","spearman"], default="pearson")
    a = ap.parse_args(); all_out=[]
    for mp in a.manifests:
        p=Path(mp); d=json.loads(p.read_text()); rows=[]
        for item in d.get("items",[]):
            x=score_metrics(item,p.parent,a.bin_beats,a.correlation)
            if x: rows.append(x)
        def mean(k): return float(np.nanmean([x[k] for x in rows])) if rows else np.nan
        all_out.append({"manifest":str(p),"n_scores":len(rows),"n_samples":sum(x["n_samples"] for x in rows),"tempo":mean("tempo"),"velocity":mean("velocity"),"human_tempo":mean("human_tempo"),"human_velocity":mean("human_velocity"),"score_rows":rows})
    Path(a.output).write_text(json.dumps({"bin_beats":a.bin_beats,"correlation":a.correlation,"results":all_out},indent=2))
    for x in all_out: print(Path(x["manifest"]).parent, x["n_scores"], x["n_samples"], f"tempo={x['tempo']:.4f}", f"velocity={x['velocity']:.4f}", f"human=({x['human_tempo']:.4f},{x['human_velocity']:.4f})")
if __name__ == "__main__": main()
