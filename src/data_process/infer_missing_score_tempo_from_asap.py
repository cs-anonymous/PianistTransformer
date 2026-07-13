#!/usr/bin/env python
import argparse
import glob
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

import pandas as pd
import torch
from miditoolkit import MidiFile

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.audit_musicxml_score_midi_tempo import audit_xml


def backup_file(path, root, backup_root, kind):
    destination = backup_root / kind / path.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("../PianoCoRe/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("../PianoCoRe/raw"))
    parser.add_argument("--refined-root", type=Path, default=Path("../PianoCoRe/refined"))
    parser.add_argument("--sidecar-tag", default="ASAP_SCORESPAN")
    parser.add_argument("--max-scale-deviation", type=float, default=0.2)
    parser.add_argument("--default-bpm", type=float, default=120.0)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    candidates = []
    pattern = str(args.processed_root / "**" / f"*.{args.sidecar_tag}.pt")
    for sidecar_path_text in glob.glob(pattern, recursive=True):
        sidecar_path = Path(sidecar_path_text)
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        scales = [
            float(perf["global_timing_scale"]["scale"])
            for perf in payload.get("performances", [])
            if "global_timing_scale" in perf
        ]
        if not scales:
            continue
        median_scale = statistics.median(scales)
        if abs(median_scale - 1.0) <= args.max_scale_deviation:
            continue
        score_rel = str(sidecar_path.relative_to(args.processed_root))
        score_rel = score_rel[: -len(f".{args.sidecar_tag}.pt")] + ".mid"
        score_midi = MidiFile(str(args.refined_root / score_rel))
        tempos = score_midi.tempo_changes
        if len(tempos) != 1 or int(tempos[0].time) != 0:
            continue
        old_bpm = float(tempos[0].tempo)
        if not math.isclose(old_bpm, args.default_bpm, rel_tol=1e-4, abs_tol=1e-3):
            continue
        score_rows = metadata[metadata["refined_score_midi_path"] == score_rel]
        xml_paths = score_rows["score_xml_path"].dropna().astype(str).unique()
        if len(xml_paths) != 1:
            continue
        _, xml_sound_tempos = audit_xml(args.raw_root / xml_paths[0], rel_tol=1e-4, abs_tol=1e-3)
        if xml_sound_tempos:
            continue
        inferred_bpm = max(1, int(round(old_bpm * median_scale)))
        candidates.append({
            "score_rel": score_rel,
            "median_scale": median_scale,
            "performance_count": len(scales),
            "old_bpm": old_bpm,
            "inferred_bpm": inferred_bpm,
        })

    report = {"candidates": candidates, "midi_changes": []}
    for candidate in candidates:
        rows = metadata[metadata["refined_score_midi_path"] == candidate["score_rel"]]
        xml_paths = set(rows["score_xml_path"].dropna().astype(str))
        related = metadata[metadata["score_xml_path"].isin(xml_paths)]
        for kind, column, root in (
            ("raw", "score_midi_path", args.raw_root),
            ("refined", "refined_score_midi_path", args.refined_root),
        ):
            for midi_rel in sorted(related[column].dropna().astype(str).unique()):
                path = root / midi_rel
                midi = MidiFile(str(path))
                if len(midi.tempo_changes) != 1 or int(midi.tempo_changes[0].time) != 0:
                    raise ValueError(f"Expected one initial tempo event in {path}")
                old_bpm = float(midi.tempo_changes[0].tempo)
                if not math.isclose(old_bpm, args.default_bpm, rel_tol=1e-4, abs_tol=1e-3):
                    raise ValueError(f"Expected default {args.default_bpm} BPM in {path}, got {old_bpm}")
                backup_file(path, root, args.backup_root, kind)
                midi.tempo_changes[0].tempo = float(candidate["inferred_bpm"])
                midi.dump(str(path))
                report["midi_changes"].append({
                    "path": str(path),
                    "old_bpm": old_bpm,
                    "new_bpm": candidate["inferred_bpm"],
                    "inferred_from_score_rel": candidate["score_rel"],
                })

    report_path = args.backup_root / "inferred_tempo_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "scores": len(candidates),
        "midi_files": len(report["midi_changes"]),
        "report": str(report_path),
        "candidates": candidates,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
