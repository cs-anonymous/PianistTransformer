#!/usr/bin/env python3
"""Create full-piece length-normalized external baseline manifests.

Prediction MIDI files are globally time-scaled to match the corresponding
score MIDI duration. Ground-truth MIDI files are left untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pretty_midi


ROOT = Path(__file__).resolve().parents[1]


def resolve_existing(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path]
    if path.is_absolute():
        text = str(path)
        candidates.append(Path(text.replace("/results/", "/backup/results0722/")))
    else:
        candidates.append(ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def normalize_midi(source: Path, target: Path, target_duration: float) -> dict:
    midi = pretty_midi.PrettyMIDI(str(source))
    source_duration = float(midi.get_end_time())
    if source_duration <= 0.0:
        raise ValueError(f"empty MIDI: {source}")
    scale = float(target_duration) / source_duration
    midi.adjust_times([0.0, source_duration], [0.0, float(target_duration)])
    target.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(target))
    fixed = pretty_midi.PrettyMIDI(str(target))
    return {
        "source_duration_s": source_duration,
        "target_duration_s": float(target_duration),
        "normalized_duration_s": float(fixed.get_end_time()),
        "time_scale": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-sidecars", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_existing(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = args.output_dir
    midi_dir = out_dir / "midis"
    report = []
    new_manifest = deepcopy(manifest)
    new_manifest["source_prediction_manifest"] = str(manifest_path.resolve())
    new_manifest["length_normalization"] = {
        "target": "score_midi_full_duration",
        "method": "global linear time scaling with pretty_midi.adjust_times",
    }
    new_manifest["items"] = []

    for score_idx, item in enumerate(manifest["items"]):
        score_midi = resolve_existing(item["score_midi"])
        target_duration = pretty_midi.PrettyMIDI(str(score_midi)).get_end_time()
        new_item = deepcopy(item)
        new_predictions = []
        new_details = []
        for sample_idx, pred_path in enumerate(item["prediction_paths"]):
            source = resolve_existing(pred_path)
            target = midi_dir / f"{score_idx:02d}__sample_{sample_idx:03d}.mid"
            stats = normalize_midi(source, target, target_duration)
            new_predictions.append(str(target.resolve()))
            detail = {
                "score_source": item["score_source"],
                "sample_id": f"sample_{sample_idx:03d}",
                "source": str(source.resolve()),
                "output": str(target.resolve()),
                **stats,
            }
            report.append(detail)
            new_details.append(detail)
            if args.copy_sidecars:
                png = source.with_suffix(".png")
                if png.exists():
                    shutil.copy2(png, target.with_suffix(".png"))
        new_item["prediction_paths"] = new_predictions
        new_item["length_normalized_predictions"] = new_details
        new_manifest["items"].append(new_item)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prediction_manifest.json").write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "length_normalization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in report:
        print(
            Path(row["output"]).name,
            "scale",
            f"{row['time_scale']:.4f}",
            "before",
            f"{row['source_duration_s']:.3f}",
            "after",
            f"{row['normalized_duration_s']:.3f}",
            "target",
            f"{row['target_duration_s']:.3f}",
        )


if __name__ == "__main__":
    main()
