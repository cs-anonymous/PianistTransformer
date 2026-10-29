"""Normalize prediction MIDIs to each score's first-to-last onset span."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from miditoolkit import MidiFile, TempoChange

ROOT = Path(__file__).resolve().parents[2]
BACKUP_ROOT = ROOT / "backup" / "repo_cleanup_20260804"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays


def resolve_score(raw: str | Path, score_source: str) -> Path:
    candidates = [Path(raw)]
    raw_text = str(raw)
    marker = "/PianistTransformer/"
    if marker in raw_text:
        relative = Path(raw_text.split(marker, 1)[1])
        candidates.extend([ROOT / relative])
        if relative.parts[:2] == ("PianoCoRe", "refined"):
            candidates.extend([
                ROOT / "data" / "ASAP_processed" / Path(*relative.parts[2:]),
                ROOT / "data" / "ASAP_processed_pedal_group" / Path(*relative.parts[2:]),
            ])
    source = Path(score_source)
    candidates.extend([
        ROOT / "data" / "ASAP_processed" / source,
        ROOT / "data" / "ASAP_processed_pedal_group" / source,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"score MIDI not found for {score_source}: {raw}")


def resolve_prediction(raw: str | Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    raw_text = str(raw)
    marker = "/PianistTransformer/"
    if marker in raw_text:
        relative = Path(raw_text.split(marker, 1)[1])
        for candidate in (ROOT / relative, BACKUP_ROOT / relative):
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"prediction MIDI not found: {raw}")


def onset_span_ms(path: Path) -> float:
    arrays = extract_note_arrays(path)
    return float(arrays["ioi"][1:].sum())


def normalize_midi(source: Path, destination: Path, target_span: float) -> dict:
    source_span = onset_span_ms(source)
    if target_span <= 0.0 or source_span <= 0.0:
        raise ValueError(f"invalid onset span for {source}: source={source_span}, target={target_span}")
    factor = target_span / source_span
    midi = MidiFile(str(source))
    if midi.tempo_changes:
        for tempo in midi.tempo_changes:
            tempo.tempo /= factor
    else:
        midi.tempo_changes = [TempoChange(120.0 / factor, 0)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(destination))
    return {"source_span_ms": source_span, "target_span_ms": target_span, "scale": factor}


def normalize_manifest(name: str, manifest_path: Path, output_root: Path) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for item_index, item in enumerate(payload["items"]):
        score_path = resolve_score(item.get("score_midi", ""), item["score_source"])
        target_span = onset_span_ms(score_path)
        normalized_paths = []
        item_records = []
        for prediction_index, prediction in enumerate(item["prediction_paths"]):
            source = resolve_prediction(prediction)
            destination = output_root / name / f"{item_index:03d}" / f"sample_{prediction_index:03d}.mid"
            stats = normalize_midi(source, destination, target_span)
            normalized_paths.append(str(destination.resolve()))
            item_records.append({"prediction_index": prediction_index, "path": str(destination.resolve()), **stats})
        item["original_prediction_paths"] = item["prediction_paths"]
        item["prediction_paths"] = normalized_paths
        item["prediction_time_normalization"] = "score_onset_span"
        item["prediction_onset_span_records"] = item_records
        records.extend(item_records)
    payload["prediction_time_normalization"] = "score_onset_span"
    payload["normalization_source_manifest"] = str(manifest_path.resolve())
    payload["normalization_protocol"] = {
        "target": "score MIDI first-to-last note onset span",
        "operation": "global tempo scaling per prediction",
    }
    output_manifest = output_root / name / "prediction_manifest.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(name, len(records), "predictions", "->", output_manifest)
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for spec in args.method:
        name, manifest = spec.split("=", 1)
        normalize_manifest(name, Path(manifest), args.output_root)


if __name__ == "__main__":
    main()
