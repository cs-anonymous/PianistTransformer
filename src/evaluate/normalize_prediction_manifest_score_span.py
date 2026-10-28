"""Create score-span-normalized copies of MIDI predictions in a manifest."""

import argparse
import json
from pathlib import Path

from miditoolkit import MidiFile, TempoChange

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays


def onset_span_ms(path: Path) -> float:
    arrays = extract_note_arrays(path)
    return float(arrays["ioi"][1:].sum())


def normalize_midi(source: Path, destination: Path, factor: float) -> None:
    midi = MidiFile(str(source))
    if midi.tempo_changes:
        for tempo in midi.tempo_changes:
            tempo.tempo /= factor
    else:
        midi.tempo_changes = [TempoChange(120.0 / factor, 0)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(destination))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    for item in payload["items"]:
        score_path = Path(item["score_midi"])
        target_span = onset_span_ms(score_path)
        normalized_paths = []
        for index, prediction in enumerate(item["prediction_paths"]):
            source = Path(prediction)
            pred_span = onset_span_ms(source)
            if target_span <= 0 or pred_span <= 0:
                raise ValueError(f"Invalid onset span for {source}: score={target_span}, pred={pred_span}")
            factor = target_span / pred_span
            destination = output_root / "midis" / source.parent.name / source.name
            normalize_midi(source, destination, factor)
            normalized_paths.append(str(destination))
        item["original_prediction_paths"] = item["prediction_paths"]
        item["prediction_paths"] = normalized_paths
        item["prediction_time_normalization"] = "score_onset_span"
    payload["prediction_time_normalization"] = "score_onset_span"
    payload["normalization_source_manifest"] = str(args.manifest.resolve())
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
