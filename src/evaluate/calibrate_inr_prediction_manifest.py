import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays
from src.inference.infer_inr_testset import score_midi_dir_from_processed
from src.utils.inr_midi import note_features_to_midi


FEATURES = [
    ("ioi", 0),
    ("duration", 1),
    ("velocity", 2),
    ("pedal_0", 3),
    ("pedal_25", 4),
    ("pedal_50", 5),
    ("pedal_75", 6),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Quantile-calibrate INR prediction raw outputs and re-render MIDIs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-split", type=str, default="train")
    parser.add_argument("--fit-performance-dataset", type=str, default="ASAP")
    parser.add_argument("--max-fit-files", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        help="Comma-separated features to calibrate, or all. Names: ioi,duration,velocity,pedal_0,pedal_25,pedal_50,pedal_75,pedal.",
    )
    return parser.parse_args()


def load_fit_paths(config, split, performance_dataset, max_files, seed):
    metadata = pd.read_csv(
        config["metadata_path"],
        usecols=[
            "tier_a",
            "split",
            "performance_dataset",
            "refined_performance_midi_path",
        ],
    )
    metadata = metadata[metadata["tier_a"].fillna(False).astype(bool)]
    metadata = metadata[metadata["split"] == split]
    metadata = metadata[metadata["performance_dataset"].fillna("").astype(str) == str(performance_dataset)]
    metadata = metadata[metadata["refined_performance_midi_path"].notna()]
    paths = sorted(metadata["refined_performance_midi_path"].unique().tolist())
    if max_files is not None and len(paths) > max_files:
        rng = np.random.default_rng(seed)
        paths = sorted(rng.choice(paths, size=max_files, replace=False).tolist())
    refined_midi_dir = score_midi_dir_from_processed(config["refined_dir"])
    return [str((refined_midi_dir / path).resolve()) for path in paths]


def extract_worker(path):
    arrays = extract_note_arrays(Path(path))
    return {name: arrays[name].astype(np.float64, copy=False) for name, _ in FEATURES}


def collect_target_distributions(paths, num_workers):
    if num_workers and num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            chunks = list(tqdm(pool.imap(extract_worker, paths, chunksize=1), total=len(paths), desc="fit MIDI features"))
    else:
        chunks = [extract_worker(path) for path in tqdm(paths, desc="fit MIDI features")]

    output = {}
    for name, _ in FEATURES:
        values = np.concatenate([chunk[name] for chunk in chunks if len(chunk[name])])
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"No finite target values for {name}")
        output[name] = np.sort(values.astype(np.float64, copy=False))
    return output


def raw_output_records(manifest):
    records = []
    for item_idx, item in enumerate(manifest["items"]):
        for sample_idx, raw_path in enumerate(item.get("raw_output_paths", [])):
            records.append((item_idx, sample_idx, Path(raw_path)))
    return records


def rank_quantile_map(values, target_sorted):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output = values.copy()
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices) == 0:
        return output
    order = finite_indices[np.argsort(values[finite_indices], kind="mergesort")]
    q = (np.arange(len(order), dtype=np.float64) + 0.5) / float(len(order))
    mapped = np.quantile(target_sorted, q, method="linear")
    output[order] = mapped
    return output


def selected_features(feature_arg):
    value = str(feature_arg or "all").strip().lower()
    if value == "all":
        return FEATURES
    requested = {item.strip() for item in value.split(",") if item.strip()}
    if "pedal" in requested:
        requested.update({"pedal_0", "pedal_25", "pedal_50", "pedal_75"})
        requested.remove("pedal")
    by_name = {name: (name, col) for name, col in FEATURES}
    unknown = sorted(requested - set(by_name))
    if unknown:
        raise ValueError(f"Unknown calibration feature: {unknown[0]}")
    return [by_name[name] for name, _ in FEATURES if name in requested]


def calibrate_raw_payloads(records, target_distributions, features):
    payloads = []
    feature_values = {name: [] for name, _ in features}
    for item_idx, sample_idx, raw_path in records:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        raw = np.asarray(payload["reconstructed_raw7"], dtype=np.float64)
        payloads.append((item_idx, sample_idx, raw_path, payload, raw))
        for name, col in features:
            feature_values[name].append(raw[:, col])

    mapped_values = {}
    for name, _ in features:
        values = np.concatenate(feature_values[name])
        mapped_values[name] = rank_quantile_map(values, target_distributions[name])

    offsets = {name: 0 for name, _ in features}
    for _, _, _, payload, raw in payloads:
        n = raw.shape[0]
        for name, col in features:
            start = offsets[name]
            raw[:, col] = mapped_values[name][start : start + n]
            offsets[name] += n
        raw[:, 0:2] = np.clip(raw[:, 0:2], 0.0, 10000.0)
        raw[:, 2:7] = np.clip(raw[:, 2:7], 0.0, 127.0)
        payload["reconstructed_raw7"] = raw.tolist()
        payload["calibration"] = {
            "method": "rank_quantile",
            "target": "fit_split_global",
        }
    return payloads


def safe_stem(score_source):
    return Path(score_source).with_suffix("").as_posix().replace("/", "__")


def render_calibrated_manifest(manifest, payloads, output_dir, config):
    midi_dir = output_dir / "midis"
    raw_dir = output_dir / "raw_outputs"
    midi_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    by_item_sample = {(item_idx, sample_idx): payload for item_idx, sample_idx, _, payload, _ in payloads}
    new_items = []
    for item_idx, item in enumerate(manifest["items"]):
        copied = dict(item)
        prediction_paths = []
        raw_output_paths = []
        score_stem = safe_stem(item["score_source"])
        for sample_idx, _ in enumerate(item.get("raw_output_paths", [])):
            payload = by_item_sample[(item_idx, sample_idx)]
            raw = payload["reconstructed_raw7"]
            midi = note_features_to_midi(
                pitch=payload["pitch"],
                continuous=raw,
                target_ticks_per_beat=500,
                target_tempo=120,
                max_time_ms=float(config.get("max_time_ms", 10000.0)),
                normalized=False,
            )
            pred_path = midi_dir / f"{score_stem}__sample_{sample_idx:03d}.mid"
            raw_path = raw_dir / f"{score_stem}__sample_{sample_idx:03d}.json"
            midi.dump(str(pred_path))
            raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            prediction_paths.append(str(pred_path.resolve()))
            raw_output_paths.append(str(raw_path.resolve()))
        copied["prediction_paths"] = prediction_paths
        copied["raw_output_paths"] = raw_output_paths
        new_items.append(copied)

    output_manifest = dict(manifest)
    output_manifest["items"] = new_items
    output_manifest["calibration"] = {"method": "rank_quantile", "fit_target": "global_train_distribution"}
    manifest_path = output_dir / "prediction_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    records = raw_output_records(manifest)
    if not records:
        raise ValueError("Prediction manifest has no raw_output_paths to calibrate")

    fit_paths = load_fit_paths(
        config,
        split=args.fit_split,
        performance_dataset=args.fit_performance_dataset,
        max_files=args.max_fit_files,
        seed=args.seed,
    )
    target_distributions = collect_target_distributions(fit_paths, args.num_workers)
    features = selected_features(args.features)
    payloads = calibrate_raw_payloads(records, target_distributions, features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = render_calibrated_manifest(manifest, payloads, args.output_dir, config)
    summary = {
        "prediction_manifest": str(args.prediction_manifest.resolve()),
        "output_manifest": str(manifest_path.resolve()),
        "fit_split": args.fit_split,
        "fit_performance_dataset": args.fit_performance_dataset,
        "fit_files": len(fit_paths),
        "features": [name for name, _ in features],
        "num_outputs": len(records),
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
