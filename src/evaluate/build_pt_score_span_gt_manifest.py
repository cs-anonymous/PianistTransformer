import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PT cheap15 prediction manifests to score-span GT manifests."
    )
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-gt-dir", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=ROOT_DIR / "data" / "ASAP_processed")
    parser.add_argument("--refined-root", type=Path, default=ROOT_DIR / "data" / "ASAP_processed")
    parser.add_argument(
        "--eval-gt-time-normalization",
        default="score_onset_span",
        choices=["score_onset_span", "none"],
    )
    return parser.parse_args()


def score_onset_span_ms(score_raw):
    return float(sum(float(row[0]) for row in score_raw[1:]))


def refined_rel_path(path, refined_root):
    resolved = Path(path).resolve()
    return resolved.relative_to(refined_root).as_posix()


def performance_by_source(work, performance_source):
    for perf in work.get("performances", []):
        if perf.get("performance_source") == performance_source:
            return perf
    raise ValueError(f"Performance source not found: {performance_source}")


def performance_shared_rows(perf):
    shared_rows = perf.get("label_shared_raw")
    if shared_rows is not None:
        return [list(row[:3]) for row in shared_rows]
    label_raw = perf.get("label_raw")
    if label_raw is not None:
        return [list(row[:3]) for row in label_raw]
    raise ValueError(f"Performance has no shared raw timing rows: {perf.get('performance_source')}")


def performance_pedal4_rows(perf):
    pedal_rows = perf.get("label_pedal4_raw")
    if pedal_rows is None:
        pedal_rows = perf.get("pedal4_raw")
    if pedal_rows is not None:
        return [list(row[:4]) for row in pedal_rows]
    label_raw = perf.get("label_raw")
    if label_raw is not None:
        return [list(row[3:7]) for row in label_raw]
    raise ValueError(f"Performance has no pedal4 raw rows: {perf.get('performance_source')}")


def raw7_rows_for_eval_gt(perf, score_raw, normalization="none"):
    shared_rows = performance_shared_rows(perf)
    pedal_rows = performance_pedal4_rows(perf)
    if len(shared_rows) != len(score_raw) or len(pedal_rows) != len(score_raw):
        raise ValueError(
            "GT row length mismatch for "
            f"{perf.get('performance_source')}: shared={len(shared_rows)}, "
            f"pedal={len(pedal_rows)}, score={len(score_raw)}"
        )

    mode = str(normalization or "none").lower()
    scale = 1.0
    if mode == "score_onset_span":
        score_span = score_onset_span_ms(score_raw)
        perf_span = float(sum(float(row[0]) for row in shared_rows[1:]))
        if score_span <= 0.0 or perf_span <= 0.0:
            raise ValueError(
                f"Invalid GT score-span normalization for {perf.get('performance_source')}: "
                f"score_span={score_span}, perf_span={perf_span}"
            )
        scale = score_span / perf_span
    elif mode not in {"none", "off", "false", "0"}:
        raise ValueError(f"Unsupported eval_gt_time_normalization={normalization}")

    rows = []
    for shared, pedal in zip(shared_rows, pedal_rows):
        rows.append(
            [
                float(shared[0]) * scale,
                float(shared[1]) * scale,
                float(shared[2]),
                *[float(value) for value in pedal[:4]],
            ]
        )
    return rows, scale


def build_gt_midis(item, processed_root, refined_root, output_gt_dir, normalization):
    score_source = item["score_source"]
    work_path = processed_root / Path(score_source).with_suffix(".json")
    if not work_path.exists():
        raise FileNotFoundError(f"Missing processed work JSON: {work_path}")
    work = json.loads(work_path.read_text(encoding="utf-8"))
    score_raw = work["score"]["score_raw"]
    pitch = list(work["score"]["pitch"])
    gt_paths = item["ground_truth_paths"]
    new_gt_paths = []
    gt_dir = output_gt_dir / "ground_truth_midis"
    gt_dir.mkdir(parents=True, exist_ok=True)
    score_stem = Path(score_source).with_suffix("").as_posix().replace("/", "__")
    original_gt_paths = item.get("original_ground_truth_paths") or gt_paths

    for gt_idx, gt_path in enumerate(gt_paths):
        rel_gt = refined_rel_path(gt_path, refined_root)
        perf = performance_by_source(work, rel_gt)
        raw7_rows, _ = raw7_rows_for_eval_gt(perf, score_raw, normalization=normalization)
        midi_obj = note_features_to_midi(
            pitch=pitch,
            continuous=raw7_rows,
            target_ticks_per_beat=500,
            target_tempo=120,
            max_time_ms=10000.0,
            normalized=False,
        )
        out_path = gt_dir / f"{score_stem}__gt_{gt_idx:03d}.mid"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj.dump(str(out_path))
        new_gt_paths.append(str(out_path.resolve()))

    return new_gt_paths, [str(Path(path).resolve()) for path in original_gt_paths]


def main():
    args = parse_args()
    payload = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    out = dict(payload)
    out["eval_gt_time_normalization"] = args.eval_gt_time_normalization
    out["processed_root"] = str(args.processed_root.resolve())
    out["refined_root"] = str(args.refined_root.resolve())

    new_items = []
    for item in payload["items"]:
        new_gt_paths, original_gt_paths = build_gt_midis(
            item,
            processed_root=args.processed_root,
            refined_root=args.refined_root,
            output_gt_dir=args.output_gt_dir,
            normalization=args.eval_gt_time_normalization,
        )
        copied = dict(item)
        copied["original_ground_truth_paths"] = original_gt_paths
        copied["ground_truth_paths"] = new_gt_paths
        copied["eval_gt_time_normalization"] = args.eval_gt_time_normalization
        new_items.append(copied)

    out["items"] = new_items
    out["num_items"] = len(new_items)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_manifest": str(args.output_manifest.resolve()),
        "output_gt_dir": str(args.output_gt_dir.resolve()),
        "items": len(new_items),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
