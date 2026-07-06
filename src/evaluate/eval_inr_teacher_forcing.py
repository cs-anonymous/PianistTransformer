import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.evaluate.evaluate_inr_saved_midis import score_level_metrics, aggregate_score_metrics
from src.inference.infer_inr_testset import (
    build_epr_score_input_rows,
    score_midi_dir_from_processed,
    select_device,
)
from src.model.integrated_pianoformer import _materialize_epr_prediction, _target5_to_raw7
from src.train.train_inr import (
    build_perf_style_prefix_cache,
    build_style_vocabs,
    create_model,
    performance_dev_velocity_pedal4_binary_rows,
    performance_dev_velocity_pedal2_rows,
    perf_style_stats_range_from_cache,
    perf_style_stats_from_cache,
    score_style_stats,
)
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(description="Full-score teacher-forcing eval for INR EPR checkpoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--batch-size-windows", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--materialize-strategy", choices=["mean", "sample"], default="mean")
    parser.add_argument("--save-midi", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_stem(score_source):
    return Path(score_source).with_suffix("").as_posix().replace("/", "__")


def stable_seed(base_seed, *parts):
    import hashlib

    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_config(config_path, checkpoint):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config["resume_path"] = str(checkpoint)
    if config.get("use_style_tokens", False):
        composer_vocab = config.get("style_composer_vocab")
        source_vocab = config.get("style_source_vocab")
        if composer_vocab is None or source_vocab is None:
            composer_vocab, source_vocab = build_style_vocabs(config["metadata_path"])
            config["style_composer_vocab"] = composer_vocab
            config["style_source_vocab"] = source_vocab
        config["style_creator_vocab_size"] = len(config["style_composer_vocab"])
        config["style_source_vocab_size"] = len(config["style_source_vocab"])
    return config


def selected_asap_perfs(work, item):
    by_source = {perf.get("performance_source"): perf for perf in work.get("performances", [])}
    return [
        by_source[source]
        for source in item.get("selected_performance_sources", [])
        if source in by_source
    ]


def labels_for_perf(config, perf, score_shared_raw):
    if str(config.get("pedal_representation", "")).lower() == "binary_4":
        labels = performance_dev_velocity_pedal4_binary_rows(
            perf,
            score_shared_raw,
            epr_timing_target=config.get("epr_timing_target", "deviation"),
            log_scale=float(config.get("timing_log_scale", 50.0)),
            split_zero_ioi_head=bool(config.get("split_zero_ioi_head", False)),
            ioi_nonzero_dev_scale=float(config.get("ioi_nonzero_dev_scale", 2.0)),
            ioi_zero_dev_scale=float(config.get("ioi_zero_dev_scale", 4.0)),
            pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
        )
    else:
        labels = performance_dev_velocity_pedal2_rows(
            perf,
            score_shared_raw,
            epr_timing_target=config.get("epr_timing_target", "deviation"),
            log_scale=float(config.get("timing_log_scale", 50.0)),
            split_zero_ioi_head=bool(config.get("split_zero_ioi_head", False)),
            ioi_nonzero_dev_scale=float(config.get("ioi_nonzero_dev_scale", 2.0)),
            ioi_zero_dev_scale=float(config.get("ioi_zero_dev_scale", 4.0)),
        )
    if labels is None:
        raise ValueError(f"Could not build labels for {perf.get('performance_source')}")
    return labels


def build_windows(total_notes, block_notes, overlap_ratio):
    if total_notes <= block_notes:
        return [(0, total_notes)]
    stride = max(1, int(round(block_notes * (1.0 - overlap_ratio))))
    windows = []
    start = 0
    while start + block_notes <= total_notes:
        windows.append((start, start + block_notes))
        start += stride
    if windows[-1][1] != total_notes:
        windows.append((max(0, total_notes - block_notes), total_notes))
    deduped = []
    seen = set()
    for window in windows:
        if window in seen:
            continue
        deduped.append(window)
        seen.add(window)
    return deduped


def teacher_forcing_predict(
    model,
    config,
    score,
    pitch,
    score_inputs,
    score_shared_raw,
    labels,
    windows,
    device,
    batch_size,
    materialize_strategy,
    style_creator_id=None,
    style_source_id=None,
):
    total_notes = len(pitch)
    pred_sum = None
    pred_count = torch.zeros(total_notes, 1, dtype=torch.float32)
    loss_sum = 0.0
    loss_count = 0
    use_style_tokens = bool(config.get("use_style_tokens", False))
    perf_style_stats_mode = str(config.get("perf_style_stats_mode", "prefix") or "prefix").lower()
    perf_style_cache = build_perf_style_prefix_cache(labels) if use_style_tokens else None

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        pitch_tensors = []
        score_tensors = []
        raw_tensors = []
        label_tensors = []
        style_score_stats = []
        style_perf_stats = []
        style_perf_is_pad = []
        lengths = []
        for start, end in batch_windows:
            pitch_tensors.append(torch.tensor(pitch[start:end], dtype=torch.long))
            score_tensors.append(torch.tensor(score_inputs[start:end], dtype=torch.float32))
            raw_tensors.append(torch.tensor(score_shared_raw[start:end], dtype=torch.float32))
            label_tensors.append(torch.tensor(labels[start:end], dtype=torch.float32))
            if use_style_tokens:
                style_score_stats.append(score_style_stats(score, start, end))
                if perf_style_stats_mode == "window":
                    style_perf_stats.append(perf_style_stats_range_from_cache(perf_style_cache, start, end))
                    style_perf_is_pad.append(False)
                else:
                    style_perf_stats.append(perf_style_stats_from_cache(perf_style_cache, start))
                    style_perf_is_pad.append(bool(start <= 0))
            lengths.append(end - start)

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=config["pitch_pad_id"]).to(device)
        continuous = pad_sequence(score_tensors, batch_first=True, padding_value=0.0).to(device)
        score_raw = pad_sequence(raw_tensors, batch_first=True, padding_value=0.0).to(device)
        label_batch = pad_sequence(label_tensors, batch_first=True, padding_value=0.0).to(device)
        attention_mask = (pitch_ids != config["pitch_pad_id"]).long()
        style_kwargs = {}
        if use_style_tokens:
            style_kwargs = {
                "style_creator_ids": torch.full(
                    (len(batch_windows),),
                    int(style_creator_id or 0),
                    dtype=torch.long,
                    device=device,
                ),
                "style_source_ids": torch.full(
                    (len(batch_windows),),
                    int(style_source_id or 0),
                    dtype=torch.long,
                    device=device,
                ),
                "style_score_stats": torch.tensor(style_score_stats, dtype=torch.float32, device=device),
                "style_perf_stats": torch.tensor(style_perf_stats, dtype=torch.float32, device=device),
                "style_perf_is_pad": torch.tensor(style_perf_is_pad, dtype=torch.bool, device=device),
            }

        with torch.no_grad():
            outputs = model(
                pitch_ids=pitch_ids,
                continuous=continuous,
                score_shared_raw=score_raw,
                labels_continuous=label_batch,
                attention_mask=attention_mask,
                continuous_sampling_strategy=materialize_strategy,
                **style_kwargs,
            )
            pred = _materialize_epr_prediction(
                model.config,
                outputs.logits,
                sampling_strategy=materialize_strategy,
                score_shared_raw=score_raw,
            )
        pred = pred.detach().float().cpu()
        if pred_sum is None:
            pred_sum = torch.zeros(total_notes, pred.shape[-1], dtype=torch.float32)
        if outputs.loss is not None:
            loss_sum += float(outputs.loss.detach().float().cpu()) * len(batch_windows)
            loss_count += len(batch_windows)

        for idx, (start, end) in enumerate(batch_windows):
            length = lengths[idx]
            pred_sum[start:end] += pred[idx, :length]
            pred_count[start:end] += 1.0

    if pred_sum is None:
        raise ValueError("No windows were processed")
    return pred_sum / pred_count.clamp_min(1.0), loss_sum / max(loss_count, 1)


def raw_arrays_from_rows(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return {
        "ioi": rows[:, 0],
        "duration": rows[:, 1],
        "velocity": rows[:, 2],
        "pedal": rows[:, 3:7].reshape(-1),
    }


def finite_wass(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def predict_work(model, device, config, item, args, score_midi_dir):
    work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
    score = work["score"]
    pitch = score["pitch"]
    score_shared_raw = [row[:3] for row in score["score_raw"]]
    score_inputs = build_epr_score_input_rows(
        score,
        use_timing_scale_bit=config.get("use_timing_scale_bit", True),
        timing_control_mode=config.get("timing_control_mode"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
        musical_feature_mode=config.get("musical_feature_mode", "categorical"),
    )
    windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
    perfs = selected_asap_perfs(work, item)
    composer_vocab = config.get("style_composer_vocab", {})
    source_vocab = config.get("style_source_vocab", {})
    meta = work.get("meta", {})
    style_creator_id = int(composer_vocab.get(str(meta.get("composer") or ""), composer_vocab.get("<unk>", 0)))
    score_stem = safe_stem(item["score_source"])
    midi_dir = args.output_dir / "midis"
    raw_dir = args.output_dir / "raw_outputs"
    midi_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    prediction_paths = []
    raw_output_paths = []
    gt_paths = []
    tf_wass_rows = []
    losses = []
    for perf_idx, perf in enumerate(perfs):
        labels = labels_for_perf(config, perf, score_shared_raw)
        seed = stable_seed(args.seed, item["score_source"], perf.get("performance_source"), args.materialize_strategy)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        pred_target, loss = teacher_forcing_predict(
            model,
            config,
            score,
            pitch,
            score_inputs,
            score_shared_raw,
            labels,
            windows,
            device,
            args.batch_size_windows,
            args.materialize_strategy,
            style_creator_id=style_creator_id,
            style_source_id=int(source_vocab.get(str(perf.get("performance_dataset") or ""), source_vocab.get("<unk>", 0))),
        )
        losses.append(loss)
        pred_raw = _target5_to_raw7(
            torch.tensor(score_shared_raw, dtype=torch.float32),
            pred_target.float().cpu(),
            config=config,
        ).cpu().numpy()
        target_raw = _target5_to_raw7(
            torch.tensor(score_shared_raw, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            config=config,
        ).cpu().numpy()

        pred_arrays = raw_arrays_from_rows(pred_raw)
        target_arrays = raw_arrays_from_rows(target_raw)
        tf_wass_rows.append(
            {
                "performance_source": perf.get("performance_source"),
                "loss": float(loss),
                **{
                    f"{key}_wass": finite_wass(pred_arrays[key], target_arrays[key])
                    for key in pred_arrays
                },
            }
        )

        if args.save_midi:
            midi_obj = note_features_to_midi(
                pitch=pitch,
                continuous=pred_raw.tolist(),
                target_ticks_per_beat=500,
                target_tempo=120,
                max_time_ms=float(config.get("max_time_ms", 10000.0)),
                normalized=False,
            )
            pred_path = midi_dir / f"{score_stem}__tf_{perf_idx:03d}.mid"
            midi_obj.dump(str(pred_path))
            prediction_paths.append(str(pred_path.resolve()))

        raw_path = raw_dir / f"{score_stem}__tf_{perf_idx:03d}.json"
        raw_payload = {
            "score_source": item["score_source"],
            "performance_source": perf.get("performance_source"),
            "materialize_strategy": args.materialize_strategy,
            "loss": float(loss),
            "pitch": [int(value) for value in pitch],
            "predicted_target": pred_target.tolist(),
            "target": labels,
            "reconstructed_raw7": pred_raw.tolist(),
            "target_raw7": target_raw.tolist(),
        }
        raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        raw_output_paths.append(str(raw_path.resolve()))
        gt_path = score_midi_dir / perf.get("performance_source")
        if gt_path.exists():
            gt_paths.append(str(gt_path.resolve()))

    return {
        "score_source": item["score_source"],
        "score_midi": str((score_midi_dir / item["score_source"]).resolve()),
        "prediction_paths": prediction_paths,
        "raw_output_paths": raw_output_paths,
        "ground_truth_paths": gt_paths,
        "teacher_forcing_rows": tf_wass_rows,
        "mean_teacher_forcing_loss": float(np.mean(losses)) if losses else float("nan"),
        "note_count": len(pitch),
        "num_windows": len(windows),
    }


def worker_loop(worker_idx, args, config, score_midi_dir, job_queue, result_queue):
    random.seed(args.seed + worker_idx)
    torch.manual_seed(args.seed + worker_idx)
    device = select_device(args.device)
    print(f"Worker {worker_idx} using device: {device}", flush=True)
    model = create_model(config)
    model.to(device)
    model.eval()
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_idx, item = job
        try:
            result_queue.put((job_idx, predict_work(model, device, config, item, args, score_midi_dir), None))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((job_idx, None, repr(exc)))


def run_dynamic_pool(args, config, manifest, score_midi_dir):
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=worker_loop, args=(idx, args, config, score_midi_dir, job_queue, result_queue))
        for idx in range(args.num_workers)
    ]
    for worker in workers:
        worker.start()
    for job_idx, item in enumerate(manifest):
        job_queue.put((job_idx, item))
    for _ in workers:
        job_queue.put(None)

    by_idx = {}
    with tqdm(total=len(manifest), desc="teacher-forcing pool") as progress:
        for _ in range(len(manifest)):
            job_idx, item, error = result_queue.get()
            if error is not None:
                for worker in workers:
                    worker.terminate()
                raise RuntimeError(f"Worker failed on job {job_idx}: {error}")
            by_idx[job_idx] = item
            progress.update(1)
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Worker {worker.pid} exited with code {worker.exitcode}")
    return [by_idx[idx] for idx in range(len(manifest))]


def run_single(args, config, manifest, score_midi_dir):
    device = select_device(args.device)
    print(f"Using device: {device}")
    model = create_model(config)
    model.to(device)
    model.eval()
    return [
        predict_work(model, device, config, item, args, score_midi_dir)
        for item in tqdm(manifest, desc="teacher-forcing")
    ]


def aggregate_tf_rows(items):
    rows = [row for item in items for row in item.get("teacher_forcing_rows", [])]
    keys = ["loss", "ioi_wass", "duration_wass", "velocity_wass", "pedal_wass"]
    output = {"num_rows": int(len(rows))}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        output[key] = float(np.mean(values)) if len(values) else float("nan")
    return output


def evaluate_prediction_manifest(items, manifest_path, output_json):
    payload = {"protocol": "teacher_forcing", "num_samples": 1, "items": items}
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    score_rows = [score_level_metrics(item) for item in tqdm(items, desc="eval tf MIDI")]
    output = {
        "prediction_manifest": str(manifest_path.resolve()),
        "protocol": "teacher_forcing",
        "num_scores": len(score_rows),
        "aggregate": {
            "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
            "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
        },
        "scores": score_rows,
    }
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, args.checkpoint)
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )
    score_midi_dir = score_midi_dir_from_processed(config["refined_dir"])
    if args.num_workers > 1:
        items = run_dynamic_pool(args, config, manifest, score_midi_dir)
    else:
        items = run_single(args, config, manifest, score_midi_dir)

    manifest_path = args.output_dir / "prediction_manifest.json"
    eval_json = args.output_dir / "eval.json"
    eval_output = evaluate_prediction_manifest(items, manifest_path, eval_json) if args.save_midi else None
    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "materialize_strategy": args.materialize_strategy,
        "num_scores": int(len(items)),
        "teacher_forcing_pairwise": aggregate_tf_rows(items),
        "midi_eval": str(eval_json.resolve()) if eval_output is not None else None,
        "midi_pp_wass": eval_output["aggregate"]["pp_wass"] if eval_output is not None else None,
        "midi_pn_wass": eval_output["aggregate"]["pn_wass"] if eval_output is not None else None,
        "items": items,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
