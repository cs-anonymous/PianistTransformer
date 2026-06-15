"""
Evaluate a PianistTransformer checkpoint on the corrected 104-window ASAP test
subset from PianoCoRe node JSONs.

This script reproduces the historical quick-eval split documented in
results/EVALUATION_STATUS.md:
  - first 32 test works
  - max 8 eval performances per work
  - max 4 eval windows per work
  - ASAP selected by performance_dataset == "ASAP"

Metrics are computed from global PT token distributions, matching the PT paper's
objective protocol more closely than EPRMetrics(bins=100):
  - velocity: 128 token bins
  - duration: 5000 token bins
  - IOI: 4991 token bins
  - pedal: 16 binary joint configurations
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from transformers import LogitsProcessorList

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.model.generate import BatchSparseForcedTokenProcessor
from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest


DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "sft_pianocore_from_scratch" / "sft_2026-06-14-14-12-24"
DEFAULT_CONFIG = ROOT_DIR / "configs" / "sft_node_config_pianocore_local.json"


FEATURE_BINS = {
    "velocity": 128,
    "duration": 5000,
    "ioi": 4991,
    "pedal": 16,
}

MS_BINS = 10001


def resolve_model_paths(model_path: Path) -> Tuple[Path, Path]:
    model_path = Path(model_path)
    if model_path.is_dir():
        return model_path / "model.safetensors", model_path / "config.json"
    return model_path, model_path.with_name("config.json")


def load_pt_model_and_config(model_path: Path):
    from safetensors.torch import load_file
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    weight_path, config_path = resolve_model_paths(model_path)
    if not weight_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Expected model.safetensors and config.json under {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = PianoT5GemmaConfig(
        encoder_layers_num=config_dict["encoder"]["num_hidden_layers"],
        decoder_layers_num=config_dict["decoder"]["num_hidden_layers"],
        hidden_size=config_dict["hidden_size"],
        intermediate_size=config_dict["encoder"]["intermediate_size"],
        num_attention_heads=config_dict["encoder"]["num_attention_heads"],
        num_key_value_heads=config_dict["encoder"]["num_key_value_heads"],
        head_dim=config_dict["encoder"]["head_dim"],
    )
    model = PianoT5Gemma(config)
    model.load_state_dict(load_file(str(weight_path)), strict=False)
    model.eval()
    return model, config


def continuous_time_to_tick(value: float, max_time_ms: float) -> int:
    # Match src/data_process/generate_pt_sft_from_node_json_multiprocess.py,
    # which created the PT SFT JSONL used for the from-scratch training run.
    return int(round(float(value) * max_time_ms))


def normalized_time_to_ms(value: float, max_time_ms: float) -> int:
    value = min(max(float(value), 0.0), 1.0)
    return int(round(np.expm1(value * np.log1p(max_time_ms))))


def bad_training_tick_to_ms(tick: int, max_time_ms: float) -> int:
    # The existing from-scratch PT checkpoint was trained on ticks produced by
    # round(log_norm * max_time_ms), then clipped by PT's valid token ranges.
    # Decode those ticks back to the ms implied by that training representation.
    norm = min(max(float(tick) / max_time_ms, 0.0), 1.0)
    return int(round(np.expm1(norm * np.log1p(max_time_ms))))


def continuous_to_pt_tokens(
    pitch_ids: Sequence[int],
    continuous: Sequence[Sequence[float]],
    config,
    max_time_ms: float,
    include_pedal: bool,
) -> List[int]:
    tokens: List[int] = []
    for pitch, row in zip(pitch_ids, continuous):
        pitch_token = int(np.clip(config.pitch_start + int(pitch), *config.valid_id_range[0]))
        pitch_token = min(pitch_token, config.valid_id_range[0][1] - 1)

        ioi = continuous_time_to_tick(row[0], max_time_ms)
        duration = continuous_time_to_tick(row[1], max_time_ms)
        velocity = int(round(float(row[2]) * 127))

        ioi_token = int(np.clip(config.timing_start + ioi, config.valid_id_range[1][0], config.valid_id_range[1][1] - 1))
        vel_token = int(np.clip(config.velocity_start + velocity, config.valid_id_range[2][0], config.valid_id_range[2][1] - 1))
        dur_token = int(np.clip(config.timing_start + duration, config.valid_id_range[3][0], config.valid_id_range[3][1] - 1))

        if include_pedal and len(row) >= 7:
            pedal_values = [int(round(float(v) * 127)) for v in row[3:7]]
        else:
            pedal_values = [0, 0, 0, 0]
        pedal_tokens = [
            int(np.clip(config.pedal_start + value, config.valid_id_range[4 + i][0], config.valid_id_range[4 + i][1] - 1))
            for i, value in enumerate(pedal_values)
        ]
        tokens.extend([pitch_token, ioi_token, vel_token, dur_token] + pedal_tokens)
    return tokens


def token_feature_counts(token_ids: Sequence[int], config) -> Dict[str, np.ndarray]:
    arr = np.asarray(token_ids, dtype=np.int64)
    usable_len = (len(arr) // 8) * 8
    arr = arr[:usable_len].reshape(-1, 8)

    counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    if arr.size == 0:
        return counts

    ioi = np.clip(arr[:, 1] - config.timing_start, 0, FEATURE_BINS["ioi"] - 1)
    duration = np.clip(arr[:, 3] - config.timing_start, 0, FEATURE_BINS["duration"] - 1)
    velocity = np.clip(arr[:, 2] - config.velocity_start, 0, FEATURE_BINS["velocity"] - 1)
    pedals = np.clip(arr[:, 4:8] - config.pedal_start, 0, 127)
    pedal_bits = (pedals > 64).astype(np.int64)
    pedal = pedal_bits[:, 0] * 8 + pedal_bits[:, 1] * 4 + pedal_bits[:, 2] * 2 + pedal_bits[:, 3]

    counts["ioi"] += np.bincount(ioi, minlength=FEATURE_BINS["ioi"])
    counts["duration"] += np.bincount(duration, minlength=FEATURE_BINS["duration"])
    counts["velocity"] += np.bincount(velocity, minlength=FEATURE_BINS["velocity"])
    counts["pedal"] += np.bincount(pedal, minlength=FEATURE_BINS["pedal"])
    return counts


def target_ms_feature_counts(target_rows: Sequence[Sequence[float]], max_time_ms: float) -> Dict[str, np.ndarray]:
    counts = {
        "velocity": np.zeros(FEATURE_BINS["velocity"], dtype=np.int64),
        "duration": np.zeros(MS_BINS, dtype=np.int64),
        "ioi": np.zeros(MS_BINS, dtype=np.int64),
        "pedal": np.zeros(FEATURE_BINS["pedal"], dtype=np.int64),
    }
    if not target_rows:
        return counts

    arr = np.asarray(target_rows, dtype=np.float64)
    ioi_ms = np.clip(
        np.rint(np.expm1(np.clip(arr[:, 0], 0.0, 1.0) * np.log1p(max_time_ms))).astype(np.int64),
        0,
        MS_BINS - 1,
    )
    duration_ms = np.clip(
        np.rint(np.expm1(np.clip(arr[:, 1], 0.0, 1.0) * np.log1p(max_time_ms))).astype(np.int64),
        0,
        MS_BINS - 1,
    )
    velocity = np.clip(np.rint(arr[:, 2] * 127).astype(np.int64), 0, FEATURE_BINS["velocity"] - 1)
    pedals = np.clip(np.rint(arr[:, 3:7] * 127).astype(np.int64), 0, 127)
    pedal_bits = (pedals > 64).astype(np.int64)
    pedal = pedal_bits[:, 0] * 8 + pedal_bits[:, 1] * 4 + pedal_bits[:, 2] * 2 + pedal_bits[:, 3]

    counts["ioi"] += np.bincount(ioi_ms, minlength=MS_BINS)
    counts["duration"] += np.bincount(duration_ms, minlength=MS_BINS)
    counts["velocity"] += np.bincount(velocity, minlength=FEATURE_BINS["velocity"])
    counts["pedal"] += np.bincount(pedal, minlength=FEATURE_BINS["pedal"])
    return counts


def pred_ms_feature_counts(token_ids: Sequence[int], config, max_time_ms: float) -> Dict[str, np.ndarray]:
    arr = np.asarray(token_ids, dtype=np.int64)
    usable_len = (len(arr) // 8) * 8
    arr = arr[:usable_len].reshape(-1, 8)
    counts = {
        "velocity": np.zeros(FEATURE_BINS["velocity"], dtype=np.int64),
        "duration": np.zeros(MS_BINS, dtype=np.int64),
        "ioi": np.zeros(MS_BINS, dtype=np.int64),
        "pedal": np.zeros(FEATURE_BINS["pedal"], dtype=np.int64),
    }
    if arr.size == 0:
        return counts

    ioi_ticks = np.clip(arr[:, 1] - config.timing_start, 0, FEATURE_BINS["ioi"] - 1)
    duration_ticks = np.clip(arr[:, 3] - config.timing_start, 0, FEATURE_BINS["duration"] - 1)
    ioi_ms = np.clip(
        np.rint(np.expm1((ioi_ticks.astype(np.float64) / max_time_ms) * np.log1p(max_time_ms))).astype(np.int64),
        0,
        MS_BINS - 1,
    )
    duration_ms = np.clip(
        np.rint(np.expm1((duration_ticks.astype(np.float64) / max_time_ms) * np.log1p(max_time_ms))).astype(np.int64),
        0,
        MS_BINS - 1,
    )
    velocity = np.clip(arr[:, 2] - config.velocity_start, 0, FEATURE_BINS["velocity"] - 1)
    pedals = np.clip(arr[:, 4:8] - config.pedal_start, 0, 127)
    pedal_bits = (pedals > 64).astype(np.int64)
    pedal = pedal_bits[:, 0] * 8 + pedal_bits[:, 1] * 4 + pedal_bits[:, 2] * 2 + pedal_bits[:, 3]

    counts["ioi"] += np.bincount(ioi_ms, minlength=MS_BINS)
    counts["duration"] += np.bincount(duration_ms, minlength=MS_BINS)
    counts["velocity"] += np.bincount(velocity, minlength=FEATURE_BINS["velocity"])
    counts["pedal"] += np.bincount(pedal, minlength=FEATURE_BINS["pedal"])
    return counts


def hist(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.ones_like(counts) / len(counts)
    out = counts / total
    out = np.clip(out, 1e-12, None)
    return out / out.sum()


def js_ia(pred_counts: np.ndarray, target_counts: np.ndarray) -> Tuple[float, float, float]:
    p = hist(pred_counts)
    q = hist(target_counts)
    js_distance = float(jensenshannon(p, q, base=2))
    js_divergence = js_distance ** 2
    return js_distance, js_divergence, float(np.minimum(p, q).sum())


def build_asap_window_samples(eval_config: Dict, model_config, max_samples=None) -> List[Dict]:
    manifest = build_work_manifest(
        metadata_path=eval_config["metadata_path"],
        refined_dir=eval_config["refined_dir"],
        split="test",
        block_notes=eval_config.get("block_notes", 512),
        overlap_ratio=eval_config.get("overlap_ratio", 0.5),
        min_notes=eval_config.get("min_notes", 64),
        max_works=eval_config.get("max_eval_works"),
    )
    dataset = PianoCoReNodeSFTDataset(
        manifest,
        split="test",
        input_feature_mode="legacy",
        shuffle=False,
        seed=eval_config.get("seed", 42),
        max_performances_per_work=eval_config.get("max_eval_performances_per_work"),
        max_windows_per_work=eval_config.get("max_eval_windows_per_work"),
    )

    max_time_ms = float(eval_config.get("max_time_ms", 10000.0))
    samples = []
    for idx in tqdm(range(len(dataset)), desc="Tokenizing ASAP windows"):
        sample = dataset[idx]
        if sample.get("performance_dataset") != "ASAP":
            continue
        pitch_ids = sample["pitch_ids"]
        x_tokens = continuous_to_pt_tokens(
            pitch_ids,
            sample["continuous"],
            model_config,
            max_time_ms=max_time_ms,
            include_pedal=False,
        )
        y_tokens = continuous_to_pt_tokens(
            pitch_ids,
            sample["labels_continuous"],
            model_config,
            max_time_ms=max_time_ms,
            include_pedal=True,
        )
        samples.append(
            {
                "x": x_tokens,
                "label": y_tokens,
                "target_continuous": sample["labels_continuous"],
                "max_time_ms": max_time_ms,
                "notes": len(pitch_ids),
                "performance_id": sample.get("performance_id", "unknown"),
            }
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def save_jsonl(path: Path, samples: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def balanced_shards(samples: Sequence[Dict], num_shards: int) -> List[List[Dict]]:
    shards = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    for sample in sorted(samples, key=lambda item: item["notes"], reverse=True):
        idx = min(range(num_shards), key=lambda i: loads[i])
        shards[idx].append(sample)
        loads[idx] += sample["notes"]
    return shards


def generate_tokens(model, input_tokens: Sequence[int], config, device) -> List[int]:
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    logits_processor = LogitsProcessorList([
        BatchSparseForcedTokenProcessor(
            input_ids,
            config,
            target_len=len(input_tokens),
            origin_len=0,
            already=0.0,
            weight=1.0,
            progress_callback=None,
        )
    ])
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=len(input_tokens),
            do_sample=True,
            logits_processor=logits_processor,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
        )
    return output[0, 1:len(input_tokens) + 1].detach().cpu().numpy().tolist()


def worker_process(worker_id: int, gpu_id: int, shard_path: str, model_path: str, queue):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu_id}")
    model, config = load_pt_model_and_config(Path(model_path))
    model = model.to(device)
    samples = load_jsonl(Path(shard_path))

    pred_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    target_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    pred_ms_counts = {
        "velocity": np.zeros(FEATURE_BINS["velocity"], dtype=np.int64),
        "duration": np.zeros(MS_BINS, dtype=np.int64),
        "ioi": np.zeros(MS_BINS, dtype=np.int64),
        "pedal": np.zeros(FEATURE_BINS["pedal"], dtype=np.int64),
    }
    target_ms_counts = {
        "velocity": np.zeros(FEATURE_BINS["velocity"], dtype=np.int64),
        "duration": np.zeros(MS_BINS, dtype=np.int64),
        "ioi": np.zeros(MS_BINS, dtype=np.int64),
        "pedal": np.zeros(FEATURE_BINS["pedal"], dtype=np.int64),
    }
    processed = 0
    notes = 0
    start = time.time()

    for sample in samples:
        pred_tokens = generate_tokens(model, sample["x"], config, device)
        label_tokens = sample["label"][:len(pred_tokens)]
        usable_len = (min(len(pred_tokens), len(label_tokens)) // 8) * 8
        pred = token_feature_counts(pred_tokens[:usable_len], config)
        target = token_feature_counts(label_tokens[:usable_len], config)
        pred_ms = pred_ms_feature_counts(pred_tokens[:usable_len], config, float(sample.get("max_time_ms", 10000.0)))
        target_ms = target_ms_feature_counts(
            sample["target_continuous"][:usable_len // 8],
            float(sample.get("max_time_ms", 10000.0)),
        )
        for name in pred_counts:
            pred_counts[name] += pred[name]
            target_counts[name] += target[name]
            pred_ms_counts[name] += pred_ms[name]
            target_ms_counts[name] += target_ms[name]
        processed += 1
        notes += usable_len // 8

    queue.put(
        {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "processed": processed,
            "notes": notes,
            "seconds": time.time() - start,
            "pred_counts": pred_counts,
            "target_counts": target_counts,
            "pred_ms_counts": pred_ms_counts,
            "target_ms_counts": target_ms_counts,
        }
    )


def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "=" * 82,
        "PT Paper-Style Global Token Distribution Metrics",
        "=" * 82,
        f"{'Feature':<12} {'JS Dist (↓)':<15} {'JS Div (↓)':<15} {'IA (↑)':<15}",
        "-" * 82,
    ]
    for name in ["velocity", "duration", "ioi", "pedal"]:
        row = metrics[name]
        lines.append(f"{name:<12} {row['js_distance']:<15.6f} {row['js_divergence']:<15.6f} {row['ia']:<15.6f}")
    lines.extend(
        [
            "-" * 82,
            f"{'Overall':<12} {metrics['overall']['js_distance']:<15.6f} {metrics['overall']['js_divergence']:<15.6f} {metrics['overall']['ia']:<15.6f}",
            "=" * 82,
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", type=str, default="results/pt_node_asap_token_eval")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        eval_config = json.load(f)

    _, model_config = load_pt_model_and_config(Path(args.model_path))
    samples = build_asap_window_samples(eval_config, model_config, max_samples=args.max_samples)
    if not samples:
        raise ValueError("No ASAP windows selected")

    print("=" * 80)
    print("PT corrected ASAP-window token evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config}")
    print(f"ASAP windows: {len(samples)}")
    print(f"ASAP notes:   {sum(sample['notes'] for sample in samples)}")
    print(f"GPUs:         {torch.cuda.device_count()}")
    print(f"Workers/GPU:  {args.workers_per_gpu}")
    print()

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA GPUs available")

    total_workers = num_gpus * args.workers_per_gpu
    shards = balanced_shards(samples, total_workers)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for idx, shard in enumerate(shards):
        path = shard_dir / f"worker_{idx:02d}.jsonl"
        save_jsonl(path, shard)
        shard_paths.append(path)
    print(f"Notes per shard: {[sum(s['notes'] for s in shard) for shard in shards]}")

    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    processes = []
    for worker_id, shard_path in enumerate(shard_paths):
        gpu_id = worker_id % num_gpus
        process = mp.Process(
            target=worker_process,
            args=(worker_id, gpu_id, str(shard_path), args.model_path, queue),
        )
        process.start()
        processes.append(process)

    worker_results = []
    for _ in tqdm(range(len(processes)), desc="Collecting workers"):
        worker_results.append(queue.get())
    for process in processes:
        process.join()

    worker_results.sort(key=lambda row: row["worker_id"])
    for row in worker_results:
        print(
            f"worker {row['worker_id']:02d} gpu{row['gpu_id']} "
            f"samples={row['processed']} notes={row['notes']} time={row['seconds']:.1f}s"
        )

    pred_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    target_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    pred_ms_counts = {
        "velocity": np.zeros(FEATURE_BINS["velocity"], dtype=np.int64),
        "duration": np.zeros(MS_BINS, dtype=np.int64),
        "ioi": np.zeros(MS_BINS, dtype=np.int64),
        "pedal": np.zeros(FEATURE_BINS["pedal"], dtype=np.int64),
    }
    target_ms_counts = {
        "velocity": np.zeros(FEATURE_BINS["velocity"], dtype=np.int64),
        "duration": np.zeros(MS_BINS, dtype=np.int64),
        "ioi": np.zeros(MS_BINS, dtype=np.int64),
        "pedal": np.zeros(FEATURE_BINS["pedal"], dtype=np.int64),
    }
    for row in worker_results:
        for name in pred_counts:
            pred_counts[name] += row["pred_counts"][name]
            target_counts[name] += row["target_counts"][name]
            pred_ms_counts[name] += row["pred_ms_counts"][name]
            target_ms_counts[name] += row["target_ms_counts"][name]

    metrics = {}
    for name in ["velocity", "duration", "ioi", "pedal"]:
        js_distance, js_divergence, ia = js_ia(pred_counts[name], target_counts[name])
        metrics[name] = {
            "js_distance": js_distance,
            "js_divergence": js_divergence,
            "ia": ia,
        }
    metrics["overall"] = {
        "js_distance": float(np.mean([metrics[name]["js_distance"] for name in ["velocity", "duration", "ioi", "pedal"]])),
        "js_divergence": float(np.mean([metrics[name]["js_divergence"] for name in ["velocity", "duration", "ioi", "pedal"]])),
        "ia": float(np.mean([metrics[name]["ia"] for name in ["velocity", "duration", "ioi", "pedal"]])),
    }

    ms_metrics = {}
    for name in ["velocity", "duration", "ioi", "pedal"]:
        js_distance, js_divergence, ia = js_ia(pred_ms_counts[name], target_ms_counts[name])
        ms_metrics[name] = {
            "js_distance": js_distance,
            "js_divergence": js_divergence,
            "ia": ia,
        }
    ms_metrics["overall"] = {
        "js_distance": float(np.mean([ms_metrics[name]["js_distance"] for name in ["velocity", "duration", "ioi", "pedal"]])),
        "js_divergence": float(np.mean([ms_metrics[name]["js_divergence"] for name in ["velocity", "duration", "ioi", "pedal"]])),
        "ia": float(np.mean([ms_metrics[name]["ia"] for name in ["velocity", "duration", "ioi", "pedal"]])),
    }

    print()
    print("Token-space metrics, using the same flawed timing tokenization as training:")
    print(format_metrics(metrics))
    print()
    print("MS-space metrics, decoding flawed timing tokens back through log-normalized ms:")
    print(format_metrics(ms_metrics))

    result = {
        "evaluation_type": "pt_paper_style_global_token_distribution",
        "subset": "asap",
        "num_samples": len(samples),
        "num_notes": int(sum(sample["notes"] for sample in samples)),
        "model_path": args.model_path,
        "config": args.config,
        "workers_per_gpu": args.workers_per_gpu,
        "metrics": metrics,
        "ms_metrics": ms_metrics,
        "ms_metric_note": (
            "Timing predictions decode generated ticks as tick/max_time_ms log-normalized values, "
            "then quantize to integer milliseconds on [0, 10000]. Targets are decoded directly "
            "from node JSON log1p-normalized times."
        ),
        "worker_results": [
            {
                "worker_id": row["worker_id"],
                "gpu_id": row["gpu_id"],
                "processed": row["processed"],
                "notes": row["notes"],
                "seconds": row["seconds"],
            }
            for row in worker_results
        ],
    }
    output_path = output_dir / "pt_node_asap_token_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
