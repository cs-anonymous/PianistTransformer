"""
Evaluate PianistTransformer on ASAP test samples stored in the pre-tokenized
SFT JSONL. This follows the original PT paper more closely than the legacy
continuous EPRMetrics(bins=100) path:

- Velocity, Duration, IOI: global token distributions over natural token bins
- Pedal: 16-way joint configuration from the four pedal tokens

The script filters `split=test` and `performance_source` containing ASAP,
then runs multi-process generation across GPUs and aggregates distribution
counts directly.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from transformers import LogitsProcessorList

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.model.generate import BatchSparseForcedTokenProcessor


DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "sft_pianocore_from_scratch" / "sft_2026-06-14-14-12-24"
DEFAULT_DATA_FILE = ROOT_DIR / "data" / "processed" / "sft" / "sft_pianocore_from_json.jsonl"


@dataclass(frozen=True)
class TokenSpec:
    name: str
    size: int


FEATURE_SPECS = {
    "velocity": TokenSpec("velocity", 128),
    "duration": TokenSpec("duration", 5000),
    "ioi": TokenSpec("ioi", 4991),
    "pedal": TokenSpec("pedal", 16),
}


def resolve_model_paths(model_path: Path) -> Tuple[Path, Path]:
    model_path = Path(model_path)
    if model_path.is_dir():
        config_path = model_path / "config.json"
        weight_path = model_path / "model.safetensors"
    else:
        config_path = model_path.with_name("config.json")
        weight_path = model_path
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for model at {model_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing model weights for model at {model_path}")
    return weight_path, config_path


def load_pt_model_and_config(model_path: Path):
    from safetensors.torch import load_file
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    weight_path, config_path = resolve_model_paths(model_path)
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
        max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
        attention_dropout=config_dict.get("attention_dropout", 0.0),
    )

    model = PianoT5Gemma(config)
    state_dict = load_file(str(weight_path))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


def is_asap_test_sample(sample: Dict) -> bool:
    if sample.get("split") != "test":
        return False
    perf_source = str(sample.get("performance_source", ""))
    return "ASAP" in perf_source


def build_asap_test_file(data_file: Path, filtered_file: Path) -> Tuple[int, int]:
    filtered_file.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    note_count = 0
    with open(data_file, "r", encoding="utf-8") as src, open(filtered_file, "w", encoding="utf-8") as dst:
        for line in src:
            if '"split": "test"' not in line or "ASAP" not in line:
                continue
            sample = json.loads(line)
            if not is_asap_test_sample(sample):
                continue
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            sample_count += 1
            note_count += len(sample["x"]) // 8
    return sample_count, note_count


def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def save_jsonl(path: Path, samples: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def slide_windows(total_len: int, window_len: int, overlap_ratio: float = 0.5) -> List[Tuple[int, int]]:
    window_len = max(8, window_len // 8 * 8)
    if total_len <= window_len:
        return [(0, total_len)]

    stride = max(8, int(window_len * (1.0 - overlap_ratio)) // 8 * 8)
    windows = []
    start = 0
    while start + window_len <= total_len:
        windows.append((start, start + window_len))
        start += stride
    if windows[-1][1] != total_len:
        last_start = max(0, total_len - window_len)
        windows.append((last_start, total_len))

    deduped = []
    seen = set()
    for window in windows:
        if window not in seen:
            deduped.append(window)
            seen.add(window)
    return deduped


def generate_windowed_tokens(model, input_tokens: Sequence[int], config, device, block_size: int = 4096,
                             overlap_ratio: float = 0.5) -> List[int]:
    total_len = len(input_tokens)
    windows = slide_windows(total_len, block_size, overlap_ratio=overlap_ratio)
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

    output_list = []
    res_tensor = None

    for i, (start, end) in enumerate(windows):
        window_input = input_ids[:, start:end]
        logits_processor = LogitsProcessorList([
            BatchSparseForcedTokenProcessor(
                window_input,
                config,
                target_len=end - start,
                origin_len=start,
                already=0.0,
                weight=1.0,
                progress_callback=None,
            )
        ])

        if i == 0:
            output = model.generate(
                input_ids=window_input,
                max_new_tokens=end - start,
                do_sample=True,
                logits_processor=logits_processor,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
            res_tensor = output[:, 1:]
        else:
            last_start, last_end = windows[i - 1]
            overlap = start - last_start
            trim = int(((last_end - last_start) - overlap) * 0.2)
            decoder_input_ids = output_list[i - 1][:, start - last_start:last_end - last_start - trim]
            bos = torch.tensor([[config.bos_token_id]], dtype=torch.long, device=device)
            decoder_input_ids = torch.cat([bos, decoder_input_ids], dim=1)
            output = model.generate(
                input_ids=window_input,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=end - last_end + trim,
                do_sample=True,
                logits_processor=logits_processor,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
            res_tensor = torch.cat(
                [res_tensor[:, :-trim], output[:, -(end - last_end + trim):]],
                dim=1,
            )

        output_list.append(output)

    pred_tokens = res_tensor[0].detach().cpu().numpy().tolist()
    return pred_tokens[:total_len]


def extract_feature_counts(token_ids: Sequence[int], config) -> Dict[str, np.ndarray]:
    arr = np.asarray(token_ids, dtype=np.int64)
    usable_len = (len(arr) // 8) * 8
    arr = arr[:usable_len].reshape(-1, 8)

    counts = {
        "velocity": np.zeros(FEATURE_SPECS["velocity"].size, dtype=np.int64),
        "duration": np.zeros(FEATURE_SPECS["duration"].size, dtype=np.int64),
        "ioi": np.zeros(FEATURE_SPECS["ioi"].size, dtype=np.int64),
        "pedal": np.zeros(FEATURE_SPECS["pedal"].size, dtype=np.int64),
    }

    if arr.size == 0:
        return counts

    ioi = np.clip(arr[:, 1] - config.timing_start, 0, FEATURE_SPECS["ioi"].size - 1)
    duration = np.clip(arr[:, 3] - config.timing_start, 0, FEATURE_SPECS["duration"].size - 1)
    velocity = np.clip(arr[:, 2] - config.velocity_start, 0, FEATURE_SPECS["velocity"].size - 1)
    pedal_tokens = np.clip(arr[:, 4:8] - config.pedal_start, 0, 127)

    pedal_binary = (pedal_tokens > 64).astype(np.int64)
    pedal = (
        pedal_binary[:, 0] * 8
        + pedal_binary[:, 1] * 4
        + pedal_binary[:, 2] * 2
        + pedal_binary[:, 3]
    )

    counts["velocity"] += np.bincount(velocity, minlength=FEATURE_SPECS["velocity"].size)
    counts["duration"] += np.bincount(duration, minlength=FEATURE_SPECS["duration"].size)
    counts["ioi"] += np.bincount(ioi, minlength=FEATURE_SPECS["ioi"].size)
    counts["pedal"] += np.bincount(pedal, minlength=FEATURE_SPECS["pedal"].size)
    return counts


def hist_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.ones_like(counts, dtype=np.float64) / len(counts)
    hist = counts / total
    hist = np.clip(hist, 1e-12, None)
    hist /= hist.sum()
    return hist


def compute_js_ia(pred_counts: np.ndarray, target_counts: np.ndarray) -> Tuple[float, float]:
    p = hist_from_counts(pred_counts)
    q = hist_from_counts(target_counts)
    js = float(jensenshannon(p, q, base=2) ** 2)
    ia = float(np.minimum(p, q).sum())
    return js, ia


def format_results(results: Dict[str, Dict[str, float]]) -> str:
    lines = []
    lines.append("=" * 86)
    lines.append("PT Global Token Distribution Metrics")
    lines.append("=" * 86)
    lines.append(f"{'Feature':<12} {'JS Div (↓)':<15} {'IA (↑)':<15}")
    lines.append("-" * 86)
    for feature in ["velocity", "duration", "ioi", "pedal"]:
        r = results[feature]
        lines.append(f"{feature:<12} {r['js']:<15.6f} {r['ia']:<15.6f}")
    lines.append("-" * 86)
    lines.append(f"{'Overall':<12} {results['overall']['js']:<15.6f} {results['overall']['ia']:<15.6f}")
    lines.append("=" * 86)
    return "\n".join(lines)


def split_round_robin_balanced(samples: Sequence[Dict], num_shards: int) -> List[List[Dict]]:
    buckets: List[List[Dict]] = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    ordered = sorted(samples, key=lambda s: len(s["x"]), reverse=True)
    for sample in ordered:
        idx = min(range(num_shards), key=lambda i: loads[i])
        buckets[idx].append(sample)
        loads[idx] += len(sample["x"])
    return buckets


def worker_process(worker_id: int, gpu_id: int, shard_path: str, model_path: str,
                   block_size: int, overlap_ratio: float, result_queue):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu_id}")
    model, config = load_pt_model_and_config(Path(model_path))
    model = model.to(device)

    shard_samples = load_jsonl(Path(shard_path))
    pred_counts = {
        "velocity": np.zeros(FEATURE_SPECS["velocity"].size, dtype=np.int64),
        "duration": np.zeros(FEATURE_SPECS["duration"].size, dtype=np.int64),
        "ioi": np.zeros(FEATURE_SPECS["ioi"].size, dtype=np.int64),
        "pedal": np.zeros(FEATURE_SPECS["pedal"].size, dtype=np.int64),
    }
    target_counts = {
        "velocity": np.zeros(FEATURE_SPECS["velocity"].size, dtype=np.int64),
        "duration": np.zeros(FEATURE_SPECS["duration"].size, dtype=np.int64),
        "ioi": np.zeros(FEATURE_SPECS["ioi"].size, dtype=np.int64),
        "pedal": np.zeros(FEATURE_SPECS["pedal"].size, dtype=np.int64),
    }

    processed = 0
    notes = 0
    t0 = time.time()

    for sample in shard_samples:
        x_tokens = sample["x"]
        y_tokens = sample["label"]

        if len(x_tokens) != len(y_tokens):
            min_len = min(len(x_tokens), len(y_tokens))
            x_tokens = x_tokens[:min_len]
            y_tokens = y_tokens[:min_len]

        pred_tokens = generate_windowed_tokens(
            model,
            x_tokens,
            config,
            device,
            block_size=block_size,
            overlap_ratio=overlap_ratio,
        )
        usable_len = min(len(pred_tokens), len(y_tokens))
        usable_len = (usable_len // 8) * 8
        pred_tokens = pred_tokens[:usable_len]
        y_tokens = y_tokens[:usable_len]

        pred_feat = extract_feature_counts(pred_tokens, config)
        target_feat = extract_feature_counts(y_tokens, config)

        for k in pred_counts:
            pred_counts[k] += pred_feat[k]
            target_counts[k] += target_feat[k]

        processed += 1
        notes += usable_len // 8

    result_queue.put(
        {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "processed": processed,
            "notes": notes,
            "seconds": time.time() - t0,
            "pred_counts": pred_counts,
            "target_counts": target_counts,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", type=str, default="results/pt_asap_jsonl_eval")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refresh-filter", action="store_true")
    args = parser.parse_args()

    data_file = Path(args.data_file)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PT ASAP JSONL Evaluation")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Model path: {model_path}")
    print(f"GPUs: {torch.cuda.device_count()}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Block size: {args.block_size}")
    print(f"Overlap ratio: {args.overlap_ratio}")
    print()

    filtered_file = output_dir / "asap_test_filtered.jsonl"
    if args.refresh_filter or not filtered_file.exists():
        print("Building ASAP test filter from JSONL...")
        sample_count, note_count = build_asap_test_file(data_file, filtered_file)
        print(f"Filtered ASAP test samples: {sample_count}")
        print(f"Filtered ASAP test notes:   {note_count}")
    else:
        print(f"Using existing filtered file: {filtered_file}")

    samples = load_jsonl(filtered_file)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("No ASAP test samples found")

    total_notes = sum(len(sample["x"]) // 8 for sample in samples)
    print(f"Loaded samples: {len(samples)}")
    print(f"Total notes:    {total_notes}")

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA GPUs available")
    total_workers = num_gpus * args.workers_per_gpu
    shards = split_round_robin_balanced(samples, total_workers)

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    shard_loads = []
    for i, shard in enumerate(shards):
        shard_path = shard_dir / f"worker_{i:02d}.jsonl"
        save_jsonl(shard_path, shard)
        shard_paths.append(shard_path)
        shard_loads.append(sum(len(sample["x"]) for sample in shard))

    print(f"Shards: {len(shards)}")
    print(f"Token load per shard: {[load // 8 for load in shard_loads]}")

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    processes = []
    for worker_id, shard_path in enumerate(shard_paths):
        gpu_id = worker_id % num_gpus
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                gpu_id,
                str(shard_path),
                str(model_path),
                args.block_size,
                args.overlap_ratio,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    all_results = []
    for _ in tqdm(range(len(processes)), desc="Collecting workers"):
        all_results.append(result_queue.get())

    for p in processes:
        p.join()

    all_results = sorted(all_results, key=lambda x: x["worker_id"])
    total_seconds = sum(r["seconds"] for r in all_results)
    wall_seconds = max(r["seconds"] for r in all_results) if all_results else 0.0
    print()
    for r in all_results:
        print(
            f"worker {r['worker_id']:02d} gpu{r['gpu_id']} "
            f"samples={r['processed']} notes={r['notes']} time={r['seconds']:.1f}s"
        )

    pred_counts = {k: np.zeros(v.size, dtype=np.int64) for k, v in FEATURE_SPECS.items()}
    target_counts = {k: np.zeros(v.size, dtype=np.int64) for k, v in FEATURE_SPECS.items()}
    for r in all_results:
        for feature in pred_counts:
            pred_counts[feature] += r["pred_counts"][feature]
            target_counts[feature] += r["target_counts"][feature]

    results = {}
    for feature in ["velocity", "duration", "ioi", "pedal"]:
        js, ia = compute_js_ia(pred_counts[feature], target_counts[feature])
        results[feature] = {"js": js, "ia": ia}

    results["overall"] = {
        "js": float(np.mean([results[f]["js"] for f in ["velocity", "duration", "ioi", "pedal"]])),
        "ia": float(np.mean([results[f]["ia"] for f in ["velocity", "duration", "ioi", "pedal"]])),
    }

    print()
    print(format_results(results))

    output = {
        "evaluation_type": "pt_global_token_distribution",
        "data_file": str(data_file),
        "model_path": str(model_path),
        "num_samples": len(samples),
        "num_notes": total_notes,
        "workers_per_gpu": args.workers_per_gpu,
        "total_workers": total_workers,
        "wall_seconds": wall_seconds,
        "sum_worker_seconds": total_seconds,
        "metrics": results,
    }
    output_file = output_dir / "pt_asap_global_token_metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
