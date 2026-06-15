"""
Evaluate PianistTransformer on pre-tokenized SFT JSONL windows.

This avoids reconstructing PT tokens from node continuous features. It uses the
exact x/label tokens from the JSONL used by SFT, then computes global
distribution metrics over generated windows.
"""

import argparse
import json
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


DEFAULT_DATA_FILE = ROOT_DIR / "data" / "processed" / "sft" / "sft_pianocore_from_json.jsonl"
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "sft_pianocore_from_scratch" / "sft_2026-06-14-14-12-24"
FEATURE_BINS = {"velocity": 128, "duration": 5000, "ioi": 4991, "pedal": 16}


def load_pt_model_and_config(model_path: Path):
    from safetensors.torch import load_file
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    model_path = Path(model_path)
    weight_path = model_path / "model.safetensors" if model_path.is_dir() else model_path
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.with_name("config.json")
    config_dict = json.load(open(config_path, "r", encoding="utf-8"))
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


def binarize_pedal_labels(tokens: Sequence[int]) -> List[int]:
    out = []
    for idx, token in enumerate(tokens):
        if idx % 8 > 3:
            out.append(5261 + 127 if token >= 5261 + 64 else 5261)
        else:
            out.append(int(token))
    return out


def make_windows(tokens: Sequence[int], labels: Sequence[int], block_size: int, overlap_ratio: float) -> List[Dict]:
    block_size = block_size // 8 * 8
    stride = max(8, int(block_size * (1.0 - overlap_ratio)) // 8 * 8)
    total_len = min(len(tokens), len(labels)) // 8 * 8
    if total_len <= block_size:
        return [{"x": list(tokens[:total_len]), "label": binarize_pedal_labels(labels[:total_len])}]
    windows = []
    start = 0
    while start + block_size <= total_len:
        windows.append({"x": list(tokens[start:start + block_size]), "label": binarize_pedal_labels(labels[start:start + block_size])})
        start += stride
    if windows[-1]["x"] != list(tokens[total_len - block_size:total_len]):
        start = total_len - block_size
        windows.append({"x": list(tokens[start:total_len]), "label": binarize_pedal_labels(labels[start:total_len])})
    return windows


def load_asap_test_windows(data_file: Path, block_size: int, overlap_ratio: float, max_windows=None) -> List[Dict]:
    windows = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if '"split": "test"' not in line or "ASAP" not in line:
                continue
            item = json.loads(line)
            if "ASAP" not in str(item.get("performance_source", "")):
                continue
            for window in make_windows(item["x"], item["label"], block_size, overlap_ratio):
                window["performance_source"] = item.get("performance_source", "")
                windows.append(window)
                if max_windows is not None and len(windows) >= max_windows:
                    return windows
    return windows


def token_feature_counts(token_ids: Sequence[int], config) -> Dict[str, np.ndarray]:
    arr = np.asarray(token_ids, dtype=np.int64)
    arr = arr[: len(arr) // 8 * 8].reshape(-1, 8)
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


def hist(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    total = counts.sum()
    out = counts / total if total > 0 else np.ones_like(counts, dtype=np.float64) / len(counts)
    out = np.clip(out, 1e-12, None)
    return out / out.sum()


def counts_to_jsonable(counts: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
    return {name: values.astype(int).tolist() for name, values in counts.items()}


def histograms_to_jsonable(pred_counts: Dict[str, np.ndarray], target_counts: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[float]]]:
    return {
        name: {
            "pred": hist(pred_counts[name]).astype(float).tolist(),
            "target": hist(target_counts[name]).astype(float).tolist(),
        }
        for name in pred_counts
    }


def js_ia(pred_counts: np.ndarray, target_counts: np.ndarray) -> Tuple[float, float, float]:
    p = hist(pred_counts)
    q = hist(target_counts)
    js_distance = float(jensenshannon(p, q, base=2))
    return js_distance, js_distance ** 2, float(np.minimum(p, q).sum())


def balanced_shards(windows: Sequence[Dict], n: int) -> List[List[Dict]]:
    shards = [[] for _ in range(n)]
    loads = [0 for _ in range(n)]
    for window in sorted(windows, key=lambda w: len(w["x"]), reverse=True):
        idx = min(range(n), key=lambda i: loads[i])
        shards[idx].append(window)
        loads[idx] += len(window["x"])
    return shards


def save_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]


def generate_window(model, config, tokens: Sequence[int], device, do_sample: bool, temperature: float, top_p: float) -> List[int]:
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    processor = LogitsProcessorList([
        BatchSparseForcedTokenProcessor(input_ids, config, target_len=len(tokens), origin_len=0, already=0.0, weight=1.0, progress_callback=None)
    ])
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=len(tokens),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            logits_processor=processor,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
        )
    return generated[0, 1:len(tokens) + 1].detach().cpu().numpy().tolist()


def worker_process(
    worker_id: int,
    gpu_id: int,
    shard_path: str,
    model_path: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    queue,
):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu_id}")
    model, config = load_pt_model_and_config(Path(model_path))
    model = model.to(device)
    rows = load_jsonl(Path(shard_path))
    pred_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    target_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    start = time.time()
    notes = 0
    for row in rows:
        pred = generate_window(model, config, row["x"], device, do_sample=do_sample, temperature=temperature, top_p=top_p)
        label = row["label"][: len(pred)]
        usable_len = min(len(pred), len(label)) // 8 * 8
        pc = token_feature_counts(pred[:usable_len], config)
        tc = token_feature_counts(label[:usable_len], config)
        for name in pred_counts:
            pred_counts[name] += pc[name]
            target_counts[name] += tc[name]
        notes += usable_len // 8
    queue.put({
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "processed": len(rows),
        "notes": notes,
        "seconds": time.time() - start,
        "pred_counts": pred_counts,
        "target_counts": target_counts,
    })


def format_metrics(metrics):
    lines = ["=" * 78, "PT JSONL Window Metrics", "=" * 78, f"{'Feature':<12} {'JS Dist':<12} {'JS Div':<12} {'IA':<12}", "-" * 78]
    for name in ["velocity", "duration", "ioi", "pedal"]:
        row = metrics[name]
        lines.append(f"{name:<12} {row['js_distance']:<12.6f} {row['js_divergence']:<12.6f} {row['ia']:<12.6f}")
    row = metrics["overall"]
    lines.extend(["-" * 78, f"{'overall':<12} {row['js_distance']:<12.6f} {row['js_divergence']:<12.6f} {row['ia']:<12.6f}", "=" * 78])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", type=str, default="results/pt_jsonl_window_eval")
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-windows", type=int, default=104)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = load_asap_test_windows(Path(args.data_file), args.block_size, args.overlap_ratio, args.max_windows)
    print(f"ASAP JSONL windows: {len(windows)}")
    print(f"ASAP JSONL notes:   {sum(len(w['x']) // 8 for w in windows)}")
    print(f"Generation: do_sample={args.do_sample}, temperature={args.temperature}, top_p={args.top_p}")
    num_gpus = torch.cuda.device_count()
    total_workers = num_gpus * args.workers_per_gpu
    shards = balanced_shards(windows, total_workers)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, shard in enumerate(shards):
        path = shard_dir / f"worker_{idx:02d}.jsonl"
        save_jsonl(path, shard)
        paths.append(path)
    print(f"Notes per shard: {[sum(len(w['x']) // 8 for w in shard) for shard in shards]}")

    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    processes = []
    for worker_id, path in enumerate(paths):
        process = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                worker_id % num_gpus,
                str(path),
                args.model_path,
                args.do_sample,
                args.temperature,
                args.top_p,
                queue,
            ),
        )
        process.start()
        processes.append(process)

    rows = [queue.get() for _ in tqdm(range(len(processes)), desc="Collecting workers")]
    for process in processes:
        process.join()
    rows.sort(key=lambda r: r["worker_id"])
    for row in rows:
        print(f"worker {row['worker_id']:02d} gpu{row['gpu_id']} windows={row['processed']} notes={row['notes']} time={row['seconds']:.1f}s")

    pred_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    target_counts = {name: np.zeros(size, dtype=np.int64) for name, size in FEATURE_BINS.items()}
    for row in rows:
        for name in pred_counts:
            pred_counts[name] += row["pred_counts"][name]
            target_counts[name] += row["target_counts"][name]
    metrics = {}
    for name in ["velocity", "duration", "ioi", "pedal"]:
        js_distance, js_divergence, ia = js_ia(pred_counts[name], target_counts[name])
        metrics[name] = {"js_distance": js_distance, "js_divergence": js_divergence, "ia": ia}
    metrics["overall"] = {
        "js_distance": float(np.mean([metrics[name]["js_distance"] for name in ["velocity", "duration", "ioi", "pedal"]])),
        "js_divergence": float(np.mean([metrics[name]["js_divergence"] for name in ["velocity", "duration", "ioi", "pedal"]])),
        "ia": float(np.mean([metrics[name]["ia"] for name in ["velocity", "duration", "ioi", "pedal"]])),
    }
    print(format_metrics(metrics))
    generation_config = {
        "model_path": args.model_path,
        "data_file": args.data_file,
        "num_windows": len(windows),
        "num_notes": int(sum(len(w["x"]) // 8 for w in windows)),
        "block_size": args.block_size,
        "overlap_ratio": args.overlap_ratio,
        "max_windows": args.max_windows,
        "workers_per_gpu": args.workers_per_gpu,
        "total_workers": total_workers,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "feature_bins": FEATURE_BINS,
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2, ensure_ascii=False)
    with open(output_dir / "pred_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts_to_jsonable(pred_counts), f, indent=2)
    with open(output_dir / "target_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts_to_jsonable(target_counts), f, indent=2)
    with open(output_dir / "feature_histograms.json", "w", encoding="utf-8") as f:
        json.dump(histograms_to_jsonable(pred_counts, target_counts), f, indent=2)
    result = {
        "evaluation_type": "pt_jsonl_window_generation",
        "num_windows": len(windows),
        "num_notes": int(sum(len(w["x"]) // 8 for w in windows)),
        "generation_config": generation_config,
        "metrics": metrics,
        "worker_results": [{k: row[k] for k in ["worker_id", "gpu_id", "processed", "notes", "seconds"]} for row in rows],
    }
    output_path = output_dir / "pt_jsonl_window_metrics.json"
    json.dump(result, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
