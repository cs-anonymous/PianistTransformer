import argparse
import json
import math
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.evaluate.evaluate_inr_saved_midis import score_level_metrics, aggregate_score_metrics
from src.model.integrated_pianoformer import _target5_to_raw7
from src.train.train_inr import performance_dev_velocity_pedal4_binary_rows
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render score-MIDI and per-note MLN3 empirical baselines, then evaluate PP/PN Wass."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument(
        "--donor-pool",
        choices=["test_asap", "all_asap", "all_processed"],
        default="all_processed",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--fit-space-eps", type=float, default=1e-4)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--em-iters", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--variants",
        type=str,
        default="independent,joint_timing,pedal_bce,joint_timing_pedal_bce,split_timing,split_timing_pedal_bce",
        help=(
            "Comma-separated MLN oracle variants: independent, joint_timing, pedal_bce, "
            "joint_timing_pedal_bce, split_timing, split_timing_pedal_bce."
        ),
    )
    parser.add_argument("--max-plot-points", type=int, default=200000)
    parser.add_argument("--score-copy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_stem(score_source):
    return Path(score_source).with_suffix("").as_posix().replace("/", "__")


def stable_seed(base_seed, *parts):
    import hashlib

    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def donor_pool_for_name(pool_name, all_perfs, target_perfs, performance_dataset):
    if pool_name == "test_asap":
        return list(target_perfs)
    if pool_name == "all_asap":
        return [
            perf
            for perf in all_perfs
            if str(perf.get("performance_dataset") or "") == str(performance_dataset)
        ]
    if pool_name == "all_processed":
        return list(all_perfs)
    raise ValueError(f"Unknown donor pool: {pool_name}")


def logit(values, eps):
    values = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(values) - np.log1p(-values)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def fit_gmm_1d(values, components=3, em_iters=50, sigma_min=0.05, sigma_max=3.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "weights": np.asarray([1.0], dtype=np.float64),
            "means": np.asarray([0.0], dtype=np.float64),
            "sigmas": np.asarray([1.0], dtype=np.float64),
        }
    if len(values) == 1:
        return {
            "weights": np.asarray([1.0], dtype=np.float64),
            "means": values.copy(),
            "sigmas": np.asarray([sigma_min], dtype=np.float64),
        }

    k = min(int(components), len(values))
    qs = np.linspace(0.0, 1.0, k + 2)[1:-1]
    means = np.quantile(values, qs)
    sig = float(np.std(values))
    sig = min(max(sig if sig > 0 else sigma_min, sigma_min), sigma_max)
    sigmas = np.full(k, sig, dtype=np.float64)
    weights = np.full(k, 1.0 / k, dtype=np.float64)
    floor = 1e-12

    for _ in range(max(1, int(em_iters))):
        diff = values[:, None] - means[None, :]
        log_prob = (
            np.log(weights[None, :] + floor)
            - np.log(sigmas[None, :] + floor)
            - 0.5 * (diff / sigmas[None, :]) ** 2
        )
        log_prob -= np.max(log_prob, axis=1, keepdims=True)
        resp = np.exp(log_prob)
        resp /= np.sum(resp, axis=1, keepdims=True).clip(min=floor)
        nk = np.sum(resp, axis=0).clip(min=floor)
        weights = nk / np.sum(nk)
        means = np.sum(resp * values[:, None], axis=0) / nk
        var = np.sum(resp * (values[:, None] - means[None, :]) ** 2, axis=0) / nk
        sigmas = np.sqrt(np.clip(var, sigma_min**2, sigma_max**2))

    order = np.argsort(means)
    return {
        "weights": weights[order],
        "means": means[order],
        "sigmas": sigmas[order],
    }


def sample_gmm_1d(params, rng):
    comp = int(rng.choice(len(params["weights"]), p=params["weights"]))
    return float(rng.normal(params["means"][comp], params["sigmas"][comp]))


def fit_gmm_nd(values, components=3, em_iters=20, sigma_min=0.05, sigma_max=3.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        return {
            "weights": np.asarray([1.0], dtype=np.float64),
            "means": np.zeros((1, 2), dtype=np.float64),
            "covs": np.eye(2, dtype=np.float64)[None, :, :],
        }
    dim = values.shape[1]
    if len(values) == 1:
        return {
            "weights": np.asarray([1.0], dtype=np.float64),
            "means": values.copy(),
            "covs": (np.eye(dim, dtype=np.float64) * sigma_min**2)[None, :, :],
        }

    k = min(int(components), len(values))
    order = np.argsort(values[:, 0])
    init_idx = np.linspace(0, len(values) - 1, k).round().astype(int)
    means = values[order[init_idx]].copy()
    cov = np.cov(values.T)
    if cov.ndim == 0:
        cov = np.eye(dim, dtype=np.float64) * float(cov)
    cov = np.asarray(cov, dtype=np.float64)
    cov += np.eye(dim, dtype=np.float64) * sigma_min**2
    covs = np.repeat(cov[None, :, :], k, axis=0)
    weights = np.full(k, 1.0 / k, dtype=np.float64)
    floor = 1e-12
    eye = np.eye(dim, dtype=np.float64)

    for _ in range(max(1, int(em_iters))):
        log_prob = []
        for comp_idx in range(k):
            comp_cov = covs[comp_idx] + eye * sigma_min**2
            sign, logdet = np.linalg.slogdet(comp_cov)
            if sign <= 0:
                comp_cov = np.diag(np.clip(np.diag(comp_cov), sigma_min**2, sigma_max**2))
                sign, logdet = np.linalg.slogdet(comp_cov)
            inv_cov = np.linalg.pinv(comp_cov)
            diff = values - means[comp_idx]
            maha = np.sum((diff @ inv_cov) * diff, axis=1)
            log_prob.append(np.log(weights[comp_idx] + floor) - 0.5 * (logdet + maha))
        log_prob = np.stack(log_prob, axis=1)
        log_prob -= np.max(log_prob, axis=1, keepdims=True)
        resp = np.exp(log_prob)
        resp /= np.sum(resp, axis=1, keepdims=True).clip(min=floor)
        nk = np.sum(resp, axis=0).clip(min=floor)
        weights = nk / np.sum(nk)
        means = (resp.T @ values) / nk[:, None]
        for comp_idx in range(k):
            diff = values - means[comp_idx]
            covs[comp_idx] = (diff.T * resp[:, comp_idx]) @ diff / nk[comp_idx]
            covs[comp_idx] += eye * sigma_min**2
            diag = np.clip(np.diag(covs[comp_idx]), sigma_min**2, sigma_max**2)
            covs[comp_idx] = covs[comp_idx].copy()
            np.fill_diagonal(covs[comp_idx], diag)

    order = np.argsort(means[:, 0])
    return {
        "weights": weights[order],
        "means": means[order],
        "covs": covs[order],
    }


def sample_gmm_nd(params, rng):
    comp = int(rng.choice(len(params["weights"]), p=params["weights"]))
    cov = np.asarray(params["covs"][comp], dtype=np.float64)
    cov = (cov + cov.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    cov = (eigvecs * eigvals) @ eigvecs.T
    return rng.multivariate_normal(params["means"][comp], cov)


def render_midi(pitch, raw_rows, path, max_time_ms):
    midi = note_features_to_midi(
        pitch=pitch,
        continuous=raw_rows.tolist() if isinstance(raw_rows, np.ndarray) else raw_rows,
        target_ticks_per_beat=500,
        target_tempo=120,
        max_time_ms=max_time_ms,
        normalized=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def make_manifest(items, protocol, num_samples, path):
    payload = {"protocol": protocol, "num_samples": int(num_samples), "items": items}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def evaluate_manifest(payload, manifest_path, output_json):
    rows = [score_level_metrics(item) for item in tqdm(payload["items"], desc=f"eval {manifest_path.name}")]
    output = {
        "prediction_manifest": str(manifest_path.resolve()),
        "protocol": payload["protocol"],
        "num_samples": payload["num_samples"],
        "num_scores": len(rows),
        "aggregate": {
            "pn_wass": aggregate_score_metrics(rows, "pn_wass"),
            "pp_wass": aggregate_score_metrics(rows, "pp_wass"),
        },
        "scores": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def sample_target_variant(
    variant,
    target_shape,
    score_shared_raw,
    scalar_params,
    joint_timing_params,
    split_timing_params,
    pedal_probs,
    rng,
):
    sampled = np.zeros(target_shape, dtype=np.float64)
    score_shared_raw = np.asarray(score_shared_raw, dtype=np.float64)
    use_joint = variant in {"joint_timing", "joint_timing_pedal_bce"}
    use_split = variant in {"split_timing", "split_timing_pedal_bce"}
    use_pedal_bce = variant in {"pedal_bce", "joint_timing_pedal_bce", "split_timing_pedal_bce"}

    for note_idx in range(target_shape[0]):
        if use_joint:
            sampled[note_idx, :2] = sigmoid(sample_gmm_nd(joint_timing_params[note_idx], rng))
        elif use_split:
            flag = "zero" if float(score_shared_raw[note_idx][0]) <= 0.0 else "nonzero"
            for feature_idx in (0, 1):
                sampled[note_idx, feature_idx] = sigmoid(
                    sample_gmm_1d(split_timing_params[flag][feature_idx], rng)
                )
        else:
            for feature_idx in (0, 1):
                sampled[note_idx, feature_idx] = sigmoid(
                    sample_gmm_1d(scalar_params[note_idx][feature_idx], rng)
                )

        sampled[note_idx, 2] = sigmoid(sample_gmm_1d(scalar_params[note_idx][2], rng))
        if use_pedal_bce:
            sampled[note_idx, 3:7] = (
                rng.random(4) < pedal_probs[note_idx]
            ).astype(np.float64)
        else:
            for feature_idx in range(3, target_shape[1]):
                sampled[note_idx, feature_idx] = sigmoid(
                    sample_gmm_1d(scalar_params[note_idx][feature_idx], rng)
                )
    return sampled


def write_duration_diag(path, target_stack, score_shared_raw, variant_samples, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    score_shared_raw = np.asarray(score_shared_raw, dtype=np.float64)
    gt_raw_duration = []
    for perf_targets in target_stack:
        raw_rows = _target5_to_raw7(
            torch.tensor(score_shared_raw, dtype=torch.float32),
            torch.tensor(perf_targets, dtype=torch.float32),
            config=config,
        ).cpu().numpy()
        gt_raw_duration.append(raw_rows[:, 1])
    payload = {
        "gt_norm_duration": target_stack[:, :, 1].reshape(-1),
        "gt_raw_duration": np.stack(gt_raw_duration, axis=0).reshape(-1),
    }
    for variant, samples in variant_samples.items():
        payload[f"{variant}_norm_duration"] = np.stack(
            [sample["target"][:, 1] for sample in samples],
            axis=0,
        ).reshape(-1)
        payload[f"{variant}_raw_duration"] = np.stack(
            [sample["raw"][:, 1] for sample in samples],
            axis=0,
        ).reshape(-1)
    payload["score_raw_duration"] = score_shared_raw[:, 1]
    np.savez_compressed(path, **payload)


def plot_duration_diagnostics(diag_paths, output_dir, variants, max_points, seed):
    if not diag_paths:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrays = {"gt_norm_duration": [], "gt_raw_duration": [], "score_raw_duration": []}
    for variant in variants:
        arrays[f"{variant}_norm_duration"] = []
        arrays[f"{variant}_raw_duration"] = []
    for diag_path in diag_paths:
        data = np.load(diag_path)
        for key in arrays:
            if key in data:
                arrays[key].append(np.asarray(data[key], dtype=np.float64))
    arrays = {key: np.concatenate(parts) for key, parts in arrays.items() if parts}

    def downsample(values):
        values = values[np.isfinite(values)]
        if len(values) > max_points:
            idx = rng.choice(len(values), size=max_points, replace=False)
            values = values[idx]
        return values

    colors = {
        "GT": "black",
        "score": "#777777",
        "independent": "#1f77b4",
        "joint_timing": "#ff7f0e",
        "pedal_bce": "#2ca02c",
        "joint_timing_pedal_bce": "#d62728",
        "split_timing": "#9467bd",
        "split_timing_pedal_bce": "#8c564b",
    }

    raw_path = output_dir / "duration_distribution_raw_ms.png"
    plt.figure(figsize=(11, 6))
    if "gt_raw_duration" in arrays:
        values = np.clip(downsample(arrays["gt_raw_duration"]), 0.0, 2000.0)
        plt.hist(values, bins=160, density=True, histtype="step", linewidth=1.8, label="GT", color=colors["GT"])
    if "score_raw_duration" in arrays:
        values = np.clip(downsample(arrays["score_raw_duration"]), 0.0, 2000.0)
        plt.hist(values, bins=160, density=True, histtype="step", linewidth=1.2, label="score", color=colors["score"])
    for variant in variants:
        key = f"{variant}_raw_duration"
        if key in arrays:
            values = np.clip(downsample(arrays[key]), 0.0, 2000.0)
            plt.hist(values, bins=160, density=True, histtype="step", linewidth=1.2, label=variant, color=colors.get(variant))
    plt.xlabel("duration raw ms, clipped to 2000")
    plt.ylabel("density")
    plt.title("MLN oracle duration distribution in raw ms")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(raw_path, dpi=160)
    plt.close()

    norm_path = output_dir / "duration_distribution_normalized_logdev.png"
    plt.figure(figsize=(11, 6))
    if "gt_norm_duration" in arrays:
        plt.hist(
            downsample(arrays["gt_norm_duration"]),
            bins=120,
            range=(0.0, 1.0),
            density=True,
            histtype="step",
            linewidth=1.7,
            label="GT target",
            color=colors["GT"],
        )
    for variant in variants:
        key = f"{variant}_norm_duration"
        if key in arrays:
            plt.hist(
                downsample(arrays[key]),
                bins=120,
                range=(0.0, 1.0),
                density=True,
                histtype="step",
                linewidth=1.2,
                label=variant,
                color=colors.get(variant),
            )
    plt.xlabel("normalized duration target")
    plt.ylabel("density")
    plt.title("MLN oracle duration distribution in normalized target space")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(norm_path, dpi=160)
    plt.close()

    summary_rows = []
    for key, values in arrays.items():
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        summary_rows.append(
            {
                "series": key,
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
            }
        )
    summary_csv = output_dir / "duration_distribution_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    return {
        "raw_duration_png": str(raw_path.resolve()),
        "normalized_duration_png": str(norm_path.resolve()),
        "duration_summary_csv": str(summary_csv.resolve()),
    }


def process_manifest_item(task):
    item, config, args_dict = task
    output_dir = Path(args_dict["output_dir"])
    refined_midi_dir = Path(config["refined_dir"]).parent / "refined"
    midi_dir = output_dir / "midis"
    score_item = None
    fit_row = None
    skipped = []
    variant_items = {}

    work_path = Path(item["path"])
    work = json.loads(work_path.read_text(encoding="utf-8"))
    score = work["score"]
    pitch = [int(value) for value in score["pitch"]]
    score_shared_raw = score["score_raw"]
    score_source = item["score_source"]
    score_stem = safe_stem(score_source)

    all_perfs = [
        perf
        for perf in work.get("performances", [])
        if perf.get("label_shared_raw") is not None and perf.get("label_pedal4_raw") is not None
    ]
    by_source = {perf.get("performance_source"): perf for perf in all_perfs}
    target_perfs = [
        by_source[source]
        for source in item.get("selected_performance_sources", [])
        if source in by_source
    ]
    gt_paths = [str((refined_midi_dir / perf.get("performance_source")).resolve()) for perf in target_perfs]
    gt_paths = [path for path in gt_paths if Path(path).exists()]
    if not gt_paths:
        skipped.append({"score_source": score_source, "reason": "no_gt_midi"})
        return score_item, variant_items, fit_row, skipped, None

    source_score_path = refined_midi_dir / score_source
    if not source_score_path.exists():
        skipped.append({"score_source": score_source, "reason": "missing_score_midi"})
    else:
        if args_dict["score_copy"]:
            score_pred_path = midi_dir / "score_midi" / f"{score_stem}__score.mid"
            score_pred_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_score_path, score_pred_path)
        else:
            score_pred_path = source_score_path
        score_item = {
            "score_source": score_source,
            "score_midi": str(source_score_path.resolve()),
            "prediction_paths": [str(score_pred_path.resolve())],
            "ground_truth_paths": gt_paths,
        }

    donor_perfs = donor_pool_for_name(
        args_dict["donor_pool"],
        all_perfs,
        target_perfs,
        args_dict["performance_dataset"],
    )
    if not donor_perfs:
        skipped.append({"score_source": score_source, "reason": "no_donor_performances"})
        return score_item, variant_items, fit_row, skipped, None

    target_arrays = []
    for perf in donor_perfs:
        rows = performance_dev_velocity_pedal4_binary_rows(
            perf,
            score_shared_raw,
            epr_timing_target=config.get("epr_timing_target", "deviation"),
            log_scale=float(config.get("timing_log_scale", 50.0)),
            split_zero_ioi_head=bool(config.get("split_zero_ioi_head", False)),
            ioi_nonzero_dev_scale=float(config.get("ioi_nonzero_dev_scale", 2.0)),
            ioi_zero_dev_scale=float(config.get("ioi_zero_dev_scale", 4.0)),
            pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
        )
        if rows is not None:
            target_arrays.append(np.asarray(rows, dtype=np.float64))
    if not target_arrays:
        skipped.append({"score_source": score_source, "reason": "no_target7_rows"})
        return score_item, variant_items, fit_row, skipped, None
    target_stack = np.stack(target_arrays, axis=0)

    scalar_params_by_note_feature = []
    joint_timing_params_by_note = []
    for note_idx in range(target_stack.shape[1]):
        note_params = []
        for feature_idx in range(target_stack.shape[2]):
            z_values = logit(target_stack[:, note_idx, feature_idx], eps=args_dict["fit_space_eps"])
            params = fit_gmm_1d(
                z_values,
                components=3,
                em_iters=args_dict["em_iters"],
                sigma_min=args_dict["sigma_min"],
                sigma_max=args_dict["sigma_max"],
            )
            note_params.append(params)
        scalar_params_by_note_feature.append(note_params)
        joint_timing_params_by_note.append(
            fit_gmm_nd(
                logit(target_stack[:, note_idx, :2], eps=args_dict["fit_space_eps"]),
                components=3,
                em_iters=args_dict["em_iters"],
                sigma_min=args_dict["sigma_min"],
                sigma_max=args_dict["sigma_max"],
            )
        )

    score_shared = np.asarray(score_shared_raw, dtype=np.float64)
    zero_mask = score_shared[:, 0] <= 0.0
    split_timing_params = {"zero": {}, "nonzero": {}}
    for flag_name, mask in (("zero", zero_mask), ("nonzero", ~zero_mask)):
        if not np.any(mask):
            mask = np.ones(len(score_shared), dtype=bool)
        for feature_idx in (0, 1):
            split_timing_params[flag_name][feature_idx] = fit_gmm_1d(
                logit(target_stack[:, mask, feature_idx].reshape(-1), eps=args_dict["fit_space_eps"]),
                components=3,
                em_iters=args_dict["em_iters"],
                sigma_min=args_dict["sigma_min"],
                sigma_max=args_dict["sigma_max"],
            )
    pedal_probs = np.mean(target_stack[:, :, 3:7], axis=0)

    variant_samples = {}
    for variant in args_dict["variants"]:
        pred_paths = []
        variant_samples[variant] = []
        for sample_idx in range(args_dict["num_samples"]):
            rng = np.random.default_rng(stable_seed(args_dict["seed"], score_source, variant, sample_idx))
            sampled = sample_target_variant(
                variant,
                (len(pitch), target_stack.shape[2]),
                score_shared_raw,
                scalar_params_by_note_feature,
                joint_timing_params_by_note,
                split_timing_params,
                pedal_probs,
                rng,
            )
            raw_rows = _target5_to_raw7(
                torch.tensor(score_shared_raw, dtype=torch.float32),
                torch.tensor(sampled, dtype=torch.float32),
                config=config,
            ).cpu().numpy()
            pred_path = midi_dir / variant / f"{score_stem}__sample_{sample_idx:03d}.mid"
            render_midi(pitch, raw_rows, pred_path, max_time_ms=float(config.get("max_time_ms", 10000.0)))
            pred_paths.append(str(pred_path.resolve()))
            variant_samples[variant].append({"target": sampled, "raw": raw_rows})
        variant_items[variant] = {
            "score_source": score_source,
            "score_midi": str(source_score_path.resolve()),
            "prediction_paths": pred_paths,
            "ground_truth_paths": gt_paths,
        }

    diag_path = output_dir / "duration_diagnostics" / f"{score_stem}.npz"
    write_duration_diag(diag_path, target_stack, score_shared_raw, variant_samples, config)
    fit_row = {
        "score_source": score_source,
        "num_donor_performances": int(len(donor_perfs)),
        "num_gt_performances": int(len(gt_paths)),
        "note_count": int(len(pitch)),
        "zero_score_ioi_notes": int(np.sum(zero_mask)),
        "nonzero_score_ioi_notes": int(np.sum(~zero_mask)),
    }
    return score_item, variant_items, fit_row, skipped, str(diag_path.resolve())


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    score_items = []
    allowed_variants = {
        "independent",
        "joint_timing",
        "pedal_bce",
        "joint_timing_pedal_bce",
        "split_timing",
        "split_timing_pedal_bce",
    }
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    unknown_variants = sorted(set(variants) - allowed_variants)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    variant_items = {variant: [] for variant in variants}
    fit_rows = []
    skipped = []
    diag_paths = []
    args_dict = {
        "output_dir": str(args.output_dir),
        "performance_dataset": args.performance_dataset,
        "donor_pool": args.donor_pool,
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "fit_space_eps": float(args.fit_space_eps),
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        "em_iters": int(args.em_iters),
        "score_copy": bool(args.score_copy),
        "variants": variants,
    }
    tasks = [(item, config, args_dict) for item in manifest]
    if args.num_workers > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            iterator = pool.imap(process_manifest_item, tasks, chunksize=1)
            results = list(tqdm(iterator, total=len(tasks), desc=f"render baselines x{args.num_workers}"))
    else:
        results = [
            process_manifest_item(task)
            for task in tqdm(tasks, desc="render baselines")
        ]

    for score_item, item_variant_items, fit_row, item_skipped, diag_path in results:
        if score_item is not None:
            score_items.append(score_item)
        for variant, variant_item in item_variant_items.items():
            if variant_item is not None:
                variant_items.setdefault(variant, []).append(variant_item)
        if fit_row is not None:
            fit_rows.append(fit_row)
        skipped.extend(item_skipped)
        if diag_path is not None:
            diag_paths.append(diag_path)

    score_manifest_path = args.output_dir / "score_midi_prediction_manifest.json"
    score_payload = make_manifest(score_items, "score_midi", 1, score_manifest_path)
    score_eval = evaluate_manifest(
        score_payload,
        score_manifest_path,
        args.output_dir / "score_midi_eval.json",
    )

    variant_evals = {}
    for variant in variants:
        variant_manifest_path = args.output_dir / f"{variant}_prediction_manifest.json"
        variant_payload = make_manifest(variant_items[variant], variant, args.num_samples, variant_manifest_path)
        variant_eval = evaluate_manifest(
            variant_payload,
            variant_manifest_path,
            args.output_dir / f"{variant}_eval.json",
        )
        variant_evals[variant] = {
            "eval": str((args.output_dir / f"{variant}_eval.json").resolve()),
            "manifest": str(variant_manifest_path.resolve()),
            "num_scores": int(len(variant_items[variant])),
            "pp_wass": variant_eval["aggregate"]["pp_wass"],
            "pn_wass": variant_eval["aggregate"]["pn_wass"],
        }

    plot_outputs = plot_duration_diagnostics(
        [Path(path) for path in diag_paths],
        args.output_dir / "plots",
        variants,
        max_points=int(args.max_plot_points),
        seed=int(args.seed),
    )

    fit_csv = args.output_dir / "mln3_fit_summary.csv"
    pd.DataFrame(fit_rows).to_csv(fit_csv, index=False)
    summary = {
        "config": str(args.config),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "donor_pool": args.donor_pool,
        "num_manifest_scores": int(len(manifest)),
        "num_score_baseline_scores": int(len(score_items)),
        "variants": variants,
        "num_samples": int(args.num_samples),
        "score_midi_eval": str((args.output_dir / "score_midi_eval.json").resolve()),
        "score_midi_pp_wass": score_eval["aggregate"]["pp_wass"],
        "variant_evals": variant_evals,
        "duration_plots": plot_outputs,
        "midi_dir": str((args.output_dir / "midis").resolve()),
        "fit_csv": str(fit_csv.resolve()),
        "duration_diag_paths": diag_paths,
        "skipped": skipped,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
