#!/usr/bin/env python
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.infer_inr_testset import (  # noqa: E402
    filter_manifest_by_performance_dataset,
    load_config,
    load_score_from_node,
    select_device,
)
from src.model.integrated_pianoformer import (  # noqa: E402
    _build_epr_decoder_rows,
    _build_prefilled_ar_note_inputs,
    _decode_mixture_value,
    _logistic_normal_params,
    _materialize_epr_prediction,
    _shift_pitch_right,
    _split_epr_mixture_params,
    _uses_deviation_ratio_targets,
)
from src.train.train_inr import build_work_manifest, create_model  # noqa: E402
from src.train.train_inr import integrated_epr_input_dim  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot real autoregressive MLN timing distributions from INR checkpoints."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-notes", type=int, default=192)
    parser.add_argument("--examples-per-group", type=int, default=6)
    parser.add_argument("--max-manifest-scan", type=int, default=30)
    parser.add_argument("--score-index", type=int, default=None)
    return parser.parse_args()


def load_model(config, checkpoint, device):
    config = dict(config)
    if not config.get("musical_feature_mode") and str(config.get("task_type", "epr")).lower() == "epr":
        input_dim = config.get("input_continuous_dim")
        if input_dim == integrated_epr_input_dim(
            timing_control_mode=config.get("timing_control_mode"),
            use_timing_scale_bit=config.get("use_timing_scale_bit", True),
            musical_feature_mode="continuous",
        ):
            config["musical_feature_mode"] = "continuous"
        elif input_dim == integrated_epr_input_dim(
            timing_control_mode=config.get("timing_control_mode"),
            use_timing_scale_bit=config.get("use_timing_scale_bit", True),
            musical_feature_mode="categorical",
        ):
            config["musical_feature_mode"] = "categorical"
    config["resume_path"] = checkpoint
    config["reset_output_heads_on_resume"] = False
    config["ignore_mismatched_resume_shapes"] = False
    model = create_model(config)
    model.to(device)
    model.eval()
    return model


def capture_t5_ar_raw(model, pitch, continuous, score_shared_raw, device):
    pitch_ids = torch.tensor(pitch, dtype=torch.long, device=device).unsqueeze(0)
    continuous_tensor = torch.tensor(continuous, dtype=torch.float32, device=device).unsqueeze(0)
    score_shared_raw_tensor = torch.tensor(score_shared_raw, dtype=torch.float32, device=device).unsqueeze(0)
    attention_mask = (pitch_ids != model.config.pitch_pad_id).long()

    if str(model.config.decoder_input_mode).lower() != "ar":
        raise ValueError("This diagnostic expects decoder_input_mode=ar")
    if _uses_deviation_ratio_targets(model.config) is False:
        raise ValueError("This diagnostic currently expects log_deviation/deviation-ratio EPR targets")
    if not hasattr(model, "model") or not hasattr(model.model, "encoder"):
        raise ValueError("This diagnostic currently implements the T5 autoregressive path")

    score_note_embeds = model.note_encoder(pitch_ids, continuous_tensor)
    encoder_outputs = model.model.encoder(
        attention_mask=attention_mask,
        inputs_embeds=score_note_embeds,
    )
    decoder_input_continuous, special_note_ids, prefix_len = _build_prefilled_ar_note_inputs(
        model.config,
        attention_mask,
        model.config.output_continuous_dim,
        prefix_predictions=None,
        score_shared_raw=score_shared_raw_tensor,
    )
    if prefix_len:
        raise ValueError("Prefix predictions are not used in this diagnostic")

    decoder_pitch_ids = _shift_pitch_right(model.config, pitch_ids, attention_mask)
    cached_past_key_values = None
    raw_steps = []
    pred_steps = []

    with torch.no_grad():
        for step in range(pitch_ids.shape[1]):
            if step == 0:
                decoder_inputs_embeds = model.decoder_note_encoder(
                    decoder_pitch_ids[:, :1],
                    decoder_input_continuous[:, :1],
                    special_note_ids=special_note_ids[:, :1],
                )
                decoder_attention_mask = attention_mask[:, :1]
                decoder_outputs = model.model(
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=True,
                    past_key_values=cached_past_key_values,
                )
            else:
                decoder_inputs_embeds = model.decoder_note_encoder(
                    decoder_pitch_ids[:, step : step + 1],
                    decoder_input_continuous[:, step : step + 1],
                    special_note_ids=special_note_ids[:, step : step + 1],
                )
                decoder_attention_mask = attention_mask[:, : step + 1]
                cache_position = torch.tensor([step], device=device, dtype=torch.long)
                decoder_outputs = model.model(
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=True,
                    past_key_values=cached_past_key_values,
                    cache_position=cache_position,
                )
            cached_past_key_values = decoder_outputs.past_key_values

            step_raw = model.continuous_decoder(decoder_outputs.last_hidden_state[:, -1:, :])
            step_pred = _materialize_epr_prediction(
                model.config,
                step_raw,
                sampling_strategy="greedy",
                score_shared_raw=score_shared_raw_tensor[:, step : step + 1],
            )
            raw_steps.append(step_raw.detach().cpu())
            pred_steps.append(step_pred.detach().cpu())

            if step + 1 < pitch_ids.shape[1]:
                decoder_input_continuous[:, step + 1] = _build_epr_decoder_rows(
                    model.config,
                    score_shared_raw_tensor[:, step : step + 1],
                    step_pred,
                )[:, 0]

    return torch.cat(raw_steps, dim=1)[0], torch.cat(pred_steps, dim=1)[0]


def logistic_normal_pdf_components(config, logits, raw_mu, raw_log_sigma, x):
    with torch.no_grad():
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        raw_mu_t = torch.as_tensor(raw_mu, dtype=torch.float32)
        raw_log_sigma_t = torch.as_tensor(raw_log_sigma, dtype=torch.float32)
        x_t = torch.as_tensor(x, dtype=torch.float32).clamp(
            float(getattr(config, "epr_distribution_eps", 1e-5)),
            1.0 - float(getattr(config, "epr_distribution_eps", 1e-5)),
        )
        mu, sigma = _logistic_normal_params(
            raw_mu_t,
            raw_log_sigma_t,
            sigma_min=getattr(config, "logistic_normal_sigma_min", 1e-3),
            sigma_max=getattr(config, "logistic_normal_sigma_max", 10.0),
        )
        z = torch.logit(x_t).unsqueeze(-1)
        weights = torch.softmax(logits_t, dim=-1)
        normal = torch.exp(torch.distributions.Normal(mu, sigma).log_prob(z))
        jacobian = 1.0 / (x_t * (1.0 - x_t)).unsqueeze(-1)
        comp = weights.unsqueeze(0) * normal * jacobian
        total = comp.sum(dim=-1)
    return total.numpy(), comp.numpy(), weights.numpy(), torch.sigmoid(mu).numpy(), sigma.numpy()


def selected_timing_params(config, raw_row, score_ioi):
    params = _split_epr_mixture_params(config, raw_row.reshape(1, 1, -1))
    is_zero = float(score_ioi) <= 0.0
    if is_zero and bool(getattr(config, "split_zero_ioi_head", False)):
        ioi = (
            params["ioi_zero_logits"][0, 0],
            params["ioi_zero_a"][0, 0],
            params["ioi_zero_b"][0, 0],
            "ioi_zero_head",
        )
    else:
        ioi = (
            params["shared_logits"][0, 0, 0],
            params["shared_a"][0, 0, 0],
            params["shared_b"][0, 0, 0],
            "ioi_shared_head",
        )
    duration = (
        params["shared_logits"][0, 0, 1],
        params["shared_a"][0, 0, 1],
        params["shared_b"][0, 0, 1],
        "duration_shared_head",
    )
    return ioi, duration


def choose_score(config, args):
    config = dict(config)
    if not config.get("musical_feature_mode") and str(config.get("task_type", "epr")).lower() == "epr":
        input_dim = config.get("input_continuous_dim")
        if input_dim == integrated_epr_input_dim(
            timing_control_mode=config.get("timing_control_mode"),
            use_timing_scale_bit=config.get("use_timing_scale_bit", True),
            musical_feature_mode="continuous",
        ):
            config["musical_feature_mode"] = "continuous"
        elif input_dim == integrated_epr_input_dim(
            timing_control_mode=config.get("timing_control_mode"),
            use_timing_scale_bit=config.get("use_timing_scale_bit", True),
            musical_feature_mode="categorical",
        ):
            config["musical_feature_mode"] = "categorical"
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=None,
        skip_work_paths=config.get("skip_work_paths"),
    )
    if args.performance_dataset:
        manifest = filter_manifest_by_performance_dataset(
            manifest,
            metadata_path=config["metadata_path"],
            split=args.split,
            performance_dataset=args.performance_dataset,
            exclude_performance_dataset=None,
        )
    if args.score_index is not None:
        candidates = [manifest[args.score_index]]
    else:
        candidates = manifest[: args.max_manifest_scan]

    musical_feature_mode = str(
        config.get("musical_feature_mode", "continuous" if config.get("task_type") == "csr" else "categorical")
    ).lower()
    for item in candidates:
        pitch, continuous, score_shared_raw, work = load_score_from_node(
            Path(item["path"]),
            use_timing_scale_bit=config.get("use_timing_scale_bit", True),
            timing_control_mode=config.get("timing_control_mode"),
            timing_log_scale=config.get("timing_log_scale", 50.0),
            musical_feature_mode=musical_feature_mode,
            task_type=config.get("task_type", "epr"),
        )
        limit = min(len(pitch), int(args.max_notes))
        zero_count = sum(1 for row in score_shared_raw[:limit] if float(row[0]) <= 0.0)
        nonzero_count = limit - zero_count
        if zero_count >= args.examples_per_group and nonzero_count >= args.examples_per_group:
            return item, pitch[:limit], continuous[:limit], score_shared_raw[:limit], work
    raise RuntimeError("Could not find a score segment with enough zero and nonzero IOI examples")


def pick_example_indices(score_shared_raw, per_group, seed):
    rng = random.Random(seed)
    zero = [idx for idx, row in enumerate(score_shared_raw) if float(row[0]) <= 0.0]
    nonzero = [idx for idx, row in enumerate(score_shared_raw) if float(row[0]) > 0.0]
    rng.shuffle(zero)
    rng.shuffle(nonzero)
    selected = sorted([(idx, "zero") for idx in zero[:per_group]] + [(idx, "nonzero") for idx in nonzero[:per_group]])
    return selected


def plot_examples(config, raw_outputs, predictions, score_shared_raw, selected, output_dir, title_prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(1e-4, 1.0 - 1e-4, 1200)
    rows = []
    fig, axes = plt.subplots(len(selected), 2, figsize=(11, max(2.0 * len(selected), 10)), sharex=True)
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_idx, (note_idx, group) in enumerate(selected):
        score_ioi = float(score_shared_raw[note_idx][0])
        score_dur = float(score_shared_raw[note_idx][1])
        raw_row = raw_outputs[note_idx]
        pred_row = predictions[note_idx]
        for col_idx, (feature_name, bundle, pred_value) in enumerate(
            [
                ("ioi_dev", selected_timing_params(config, raw_row, score_ioi)[0], float(pred_row[0])),
                ("duration_dev", selected_timing_params(config, raw_row, score_ioi)[1], float(pred_row[1])),
            ]
        ):
            logits, raw_mu, raw_log_sigma, head_name = bundle
            total, comp, weights, means, sigmas = logistic_normal_pdf_components(
                config,
                logits,
                raw_mu,
                raw_log_sigma,
                x,
            )
            ax = axes[row_idx, col_idx]
            ax.plot(x, total, color="#1f2937", linewidth=1.7, label="mixture")
            colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
            for comp_idx in range(comp.shape[1]):
                ax.plot(
                    x,
                    comp[:, comp_idx],
                    color=colors[comp_idx % len(colors)],
                    linewidth=1.0,
                    alpha=0.75,
                    linestyle="--",
                    label=f"c{comp_idx}: w={weights[comp_idx]:.2f}",
                )
            ax.axvline(pred_value, color="#111827", linestyle=":", linewidth=1.0)
            ax.set_title(
                f"note {note_idx} {group} {feature_name} {head_name}",
                fontsize=9,
            )
            ax.set_ylabel("pdf")
            ax.grid(alpha=0.18)
            if row_idx == 0:
                ax.legend(fontsize=7, loc="upper right")
            effective_components = float(np.exp(-(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum()))
            rows.append(
                {
                    "note_idx": note_idx,
                    "group": group,
                    "feature": feature_name,
                    "head": head_name,
                    "score_ioi_ms": score_ioi,
                    "score_duration_ms": score_dur,
                    "pred_norm": pred_value,
                    "weight_entropy_effective_components": effective_components,
                    "weight_max": float(weights.max()),
                    "sigmoid_mu_min": float(means.min()),
                    "sigmoid_mu_max": float(means.max()),
                    "sigmoid_mu_range": float(means.max() - means.min()),
                    "sigma_min": float(sigmas.min()),
                    "sigma_max": float(sigmas.max()),
                    "weights": ";".join(f"{v:.6g}" for v in weights),
                    "sigmoid_mus": ";".join(f"{v:.6g}" for v in means),
                    "sigmas": ";".join(f"{v:.6g}" for v in sigmas),
                }
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("normalized target value")
    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    plot_path = output_dir / "mln3_real_prediction_timing_pdfs.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    csv_path = output_dir / "mln3_real_prediction_timing_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return plot_path, csv_path


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    config = load_config(args.config, args.checkpoint)
    model = load_model(config, args.checkpoint, device)

    item, pitch, continuous, score_shared_raw, _ = choose_score(config, args)
    selected = pick_example_indices(score_shared_raw, args.examples_per_group, args.seed)
    raw_outputs, predictions = capture_t5_ar_raw(model, pitch, continuous, score_shared_raw, device)

    run_name = args.output_dir.name
    title = f"{run_name} | {item['score_source']} | first {len(pitch)} notes"
    plot_path, csv_path = plot_examples(
        model.config,
        raw_outputs,
        predictions,
        score_shared_raw,
        selected,
        args.output_dir,
        title,
    )
    meta = {
        "config": str(args.config),
        "checkpoint": args.checkpoint,
        "score_source": item["score_source"],
        "score_json": item["path"],
        "note_count_used": len(pitch),
        "selected": [{"note_idx": idx, "group": group} for idx, group in selected],
    }
    meta_path = args.output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {csv_path}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
