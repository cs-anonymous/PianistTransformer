import random
from pathlib import Path

import pandas as pd
import torch

from src.data_process.fixed_window_split import load_windows_from_fixed_split


def score_json_path(refined_dir, score_rel_path):
    score_path = Path(refined_dir) / score_rel_path
    candidates = [
        score_path.with_suffix(".json"),
        score_path.parent / f"{score_path.stem}.node_a.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def make_windows(total_notes, block_notes, overlap_ratio, min_notes):
    total_notes = int(total_notes)
    if total_notes < min_notes:
        return []
    if total_notes <= block_notes:
        return [(0, total_notes)]

    stride = max(1, int(block_notes * (1.0 - overlap_ratio)))
    windows = []
    start = 0
    while start + block_notes <= total_notes:
        windows.append((start, start + block_notes))
        start += stride
    if windows[-1][1] != total_notes and total_notes - start >= min_notes:
        windows.append((total_notes - block_notes, total_notes))

    deduped = []
    seen = set()
    for window in windows:
        if window in seen:
            continue
        deduped.append(window)
        seen.add(window)
    return deduped


def work_token_count(path, metadata_note_count, prepared_sidecar_tag=None):
    try:
        metadata_note_count = int(float(metadata_note_count))
    except (TypeError, ValueError, OverflowError):
        metadata_note_count = 0
    if metadata_note_count > 0:
        return metadata_note_count

    source = Path(path)
    sidecar_paths = []
    if prepared_sidecar_tag:
        sidecar_paths.append(source.with_suffix(f".{prepared_sidecar_tag}.pt"))
    sidecar_paths.append(source.with_suffix(".pt"))

    seen = set()
    for sidecar_path in sidecar_paths:
        sidecar_path = Path(sidecar_path)
        if str(sidecar_path) in seen or not sidecar_path.exists():
            continue
        seen.add(str(sidecar_path))
        try:
            payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(sidecar_path, map_location="cpu")
        except Exception:
            payload = None
        if isinstance(payload, dict):
            score = payload.get("score")
            if isinstance(score, dict):
                for key in ("pitch", "score_raw"):
                    value = score.get(key)
                    if value is not None:
                        return int(len(value))

    try:
        import json

        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        score = payload.get("score") if isinstance(payload, dict) else None
        if isinstance(score, dict):
            for key in ("pitch", "score_raw"):
                value = score.get(key)
                if value is not None:
                    return int(len(value))
    except Exception:
        pass
    return int(metadata_note_count)


def build_work_manifest(
    metadata_path,
    refined_dir,
    split,
    block_notes,
    overlap_ratio,
    min_notes,
    max_works=None,
    include_all_performance_dataset=None,
    max_non_asap_performances_per_work=None,
    selection_seed=42,
    skip_work_paths=None,
    performance_dataset=None,
    exclude_performance_dataset=None,
    window_split_scheme=None,
    window_split_name=None,
    window_split_summary_path=None,
    prepared_sidecar_tag=None,
):
    columns = [
        "tier_a",
        "split",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
        "refined_score_note_count",
        "performance_dataset",
    ]
    df = pd.read_csv(metadata_path, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    dataset = df["performance_dataset"].fillna("").astype(str)
    if performance_dataset is not None:
        df = df[dataset == str(performance_dataset)]
        dataset = df["performance_dataset"].fillna("").astype(str)
    if exclude_performance_dataset is not None:
        df = df[dataset != str(exclude_performance_dataset)]
    df = df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")

    manifest = []
    skip_work_paths = set(skip_work_paths or [])
    for score_rel_path, group in df.groupby("refined_score_midi_path", sort=True):
        selected_group = group
        if include_all_performance_dataset is not None and max_non_asap_performances_per_work is not None:
            dataset = group["performance_dataset"].fillna("").astype(str)
            always_mask = dataset == str(include_all_performance_dataset)
            always = group[always_mask]
            other = group[~always_mask]
            if len(other) > max_non_asap_performances_per_work:
                rng = random.Random(f"{selection_seed}:{score_rel_path}")
                sampled_indices = rng.sample(list(other.index), max_non_asap_performances_per_work)
                other = other.loc[sampled_indices]
            selected_group = pd.concat([always, other], axis=0).sort_values(
                ["refined_performance_midi_path"],
                kind="stable",
            )

        path = score_json_path(refined_dir, score_rel_path)
        if not path.exists():
            continue
        if str(path) in skip_work_paths or score_rel_path in skip_work_paths:
            print(f"Skipping configured work JSON: {path}", flush=True)
            continue
        note_count = work_token_count(
            path,
            group["refined_score_note_count"].iloc[0],
            prepared_sidecar_tag=prepared_sidecar_tag,
        )
        windows = make_windows(note_count, block_notes, overlap_ratio, min_notes)
        if window_split_scheme is not None and window_split_name is not None:
            windows = load_windows_from_fixed_split(
                path,
                scheme_name=window_split_scheme,
                split_name=window_split_name,
                canonical_windows=windows,
                summary_path=window_split_summary_path,
            )
        if not windows:
            continue
        selected_sources = selected_group["refined_performance_midi_path"].tolist()
        manifest.append(
            {
                "path": str(path),
                "score_source": score_rel_path,
                "note_count": note_count,
                "windows": windows,
                "selected_performance_sources": selected_sources,
                "performance_dataset_counts": {
                    str(key): int(value)
                    for key, value in selected_group["performance_dataset"]
                    .fillna("unknown")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "estimated_performances": int(len(selected_sources)),
                "estimated_examples": int(len(windows) * len(selected_sources)),
            }
        )
    if max_works is not None:
        manifest = manifest[:max_works]
    return manifest
