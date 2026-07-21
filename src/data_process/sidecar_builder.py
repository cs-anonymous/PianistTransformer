import math
import os
import copy
from pathlib import Path
import torch


def _validate_score_feature_payload(score_payload, path):
    pitch = score_payload.get("pitch") or []
    score_feature = score_payload.get("score_feature")
    has_score_feature = score_payload.get("has_score_feature")
    if not isinstance(score_feature, list):
        raise ValueError(f"missing_score_feature_for_sidecar: {path}")
    if not isinstance(has_score_feature, list):
        raise ValueError(f"missing_has_score_feature_for_sidecar: {path}")
    if len(score_feature) != len(pitch):
        raise ValueError(f"score_feature_length_mismatch_for_sidecar: {path}")
    if len(has_score_feature) != len(pitch):
        raise ValueError(f"has_score_feature_length_mismatch_for_sidecar: {path}")
    for idx, (row, has_feature) in enumerate(zip(score_feature, has_score_feature)):
        if not bool(has_feature):
            continue
        if len(row) != 9:
            raise ValueError(f"score_feature_width_mismatch_for_sidecar[{idx}]: {path}")
        mo_idx = int(round(float(row[0])))
        md_idx = int(round(float(row[1])))
        ml_idx = int(round(float(row[2])))
        if mo_idx < 0 or mo_idx > 144 or md_idx < 0 or md_idx > 144 or ml_idx < 0 or ml_idx > 144:
            raise ValueError(f"score_feature_idx_out_of_range[{idx}]: {path}")
        for value in row[3:9]:
            if float(value) not in (0.0, 1.0):
                raise ValueError(f"score_feature_binary_value_invalid[{idx}]: {path}")
    matched = sum(1 for value in has_score_feature if bool(value))
    if pitch and matched <= 0:
        raise ValueError(f"zero_score_feature_coverage_for_sidecar: {path}")
    total_abs = 0.0
    for row in score_feature:
        total_abs += sum(abs(float(value)) for value in row)
    if pitch and total_abs <= 0.0:
        raise ValueError(f"all_zero_score_feature_for_sidecar: {path}")


def _normalize_performance_to_score_onset_span(score_raw, perf):
    """Scale performance IOI/duration so its first-to-last onset span matches the score."""
    score_span_ms = float(sum(float(row[0]) for row in score_raw[1:]))
    shared = perf.get("label_shared_raw") or perf.get("label_raw")
    if not shared:
        raise ValueError("Performance has no raw timing labels")
    perf_span_ms = float(sum(float(row[0]) for row in shared[1:]))
    if not math.isfinite(score_span_ms) or score_span_ms <= 0.0:
        raise ValueError(f"Invalid score onset span: {score_span_ms}")
    if not math.isfinite(perf_span_ms) or perf_span_ms <= 0.0:
        raise ValueError(f"Invalid performance onset span: {perf_span_ms}")

    scale = score_span_ms / perf_span_ms
    normalized = []
    for row in shared:
        new_row = list(row)
        new_row[0] = float(new_row[0]) * scale
        new_row[1] = float(new_row[1]) * scale
        normalized.append(new_row)
    perf["label_shared_raw"] = normalized
    perf.pop("label_raw", None)
    perf["global_timing_scale"] = {
        "method": "score_onset_span",
        "scale": scale,
        "log_scale": math.log(scale),
        "score_onset_span_ms": score_span_ms,
        "performance_onset_span_ms": perf_span_ms,
        "normalized_onset_span_ms": perf_span_ms * scale,
    }


def build_sidecar_for_work(
    dataset,
    path,
    selected_sources=None,
    performance_time_normalization=None,
):
    work = dataset._load_work(path)
    if selected_sources is not None:
        selected_sources = set(selected_sources)
        work = dict(work)
        work["performances"] = [
            perf for perf in work.get("performances", [])
            if perf.get("performance_source") in selected_sources
        ]

    prepared = dataset._prepare_work(
        path,
        work,
        eager_labels=False,
        slim_performances=True,
        split_filter=False,
        force_rebuild=True,
        derive_features=False,
    )
    score_payload = prepared.get("score") or {}
    if "score_feature" not in score_payload or "has_score_feature" not in score_payload:
        full_prepared = dataset._prepare_work(
            path,
            work,
            eager_labels=False,
            slim_performances=False,
            split_filter=False,
            force_rebuild=True,
            derive_features=False,
        )
        full_score = full_prepared.get("score") or {}
        if "score_feature" in full_score:
            score_payload["score_feature"] = full_score["score_feature"]
        if "has_score_feature" in full_score:
            score_payload["has_score_feature"] = full_score["has_score_feature"]
    _validate_score_feature_payload(score_payload, path)
    # Persist raw per-performance targets inside the sidecar so training can
    # stay strictly read-only and validate the cache payload against the current
    # raw-sidecar schema.
    raw_performances = []
    for perf in prepared.get("performances", []):
        raw_perf = {
            "performance_source": perf.get("performance_source"),
            "performance_id": perf.get("performance_id", "unknown"),
            "performance_dataset": perf.get("performance_dataset", "unknown"),
            "split": perf.get("split", dataset.split),
            "interpolated": perf["interpolated"],
        }
        for key in (
            "label_shared_raw",
            "label_pedal4_raw",
            "label_raw",
            "pedal4_raw",
        ):
            if key in perf:
                raw_perf[key] = perf[key]
        if performance_time_normalization == "score_onset_span":
            _normalize_performance_to_score_onset_span(score_payload["score_raw"], raw_perf)
        elif performance_time_normalization not in (None, "none"):
            raise ValueError(
                f"Unsupported performance_time_normalization={performance_time_normalization}"
            )
        raw_performances.append(raw_perf)
    prepared["performances"] = raw_performances
    prepared["performances_by_source"] = {
        perf.get("performance_source"): perf
        for perf in raw_performances
        if perf.get("performance_source") is not None
    }
    prepared.pop("score_input", None)
    prepared.pop("score_musical", None)
    prepared.pop("has_score_feature", None)
    prepared["label_cache"] = {}
    if isinstance(work.get("meta"), dict):
        prepared["meta"] = dict(work["meta"])
    prepared["performance_time_normalization"] = performance_time_normalization or "none"
    dataset._save_prepared_to_disk(path, prepared)
    return dataset._prepared_disk_cache_path(path)


def build_ready_sidecar_for_work(
    dataset,
    path,
    selected_sources=None,
    performance_time_normalization=None,
):
    work = dataset._load_work(path)
    # Ready sidecars normally derive labels directly from the source payload.
    # Apply the same optional normalization as raw sidecars before eager label
    # construction so a raw score-span cache can be upgraded in place without
    # changing its target semantics.
    if performance_time_normalization not in (None, "none"):
        work = copy.deepcopy(work)
        if performance_time_normalization == "score_onset_span":
            score_raw = (work.get("score") or {}).get("score_raw")
            if not score_raw:
                raise ValueError("Work has no score_raw for score_onset_span normalization")
            for perf in work.get("performances", []):
                _normalize_performance_to_score_onset_span(score_raw, perf)
        else:
            raise ValueError(
                f"Unsupported performance_time_normalization={performance_time_normalization}"
            )
    if selected_sources is not None:
        selected_sources = set(selected_sources)
        work = dict(work)
        work["performances"] = [
            perf for perf in work.get("performances", [])
            if perf.get("performance_source") in selected_sources
        ]
    prepared = dataset._prepare_work(
        path,
        work,
        eager_labels=True,
        slim_performances=True,
        split_filter=False,
        force_rebuild=True,
        derive_features=True,
    )
    score_payload = prepared.get("score") or {}
    if "score_feature" not in score_payload:
        source = Path(path)
        for annotation_path in (
            source.with_suffix(".ASAP.pt"),
            source.with_suffix(".pt"),
        ):
            if not annotation_path.exists():
                continue
            annotation_payload = torch.load(
                annotation_path,
                map_location="cpu",
                weights_only=False,
            )
            annotation_score = annotation_payload.get("score") or {}
            if annotation_score.get("pitch") != score_payload.get("pitch"):
                continue
            if "score_feature" not in annotation_score:
                continue
            score_payload["score_feature"] = annotation_score["score_feature"]
            score_payload["has_score_feature"] = annotation_score.get(
                "has_score_feature",
                [1] * len(score_payload.get("pitch", [])),
            )
            break
    for perf in prepared.get("performances", []):
        perf["labels_by_target"] = {
            "floor_log_deviation": perf.pop("labels"),
        }
        perf.pop("label_bins", None)
    prepared["_cache_signature"] = dataset._build_ready_sidecar_signature()
    prepared["_source_identity"] = dataset._source_identity(path)
    prepared["performance_time_normalization"] = performance_time_normalization or "none"
    cache_path = dataset._prepared_disk_cache_path(path)
    if cache_path is None:
        cache_path = dataset._prepared_sidecar_paths(path)[0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    torch.save(prepared, tmp_path)
    os.replace(tmp_path, cache_path)
    return Path(cache_path)
