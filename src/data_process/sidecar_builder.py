import math


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
    pitch = score_payload.get("pitch") or []
    if "score_feature" not in score_payload:
        score_payload["score_feature"] = [[0.0] * 9 for _ in pitch]
    if "has_score_feature" not in score_payload:
        score_payload["has_score_feature"] = [0] * len(pitch)
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
