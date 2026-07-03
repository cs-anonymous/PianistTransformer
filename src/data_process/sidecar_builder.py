def build_sidecar_for_work(dataset, path, selected_sources=None):
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
            "label_pedal2_raw",
            "label_pedal4_raw",
            "label_raw",
            "pedal2_raw",
            "pedal4_raw",
        ):
            if key in perf:
                raw_perf[key] = perf[key]
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
    dataset._save_prepared_to_disk(path, prepared)
    return dataset._prepared_disk_cache_path(path)
