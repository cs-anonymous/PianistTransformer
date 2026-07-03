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
    prepared.pop("score_input", None)
    prepared.pop("score_musical", None)
    prepared.pop("has_score_feature", None)
    prepared["label_cache"] = {}
    if isinstance(work.get("meta"), dict):
        prepared["meta"] = dict(work["meta"])
    dataset._save_prepared_to_disk(path, prepared)
    return dataset._prepared_disk_cache_path(path)
