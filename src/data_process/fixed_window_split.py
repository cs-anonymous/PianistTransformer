import json
from pathlib import Path


_FIXED_WINDOW_SPLIT_INDEX_CACHE = {}


def _normalized_work_path(path):
    return str(Path(path).expanduser().resolve())


def _processed_relative_path(path):
    parts = Path(path).parts
    try:
        processed_index = parts.index("processed")
    except ValueError:
        return None
    return "/".join(parts[processed_index + 1 :])


def _load_fixed_window_split_index(summary_path, scheme_name):
    cache_key = (str(summary_path), str(scheme_name))
    if cache_key in _FIXED_WINDOW_SPLIT_INDEX_CACHE:
        return _FIXED_WINDOW_SPLIT_INDEX_CACHE[cache_key]

    with open(summary_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("scheme_name") != str(scheme_name):
        raise ValueError(
            f"Fixed window split summary scheme mismatch: expected {scheme_name}, "
            f"got {payload.get('scheme_name')} from {summary_path}"
        )

    index = {}
    for row in payload.get("work_summaries", []):
        work_path = row.get("path")
        valid_window = row.get("valid_window")
        train_window_count = int(row.get("train_window_count", 0) or 0)
        valid_window_count = int(row.get("valid_window_count", 0) or 0)
        entry = {
            "valid_window": valid_window,
            "train_window_count": train_window_count,
            "valid_window_count": valid_window_count,
        }
        index[str(work_path)] = entry
        index[_normalized_work_path(work_path)] = entry
        relative_path = _processed_relative_path(work_path)
        if relative_path:
            index[("processed-relative", relative_path)] = entry
    _FIXED_WINDOW_SPLIT_INDEX_CACHE[cache_key] = index
    return index


def load_windows_from_fixed_split(path, scheme_name, split_name, canonical_windows=None, summary_path=None):
    if summary_path:
        index = _load_fixed_window_split_index(summary_path, scheme_name)
        entry = index.get(str(path)) or index.get(_normalized_work_path(path))
        if entry is None:
            relative_path = _processed_relative_path(path)
            if relative_path:
                entry = index.get(("processed-relative", relative_path))
        if entry is None:
            raise KeyError(f"Missing work entry for fixed split scheme={scheme_name} in summary {summary_path}: {path}")
        if canonical_windows is None:
            raise ValueError(
                f"canonical_windows is required when using fixed split summary {summary_path} for {path}"
            )
        canonical_windows = [(int(window[0]), int(window[1])) for window in canonical_windows]
        valid_window = entry.get("valid_window")
        valid_window = (
            (int(valid_window[0]), int(valid_window[1]))
            if isinstance(valid_window, (list, tuple)) and len(valid_window) == 2
            else None
        )
        if str(split_name) == "valid":
            if valid_window is None:
                return []
            return [valid_window]
        if str(split_name) == "train":
            if valid_window is None:
                return canonical_windows
            return [window for window in canonical_windows if window != valid_window]
        raise ValueError(
            f"Unsupported fixed window split name={split_name} with summary-backed lookup for {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    meta = payload.get("meta") or {}
    schemes = meta.get("window_split_schemes") or {}
    scheme = schemes.get(str(scheme_name))
    if not isinstance(scheme, dict):
        raise KeyError(f"Missing window split scheme={scheme_name} in {path}")

    assignments = scheme.get("window_assignments") or []
    windows = []
    for entry in assignments:
        if str(entry.get("split")) != str(split_name):
            continue
        window = entry.get("window")
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            continue
        windows.append((int(window[0]), int(window[1])))

    if not windows:
        available = sorted({str(entry.get("split")) for entry in assignments if entry.get("split") is not None})
        raise ValueError(
            f"No windows for split={split_name} under scheme={scheme_name} in {path}. "
            f"Available splits: {available}"
        )

    deduped = []
    seen = set()
    for window in windows:
        if window in seen:
            continue
        seen.add(window)
        deduped.append(window)
    return deduped
