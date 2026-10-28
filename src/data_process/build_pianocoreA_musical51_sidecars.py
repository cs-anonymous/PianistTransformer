#!/usr/bin/env python3
"""Build fresh musical51 raw sidecars for every PianoCoRe-A training work.

Read the current raw-label JSON and reproject features from source MusicXML.
The JSON and existing sidecars remain untouched. If XML projection fails,
legacy annotations are accepted only after exact score-identity validation.
Missing annotations retain their explicit masks and are audited per work.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.data_process.build_asap_musical51_sidecars import _convert_row
from src.data_process.update_json_score_feature_with_xml import build_score_features, init_worker


def initialize(raw_root):
    torch.set_num_threads(1)
    init_worker(str(raw_root), {
        "timeout_sec": 180., "max_sequence_matcher_notes": 13000,
        "disable_sequence_matcher": False,
    })


def build_one(task):
    from src.data_process import update_json_score_feature_with_xml as feature_module
    row, root, tag = task
    path = (Path(root) / row["refined_score_midi_path"]).with_suffix(".json")
    target = path.with_suffix(f".{tag}.pt")
    stat = path.stat()
    identity = {"path": str(path.resolve()), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    if target.exists():
        cached = torch.load(target, map_location="cpu", weights_only=False)
        if (cached.get("_source_identity") == identity and
                cached.get("meta", {}).get("preparation_protocol") == "fresh_xml_musical51_v1"):
            return cached["meta"]["preparation_audit"] | {"reused": True}
    work = json.loads(path.read_text())
    score = work["score"]
    pitch = [int(p) for p in score["pitch"]]
    legacy = None
    def validated_legacy():
        nonlocal legacy
        if legacy is None:
            legacy = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
            old_score = legacy.get("score", {})
            if old_score.get("pitch") != score["pitch"] or old_score.get("score_raw") != score["score_raw"]:
                raise ValueError(f"Legacy annotation score does not match current JSON: {path}")
        return legacy
    feature_source = "fresh_musicxml_projection"
    xml_error = None
    try:
        if row.get("_xml_preparation_skip_reason"):
            reason = row["_xml_preparation_skip_reason"]
            if reason.startswith("XMLPreparationTimeout:"):
                raise TimeoutError(reason)
            raise ValueError(f"Recorded fresh XML preparation failure: {reason}")
        features = build_score_features(row["score_xml_path"], row["score_midi_path"], pitch, feature_module.WORKER_ARGS)
        if not sum(features["has_score_feature"]):
            raise ValueError("No MusicXML notes aligned")
        musical_features = [_convert_row(r) if h else [0.] * 9
                            for r, h in zip(features["score_feature"], features["has_score_feature"])]
    except Exception as exc:
        # Some source MusicXML files contain invalid pitch spellings. The
        # validated old raw schema already stores their aligned quarter-length
        # features. Its first eight fields equal musical51's first eight fields.
        old = validated_legacy()
        old_score = old["score"]
        raw_rows = old_score["score_feature"]
        mask = old_score["has_score_feature"]
        if old.get("meta", {}).get("score_feature_keys") != ["mo", "md", "ml", "first", "staff", "trill", "grace", "staccato"]:
            raise ValueError(f"Unsupported legacy score-feature schema: {path}") from exc
        musical_features = []
        for p, r, h in zip(pitch, raw_rows, mask):
            if not h:
                musical_features.append([0.] * 9)
                continue
            if len(r) != 8:
                raise ValueError(f"Invalid legacy feature width: {path}") from exc
            middle_line = 71 if float(r[4]) < .5 else 50
            stem_code = 2. if p >= middle_line else 1.
            musical_features.append(list(r) + [stem_code])
        features = {"has_score_feature": mask, "status": "validated_legacy_annotation"}
        feature_source = "pitch_and_score_raw_matched_legacy8_with_standard_stem_rule"
        xml_error = f"{type(exc).__name__}: {exc}"
    matched = int(sum(features["has_score_feature"]))
    if matched == 0:
        # A few scores have no usable source annotations in either source.
        # Preserve their original missing-feature masks, rather than inventing
        # an alignment or dropping otherwise valid performance training data.
        if feature_source != "pitch_and_score_raw_matched_legacy8_with_standard_stem_rule":
            raise ValueError(f"No validated annotation mask for {path}")
        feature_source = "score_matched_legacy_explicit_missing_annotation_mask"
        features["status"] = "missing_score_annotations"
    prepared_score = {
        "pitch": score["pitch"], "score_raw": score["score_raw"],
        "score_feature": musical_features,
        "has_score_feature": features["has_score_feature"],
    }
    wanted = set(row["selected_performance_sources"])
    performances = []
    for p in work["performances"]:
        if p.get("performance_source") not in wanted or p.get("split", "train") != "train":
            continue
        fields = ("performance_source", "performance_id", "performance_dataset", "split", "interpolated",
                  "label_shared_raw", "label_pedal4_raw", "label_raw", "pedal4_raw")
        copied = {k: p[k] for k in fields if k in p}
        if len(copied.get("label_shared_raw", copied.get("label_raw", []))) != len(pitch):
            raise ValueError(f"Performance length mismatch for {p.get('performance_source')}")
        performances.append(copied)
    available = {p["performance_source"] for p in performances}
    restored_sources = []
    if wanted - available:
        old = validated_legacy()
        for p in old["performances"]:
            if p.get("performance_source") in wanted - available and p.get("split", "train") == "train":
                fields = ("performance_source", "performance_id", "performance_dataset", "split", "interpolated",
                          "label_shared_raw", "label_pedal4_raw", "label_raw", "pedal4_raw")
                copied = {k: p[k] for k in fields if k in p}
                if len(copied.get("label_shared_raw", copied.get("label_raw", []))) != len(pitch):
                    raise ValueError(f"Invalid restored performance length: {path}")
                performances.append(copied)
                available.add(p["performance_source"])
                restored_sources.append(p["performance_source"])
    if available != wanted:
        failures = {p.get("performance_source"): p.get("reason") for p in work.get("failed_performances", [])}
        missing = wanted - available
        unexplained = [s for s in missing if not failures.get(s)]
        if unexplained:
            raise ValueError(f"Missing {len(unexplained)} unexplained metadata performances in {path}")
    else:
        failures, missing = {}, set()
    audit = {"path": str(path), "sidecar": str(target), "notes": len(pitch),
             "matched_notes": matched, "coverage": matched / len(pitch),
             "performances": len(performances), "status": features["status"], "reused": False,
             "feature_source": feature_source, "xml_error": xml_error,
             "restored_performance_sources": restored_sources,
             "excluded_performances": [{"source": s, "reason": failures[s]} for s in sorted(missing)]}
    meta = dict(work.get("meta") or {})
    meta.update({"preparation_protocol": "fresh_xml_musical51_v1", "preparation_audit": audit,
                 "score_feature_layout": "musical51_from_compact_4slot",
                 "score_feature_source_layout": "compact_4slot_v2", "score_feature_unit": "quarter_length_raw",
                 "score_feature_keys": ["mo_q", "md_q", "ml_q", "first", "hand", "trill", "grace", "staccato", "stem_code"]})
    payload = {"score": prepared_score, "performances": performances,
               "performances_by_source": {p["performance_source"]: p for p in performances},
               "label_cache": {}, "perf_style_cache": {}, "meta": meta,
               "performance_time_normalization": "none", "_source_identity": identity,
               "_cache_signature": json.dumps({"kind": "inr_raw_sidecar", "schema": 5}, sort_keys=True, separators=(",", ":"))}
    temp = target.with_name(target.name + f".{os.getpid()}.tmp")
    torch.save(payload, temp)
    temp.replace(target)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("backup/PianoCoRe/metadata.csv"))
    parser.add_argument("--json-root", type=Path, default=Path("backup/PianoCoRe/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("backup/PianoCoRe/raw"))
    parser.add_argument("--tag", default="PCOREA_MUSICAL51")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--retry-errors-from", type=Path,
                        help="Reuse successful audit records and retry only failures from a completed preparation.")
    args = parser.parse_args()
    start = time.monotonic()
    df = pd.read_csv(args.metadata, low_memory=False)
    df = df[df.tier_a.fillna(False).astype(bool) & df.split.eq("train")]
    df = df.dropna(subset=["refined_score_midi_path", "refined_performance_midi_path", "refined_alignment_path"])
    tasks = []
    for _, group in df.groupby("refined_score_midi_path", sort=True):
        row = group.iloc[0].to_dict()
        row["selected_performance_sources"] = sorted(set(group.refined_performance_midi_path))
        tasks.append((row, str(args.json_root), args.tag))
    if args.limit:
        tasks = tasks[:args.limit]
    all_tasks = tasks
    prior_results = []
    if args.retry_errors_from:
        previous = json.loads(args.retry_errors_from.read_text())
        prior_results = [json.loads(line) for line in Path(previous["details"]).read_text().splitlines()
                         if line.strip() and "error" not in json.loads(line)]
        failed_sources = {item["score_source"] for item in previous["errors"]}
        tasks = [task for task in tasks if task[0]["refined_score_midi_path"] in failed_sources]
        failures_by_source = {item["score_source"]: item["error"] for item in previous["errors"]}
        for row, _, _ in tasks:
            reason = failures_by_source[row["refined_score_midi_path"]]
            if (reason.startswith("XMLPreparationTimeout:") or "No MusicXML notes aligned" in reason
                    or reason.startswith("PitchException:")):
                row["_xml_preparation_skip_reason"] = reason
        if len(prior_results) + len(tasks) != len(all_tasks):
            raise ValueError("Previous preparation does not cover the requested work set")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    details = args.summary.with_suffix(".jsonl")
    results, errors = list(prior_results), []
    print(json.dumps({"event": "prepare_start", "works": len(tasks), "workers": args.workers}), flush=True)
    with details.open("w") as ledger, ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.raw_root,)) as pool:
        for result in prior_results:
            ledger.write(json.dumps(result, ensure_ascii=False) + "\n")
        ledger.flush()
        futures = {pool.submit(build_one, t): t[0]["refined_score_midi_path"] for t in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                result = {"score_source": futures[future], "error": f"{type(exc).__name__}: {exc}"}
                errors.append(result)
            ledger.write(json.dumps(result, ensure_ascii=False) + "\n")
            ledger.flush()
            done = len(results) + len(errors)
            if done % 20 == 0 or done == len(all_tasks):
                print(json.dumps({"event": "prepare_progress", "done": done, "works": len(all_tasks), "errors": len(errors)}), flush=True)
    summary = {"works": len(all_tasks), "successful_works": len(results), "errors": errors,
               "performances": sum(r["performances"] for r in results),
               "notes": sum(r["notes"] for r in results),
               "matched_notes": sum(r["matched_notes"] for r in results),
               "elapsed_seconds": time.monotonic() - start, "details": str(details), "tag": args.tag}
    summary["excluded_performances"] = [dict(work=r["path"], **p) for r in results for p in r.get("excluded_performances", [])]
    summary["restored_performances"] = sum(len(r.get("restored_performance_sources", [])) for r in results)
    summary["annotation_fallback_works"] = [r["path"] for r in results if r.get("xml_error")]
    summary["zero_annotation_works"] = [r["path"] for r in results if r["matched_notes"] == 0]
    summary["metadata_performances"] = sum(len(t[0]["selected_performance_sources"]) for t in all_tasks)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
