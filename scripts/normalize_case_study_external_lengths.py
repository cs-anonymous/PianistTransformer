#!/usr/bin/env python3
"""Normalize external case-study MIDI fragments to the score segment length."""

from __future__ import annotations

import json
from pathlib import Path

import pretty_midi


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "results" / "case_study"


def normalize_midi(path: Path, target_duration: float) -> dict:
    midi = pretty_midi.PrettyMIDI(str(path))
    current_duration = midi.get_end_time()
    if current_duration <= 0:
        raise ValueError(f"empty midi: {path}")
    if abs(current_duration - target_duration) < 1e-6:
        return {
            "before_duration": current_duration,
            "after_duration": current_duration,
            "scale": 1.0,
        }
    midi.adjust_times([0.0, current_duration], [0.0, target_duration])
    midi.write(str(path))
    fixed = pretty_midi.PrettyMIDI(str(path))
    return {
        "before_duration": current_duration,
        "after_duration": fixed.get_end_time(),
        "scale": target_duration / current_duration,
    }


def main() -> None:
    cases = json.load(open(CASE_ROOT / "manifest.json"))
    report = []
    for case in cases:
        target_duration = pretty_midi.PrettyMIDI(
            str(ROOT / [o for o in case["outputs"] if o["kind"] == "score"][0]["output"])
        ).get_end_time()
        for record in case["baseline_outputs"]:
            if record["method_tag"] not in {"dexter", "virtuosonet_han"}:
                continue
            path = ROOT / record["output"]
            stats = normalize_midi(path, target_duration)
            record["duration_s"] = stats["after_duration"]
            record["normalization"] = {
                "target_duration": target_duration,
                "scale": stats["scale"],
                "source_duration": stats["before_duration"],
            }
            report.append(
                {
                    "case": case["key"],
                    "method_tag": record["method_tag"],
                    "sample_id": record["sample_id"],
                    **stats,
                }
            )
        (CASE_ROOT / case["key"] / "manifest.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (CASE_ROOT / "manifest.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (CASE_ROOT / "external_length_normalization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in report:
        print(
            row["case"],
            row["method_tag"],
            row["sample_id"],
            "scale",
            round(row["scale"], 4),
            "before",
            round(row["before_duration"], 3),
            "after",
            round(row["after_duration"], 3),
        )


if __name__ == "__main__":
    main()
