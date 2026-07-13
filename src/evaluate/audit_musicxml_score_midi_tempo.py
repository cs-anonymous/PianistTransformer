#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from miditoolkit import MidiFile


BEAT_QUARTERS = {
    "maxima": 32.0,
    "long": 16.0,
    "breve": 8.0,
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
    "128th": 0.03125,
    "256th": 0.015625,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit MusicXML metronome/sound tempo and score MIDI tempo consistency.")
    parser.add_argument("--metadata", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("../PianoCoRe/raw"))
    parser.add_argument("--refined-root", type=Path, default=Path("../PianoCoRe/refined"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=36)
    return parser.parse_args()


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def children(node, name):
    return [child for child in list(node) if local_name(child.tag) == name]


def descendants(node, name):
    return [child for child in node.iter() if local_name(child.tag) == name]


def load_xml_root(path):
    if path.suffix.lower() != ".mxl":
        return ET.parse(path).getroot()
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.lower().endswith((".xml", ".musicxml")) and "meta-inf/" not in name.lower()
            ]
            if not candidates:
                raise ValueError("MXL has no score XML member")
            candidates.sort(key=lambda name: (name.count("/"), len(name), name))
            return ET.fromstring(archive.read(candidates[0]))
    except zipfile.BadZipFile:
        return ET.parse(path).getroot()


def metronome_quarter_bpm(metronome):
    beat_units = descendants(metronome, "beat-unit")
    per_minutes = descendants(metronome, "per-minute")
    if not beat_units or not per_minutes:
        return None, None, None
    beat_unit = (beat_units[0].text or "").strip().lower()
    factor = BEAT_QUARTERS.get(beat_unit)
    if factor is None:
        return None, beat_unit, (per_minutes[0].text or "").strip()
    dots = len(descendants(metronome, "beat-unit-dot"))
    dot_factor = sum(0.5 ** index for index in range(dots + 1))
    per_minute_text = (per_minutes[0].text or "").strip()
    try:
        per_minute = float(per_minute_text)
    except ValueError:
        return None, beat_unit, per_minute_text
    return per_minute * factor * dot_factor, beat_unit + ("." * dots), per_minute_text


def audit_xml(path, rel_tol, abs_tol):
    root = load_xml_root(path)
    events = []
    for direction_index, direction in enumerate(descendants(root, "direction")):
        sounds = descendants(direction, "sound")
        sound_tempos = []
        for sound in sounds:
            value = sound.attrib.get("tempo")
            if value is not None:
                try:
                    sound_tempos.append(float(value))
                except ValueError:
                    pass
        metronomes = descendants(direction, "metronome")
        for metronome_index, metronome in enumerate(metronomes):
            expected, beat_unit, per_minute = metronome_quarter_bpm(metronome)
            sound_tempo = sound_tempos[0] if sound_tempos else None
            matches = None
            ratio = None
            if expected is not None and sound_tempo is not None:
                matches = math.isclose(sound_tempo, expected, rel_tol=rel_tol, abs_tol=abs_tol)
                ratio = sound_tempo / expected if expected else None
            events.append({
                "direction_index": direction_index,
                "metronome_index": metronome_index,
                "beat_unit": beat_unit,
                "per_minute": per_minute,
                "expected_quarter_bpm": expected,
                "sound_tempo": sound_tempo,
                "sound_to_expected_ratio": ratio,
                "xml_tempo_matches": matches,
            })
    all_sound_tempos = []
    for sound in descendants(root, "sound"):
        value = sound.attrib.get("tempo")
        if value is not None:
            try:
                all_sound_tempos.append(float(value))
            except ValueError:
                pass
    return events, all_sound_tempos


def midi_tempos(path):
    midi = MidiFile(str(path))
    return [(int(event.time), float(event.tempo)) for event in midi.tempo_changes]


def audit_xml_job(args):
    xml_rel, xml_path, rel_tol, abs_tol = args
    try:
        events, sound_tempos = audit_xml(xml_path, rel_tol, abs_tol)
        value = {"events": events, "sound_tempos": sound_tempos, "error": None}
    except Exception as exc:  # noqa: BLE001
        value = {"events": [], "sound_tempos": [], "error": repr(exc)}
    return xml_rel, value


def midi_job(args):
    index, midi_path = args
    try:
        return index, midi_tempos(midi_path), None
    except Exception as exc:  # noqa: BLE001
        return index, [], repr(exc)


def tempo_sets_match(left, right, rel_tol, abs_tol):
    if not left or not right:
        return None
    left = sorted(set(round(float(value), 9) for value in left))
    right = sorted(set(round(float(value), 9) for value in right))
    return all(any(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for b in right) for a in left)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata, usecols=["score_xml_path", "score_midi_path", "refined_score_midi_path"])
    records = metadata.drop_duplicates().to_dict("records")
    xml_rels = sorted({
        record.get("score_xml_path") for record in records
        if isinstance(record.get("score_xml_path"), str) and record.get("score_xml_path")
    })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        xml_cache = dict(pool.map(
            audit_xml_job,
            [(rel, args.raw_root / rel, args.rel_tol, args.abs_tol) for rel in xml_rels],
        ))

    midi_paths = {}
    for record in records:
        for midi_kind, midi_rel, root in (
            ("raw", record.get("score_midi_path"), args.raw_root),
            ("refined", record.get("refined_score_midi_path"), args.refined_root),
        ):
            if isinstance(midi_rel, str) and midi_rel:
                midi_paths[(midi_kind, midi_rel)] = root / midi_rel
    midi_specs = list(midi_paths.items())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        midi_cache = {
            index: (events, error)
            for index, events, error in pool.map(midi_job, midi_specs)
        }

    rows = []
    for record_index, record in enumerate(records):
        xml_rel = record.get("score_xml_path")
        if not isinstance(xml_rel, str) or not xml_rel:
            continue
        audit = xml_cache[xml_rel]
        for midi_kind, midi_rel, root in (
            ("raw", record.get("score_midi_path"), args.raw_root),
            ("refined", record.get("refined_score_midi_path"), args.refined_root),
        ):
            if not isinstance(midi_rel, str) or not midi_rel:
                continue
            midi_events, midi_error = midi_cache[(midi_kind, midi_rel)]
            event_rows = audit["events"] or [{}]
            for event in event_rows:
                midi_values = [tempo for _, tempo in midi_events]
                rows.append({
                    "score_xml_path": xml_rel,
                    "midi_kind": midi_kind,
                    "score_midi_path": midi_rel,
                    "xml_error": audit["error"],
                    "midi_error": midi_error,
                    **event,
                    "xml_sound_tempos": json.dumps(audit["sound_tempos"]),
                    "midi_tempo_events": json.dumps(midi_events),
                    "midi_tempos_match_xml_sound_set": tempo_sets_match(
                        midi_values, audit["sound_tempos"], args.rel_tol, args.abs_tol
                    ),
                })
    frame = pd.DataFrame(rows).drop_duplicates()
    frame.to_csv(args.output_dir / "tempo_audit_all.csv", index=False)
    xml_mismatches = frame[frame["xml_tempo_matches"] == False].copy()  # noqa: E712
    xml_mismatches.to_csv(args.output_dir / "xml_metronome_sound_mismatches.csv", index=False)
    xml_unique = xml_mismatches.drop_duplicates(
        ["score_xml_path", "direction_index", "metronome_index"]
    ).copy()
    xml_unique["relative_error"] = (
        xml_unique["sound_tempo"] / xml_unique["expected_quarter_bpm"] - 1.0
    ).abs()
    significant = xml_unique[xml_unique["relative_error"] > 0.05].copy()
    significant.to_csv(args.output_dir / "xml_tempo_mismatches_over_5pct.csv", index=False)
    severe = xml_unique[
        (xml_unique["sound_to_expected_ratio"] >= 1.5)
        | (xml_unique["sound_to_expected_ratio"] <= (2.0 / 3.0))
    ].copy()
    severe.to_csv(args.output_dir / "xml_tempo_severe_ratio_mismatches.csv", index=False)
    midi_mismatches = frame[frame["midi_tempos_match_xml_sound_set"] == False].copy()  # noqa: E712
    midi_mismatches.to_csv(args.output_dir / "midi_xml_sound_mismatches.csv", index=False)
    summary = {
        "unique_metadata_rows": len(records),
        "unique_musicxml": len(xml_cache),
        "musicxml_parse_errors": sum(value["error"] is not None for value in xml_cache.values()),
        "musicxml_with_metronome": sum(bool(value["events"]) for value in xml_cache.values()),
        "xml_metronome_sound_mismatch_rows": int(len(xml_mismatches)),
        "xml_metronome_sound_mismatch_scores": int(xml_mismatches["score_xml_path"].nunique()),
        "unique_xml_mismatch_events": int(len(xml_unique)),
        "over_5pct_mismatch_events": int(len(significant)),
        "over_5pct_mismatch_scores": int(significant["score_xml_path"].nunique()),
        "severe_ratio_mismatch_events": int(len(severe)),
        "severe_ratio_mismatch_scores": int(severe["score_xml_path"].nunique()),
        "midi_xml_sound_mismatch_rows": int(len(midi_mismatches)),
        "midi_xml_sound_mismatch_files": int(midi_mismatches["score_midi_path"].nunique()),
        "outputs": {
            "all": str((args.output_dir / "tempo_audit_all.csv").resolve()),
            "xml_mismatches": str((args.output_dir / "xml_metronome_sound_mismatches.csv").resolve()),
            "xml_mismatches_over_5pct": str((args.output_dir / "xml_tempo_mismatches_over_5pct.csv").resolve()),
            "xml_severe_ratio_mismatches": str((args.output_dir / "xml_tempo_severe_ratio_mismatches.csv").resolve()),
            "midi_mismatches": str((args.output_dir / "midi_xml_sound_mismatches.csv").resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
