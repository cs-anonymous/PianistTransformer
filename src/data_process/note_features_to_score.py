#!/usr/bin/env python3
"""Render PianoCoRe node JSON files back to MusicXML/MXL scores.

The current JSON schema stores score-side timing and annotations, but not the
MIDI2ScoreTransformer output streams for key signature, spelling accidental,
stem, or voice.  This script reconstructs those missing streams with stable
heuristics:

- infer measure-local pitch spelling under a globally smoothed key-signature
  path;
- heavily penalize key-signature changes and complex key signatures;
- infer voices/stems from staff, overlap, onset, duration, and pitch height.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import zipfile
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from miditoolkit import MidiFile
from music21 import articulations, clef, duration as m21_duration, expressions, instrument, metadata
from music21 import key as m21_key
from music21 import meter, note, pitch as m21_pitch, stream
from music21.common.numberTools import opFrac
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_ROOT = REPO_ROOT / "MIDI2ScoreTransformer" / "midi2scoretransformer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_ROOT))

from score_utils import postprocess_score  # noqa: E402
from src.utils.node_midi import sorted_piano_notes  # noqa: E402


SCORE_GRID = 1.0 / 24.0
LETTERS = ("C", "D", "E", "F", "G", "A", "B")
LETTER_BASE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
ACCIDENTAL_NAMES = {
    -2: "double-flat",
    -1: "flat",
    0: "natural",
    1: "sharp",
    2: "double-sharp",
}


@dataclass
class NoteEvent:
    index: int
    pitch: int
    measure: int
    offset: float
    duration: float
    staff: int
    trill: bool
    grace: bool
    staccato: bool
    voice: int = 1
    stem: str | None = None
    tie_start: bool = False
    tie_stop: bool = False


@dataclass
class MeasureInfo:
    length: float
    numerator: int
    denominator: int


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


def quantize(value: float, grid: float = SCORE_GRID) -> float:
    if value <= 0:
        return 0.0
    return round(float(value) / grid) * grid


def denorm_score_feature(row: list[float]) -> tuple[float, float, float]:
    mo = quantize(clamp(row[0], 0.0, 1.0) * 6.0)
    md = quantize(clamp(row[1], 0.0, 1.0) * 4.0)
    ml = quantize(clamp(row[2], 0.0, 1.0) * 6.0)
    return mo, md, ml


def time_signature_for_length(length: float) -> tuple[int, int]:
    frac = Fraction(max(float(length), SCORE_GRID) / 4.0).limit_denominator(16)
    if frac.denominator == 1:
        return frac.numerator * 4, 4
    if frac.denominator == 2:
        return frac.numerator * 4, 8
    if frac.denominator == 4:
        return frac.numerator, 4
    if frac.denominator == 8:
        return frac.numerator, 8
    if frac.denominator == 16:
        return frac.numerator, 16
    return 4, 4


def score_source_path(payload: dict[str, Any]) -> str | None:
    score = payload.get("score") or {}
    meta = payload.get("meta") or {}
    return score.get("score_source") or meta.get("score_source")


def feature_flags(payload: dict[str, Any], index: int, pitch_value: int) -> tuple[int, bool, bool, bool]:
    score = payload["score"]
    features = score.get("score_feature") or []
    has = score.get("has_score_feature") or []
    if index < len(features) and (not has or bool(has[index])):
        row = features[index]
        staff = 1 if len(row) > 4 and row[4] >= 0.5 else 0
        trill = bool(len(row) > 5 and row[5] >= 0.5)
        grace = bool(len(row) > 6 and row[6] >= 0.5)
        staccato = bool(len(row) > 7 and row[7] >= 0.5)
        return staff, trill, grace, staccato
    return (0 if pitch_value >= 60 else 1), False, False, False


def clone_event_segment(
    event: NoteEvent,
    index: int,
    measure: int,
    offset: float,
    duration: float,
    tie_start: bool,
    tie_stop: bool,
) -> NoteEvent:
    return NoteEvent(
        index=index,
        pitch=event.pitch,
        measure=measure,
        offset=offset,
        duration=duration,
        staff=event.staff,
        trill=event.trill and not tie_stop,
        grace=event.grace,
        staccato=event.staccato and not tie_stop,
        voice=event.voice,
        stem=event.stem,
        tie_start=tie_start,
        tie_stop=tie_stop,
    )


def split_cross_measure_events(events: list[NoteEvent], measure_infos: list[MeasureInfo]) -> list[NoteEvent]:
    output: list[NoteEvent] = []
    next_index = 0
    for event in events:
        if event.grace:
            event.index = next_index
            next_index += 1
            output.append(event)
            continue
        measure = event.measure
        offset = event.offset
        remaining = max(event.duration, SCORE_GRID)
        first_segment = True
        while remaining > 1e-6:
            while measure >= len(measure_infos):
                measure_infos.append(measure_infos[-1] if measure_infos else MeasureInfo(4.0, 4, 4))
            measure_length = measure_infos[measure].length
            if offset >= measure_length - 1e-6:
                measure += 1
                offset = 0.0
                continue
            available = max(measure_length - offset, SCORE_GRID)
            segment_duration = min(remaining, available)
            tie_start = remaining - segment_duration > 1e-6
            output.append(
                clone_event_segment(
                    event,
                    next_index,
                    measure,
                    offset,
                    max(segment_duration, SCORE_GRID),
                    tie_start=tie_start,
                    tie_stop=not first_segment,
                )
            )
            next_index += 1
            remaining -= segment_duration
            measure += 1
            offset = 0.0
            first_segment = False
    return output


def build_measure_grid(midi: MidiFile, max_tick: int) -> tuple[list[int], list[MeasureInfo]]:
    ticks_per_beat = midi.ticks_per_beat
    time_sigs = sorted(midi.time_signature_changes, key=lambda item: item.time)
    if not time_sigs or time_sigs[0].time != 0:
        default = type("DefaultTS", (), {"numerator": 4, "denominator": 4, "time": 0})()
        time_sigs = [default, *time_sigs]

    starts: list[int] = []
    infos: list[MeasureInfo] = []
    ts_idx = 0
    tick = 0
    while tick <= max_tick + ticks_per_beat:
        while ts_idx + 1 < len(time_sigs) and time_sigs[ts_idx + 1].time <= tick:
            ts_idx += 1
        ts = time_sigs[ts_idx]
        length_q = float(ts.numerator) * 4.0 / float(ts.denominator)
        length_ticks = max(1, int(round(length_q * ticks_per_beat)))
        next_ts_tick = time_sigs[ts_idx + 1].time if ts_idx + 1 < len(time_sigs) else None
        starts.append(tick)
        infos.append(MeasureInfo(length=length_q, numerator=int(ts.numerator), denominator=int(ts.denominator)))
        if next_ts_tick is not None and tick < next_ts_tick < tick + length_ticks:
            tick = int(next_ts_tick)
        else:
            tick += length_ticks
    return starts, infos


def events_from_midi(payload: dict[str, Any], json_path: Path, refined_dir: Path) -> tuple[list[NoteEvent], list[MeasureInfo]] | None:
    rel = score_source_path(payload)
    if not rel:
        return None
    midi_path = refined_dir / rel
    if not midi_path.exists():
        return None

    midi = MidiFile(str(midi_path))
    midi_notes = sorted_piano_notes(midi)
    pitches = [int(value) for value in payload["score"]["pitch"]]
    if len(midi_notes) != len(pitches):
        return None
    if any(int(note.pitch) != pitch_value for note, pitch_value in zip(midi_notes, pitches)):
        return None

    max_tick = max((note.end for note in midi_notes), default=0)
    starts, measure_infos = build_measure_grid(midi, max_tick)
    events = []
    ticks_per_beat = midi.ticks_per_beat
    for idx, midi_note in enumerate(midi_notes):
        measure_idx = max(0, bisect_right(starts, midi_note.start) - 1)
        if measure_idx >= len(measure_infos):
            measure_idx = len(measure_infos) - 1
        offset = quantize((midi_note.start - starts[measure_idx]) / ticks_per_beat)
        dur = quantize((midi_note.end - midi_note.start) / ticks_per_beat)
        if offset >= measure_infos[measure_idx].length - 1e-6 and measure_idx + 1 < len(measure_infos):
            measure_idx += 1
            offset = 0.0
        staff, trill, grace, staccato = feature_flags(payload, idx, int(midi_note.pitch))
        events.append(
            NoteEvent(
                index=len(events),
                pitch=int(midi_note.pitch),
                measure=measure_idx,
                offset=offset,
                duration=max(dur, SCORE_GRID),
                staff=staff,
                trill=trill,
                grace=grace,
                staccato=staccato,
            )
        )

    last_measure = max((event.measure for event in events), default=-1)
    measure_infos = measure_infos[: last_measure + 1]
    return split_cross_measure_events(events, measure_infos), measure_infos


def events_from_score_feature(payload: dict[str, Any]) -> tuple[list[NoteEvent], list[MeasureInfo]]:
    score = payload["score"]
    pitches = [int(value) for value in score["pitch"]]
    features = score.get("score_feature") or []
    has = score.get("has_score_feature") or [1] * len(pitches)

    events: list[NoteEvent] = []
    measure_infos: list[MeasureInfo] = []
    measure = 0
    prev_offset = 0.0
    current_length = 4.0
    for idx, pitch_value in enumerate(pitches):
        row = features[idx] if idx < len(features) else [0.0] * 8
        mo, md, ml = denorm_score_feature(row)
        has_feature = idx < len(has) and bool(has[idx])
        starts_new_measure = idx > 0 and has_feature and ml > 0 and (row[3] >= 0.5 or mo <= prev_offset + 1e-6)
        if starts_new_measure:
            current_length = max(ml, SCORE_GRID)
            num, den = time_signature_for_length(current_length)
            measure_infos.append(MeasureInfo(current_length, num, den))
            measure += 1
        staff, trill, grace, staccato = feature_flags(payload, idx, pitch_value)
        if not has_feature and events:
            mo = min(events[-1].offset + SCORE_GRID, current_length - SCORE_GRID)
            md = SCORE_GRID
        events.append(
            NoteEvent(
                index=idx,
                pitch=pitch_value,
                measure=measure,
                offset=mo,
                duration=max(md, SCORE_GRID),
                staff=staff,
                trill=trill,
                grace=grace,
                staccato=staccato,
            )
        )
        prev_offset = mo

    while len(measure_infos) <= measure:
        measure_events = [event for event in events if event.measure == len(measure_infos)]
        inferred = max((event.offset + event.duration for event in measure_events), default=current_length)
        length = max(current_length, quantize(inferred))
        num, den = time_signature_for_length(length)
        measure_infos.append(MeasureInfo(length, num, den))
    return split_cross_measure_events(events, measure_infos), measure_infos


def key_default_alters(fifths: int) -> dict[str, int]:
    alters = {letter: 0 for letter in LETTERS}
    if fifths > 0:
        for letter in SHARP_ORDER[: min(fifths, 7)]:
            alters[letter] = 1
    elif fifths < 0:
        for letter in FLAT_ORDER[: min(-fifths, 7)]:
            alters[letter] = -1
    return alters


def spelling_candidates(midi_pitch: int) -> list[tuple[str, int, int]]:
    pc = midi_pitch % 12
    candidates = []
    for letter, base_pc in LETTER_BASE_PC.items():
        for alter in range(-2, 3):
            if (base_pc + alter) % 12 == pc:
                octave_num = (midi_pitch - base_pc - alter) // 12 - 1
                if (octave_num + 1) * 12 + base_pc + alter == midi_pitch:
                    candidates.append((letter, alter, octave_num))
    return candidates


def spelling_cost_for_pc(pc: int, fifths: int) -> float:
    default_alters = key_default_alters(fifths)
    best = float("inf")
    for letter, base_pc in LETTER_BASE_PC.items():
        for alter in range(-2, 3):
            if (base_pc + alter) % 12 != pc:
                continue
            default = default_alters[letter]
            if alter == default:
                cost = 0.0
            else:
                cost = 1.0 + 0.25 * abs(alter - default)
            cost += 0.06 * abs(alter)
            if abs(alter) > 1:
                cost += 4.0
            if fifths > 0 and alter < 0:
                cost += 0.18
            elif fifths < 0 and alter > 0:
                cost += 0.18
            best = min(best, cost)
    return best


def measure_key_cost(pitch_counts: Counter[int], fifths: int, key_complexity: float) -> float:
    if not pitch_counts:
        return key_complexity * abs(fifths)
    total = 0.0
    for pc, count in pitch_counts.items():
        total += spelling_cost_for_pc(pc, fifths) * (1.0 + 0.12 * math.log1p(count))
    return total + key_complexity * abs(fifths)


def smooth_short_key_runs(keys: list[int], pitch_by_measure: list[Counter[int]], key_complexity: float, min_run: int) -> list[int]:
    if min_run <= 1 or not keys:
        return keys
    keys = list(keys)
    idx = 0
    while idx < len(keys):
        end = idx + 1
        while end < len(keys) and keys[end] == keys[idx]:
            end += 1
        if end - idx < min_run:
            left = keys[idx - 1] if idx > 0 else None
            right = keys[end] if end < len(keys) else None
            candidates = [candidate for candidate in (left, right) if candidate is not None]
            if candidates:
                def candidate_cost(candidate: int) -> float:
                    return sum(measure_key_cost(pitch_by_measure[pos], candidate, key_complexity) for pos in range(idx, end))

                replacement = min(candidates, key=candidate_cost)
                for pos in range(idx, end):
                    keys[pos] = replacement
        idx = end
    return keys


def infer_key_fifths_by_measure(
    events: list[NoteEvent],
    measure_count: int,
    key_complexity: float,
    change_penalty: float,
    min_run: int,
) -> list[int]:
    pitch_by_measure = [Counter() for _ in range(measure_count)]
    for event in events:
        if not event.grace and 0 <= event.measure < measure_count:
            pitch_by_measure[event.measure][event.pitch % 12] += 1

    candidates = list(range(-7, 8))
    costs = [[measure_key_cost(pitches, fifths, key_complexity) for fifths in candidates] for pitches in pitch_by_measure]
    if not costs:
        return []

    dp: list[list[float]] = []
    back: list[list[int]] = []
    dp.append(costs[0][:])
    back.append([-1] * len(candidates))
    for measure_idx in range(1, measure_count):
        row = []
        back_row = []
        for key_idx, fifths in enumerate(candidates):
            best_prev = 0
            best_cost = float("inf")
            for prev_idx, prev_fifths in enumerate(candidates):
                transition = 0.0
                if prev_fifths != fifths:
                    transition = change_penalty + 0.15 * abs(prev_fifths - fifths)
                value = dp[measure_idx - 1][prev_idx] + transition
                if value < best_cost:
                    best_cost = value
                    best_prev = prev_idx
            row.append(best_cost + costs[measure_idx][key_idx])
            back_row.append(best_prev)
        dp.append(row)
        back.append(back_row)

    idx = min(range(len(candidates)), key=lambda key_idx: dp[-1][key_idx])
    keys = [0] * measure_count
    for measure_idx in range(measure_count - 1, -1, -1):
        keys[measure_idx] = candidates[idx]
        idx = back[measure_idx][idx]
        if idx < 0:
            break
    return smooth_short_key_runs(keys, pitch_by_measure, key_complexity, min_run)


def choose_spelling(midi_pitch: int, fifths: int) -> m21_pitch.Pitch:
    default_alters = key_default_alters(fifths)
    best: tuple[float, str, int, int] | None = None
    for letter, alter, octave_num in spelling_candidates(midi_pitch):
        default = default_alters[letter]
        if alter == default:
            cost = 0.0
        else:
            cost = 1.0 + 0.25 * abs(alter - default)
        cost += 0.06 * abs(alter)
        if abs(alter) > 1:
            cost += 4.0
        if fifths > 0 and alter < 0:
            cost += 0.18
        elif fifths < 0 and alter > 0:
            cost += 0.18
        candidate = (cost, letter, alter, octave_num)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        out = m21_pitch.Pitch()
        out.midi = midi_pitch
        return out

    _, letter, alter, octave_num = best
    out = m21_pitch.Pitch()
    out.step = letter
    out.octave = octave_num
    default = default_alters[letter]
    if alter != 0 or default != 0:
        accidental = m21_pitch.Accidental(ACCIDENTAL_NAMES[alter])
        accidental.displayStatus = alter != default
        out.accidental = accidental
    return out


def choose_spelling_parts(midi_pitch: int, fifths: int) -> tuple[str, int, int, bool]:
    pitch_obj = choose_spelling(midi_pitch, fifths)
    alter = int(pitch_obj.accidental.alter) if pitch_obj.accidental is not None else 0
    show_accidental = bool(
        pitch_obj.accidental is not None
        and pitch_obj.accidental.displayStatus is not False
    )
    return pitch_obj.step, alter, int(pitch_obj.octave), show_accidental


def assign_voices_and_stems(events: list[NoteEvent]) -> None:
    by_measure_staff: dict[tuple[int, int], list[NoteEvent]] = {}
    for event in events:
        by_measure_staff.setdefault((event.measure, event.staff), []).append(event)

    for (_measure, staff), staff_events in by_measure_staff.items():
        groups: dict[tuple[float, float, bool], list[NoteEvent]] = {}
        for event in staff_events:
            groups.setdefault((event.offset, event.duration, event.grace), []).append(event)
        sorted_groups = sorted(groups.values(), key=lambda group: (group[0].offset, -sum(e.pitch for e in group) / len(group)))
        voice_ends = [0.0, 0.0, 0.0, 0.0]
        onset_to_groups: dict[float, list[list[NoteEvent]]] = {}
        for group in sorted_groups:
            onset_to_groups.setdefault(group[0].offset, []).append(group)

        group_voice: dict[int, int] = {}
        for onset, onset_groups in sorted(onset_to_groups.items()):
            onset_groups.sort(key=lambda group: sum(e.pitch for e in group) / len(group), reverse=True)
            for rank, group in enumerate(onset_groups):
                preferred = 0 if rank == 0 else 1
                free = [idx for idx, end in enumerate(voice_ends) if end <= onset + 1e-6]
                if preferred in free:
                    voice_idx = preferred
                elif free:
                    voice_idx = free[0]
                else:
                    voice_idx = min(range(len(voice_ends)), key=lambda idx: voice_ends[idx])
                end = onset if group[0].grace else onset + max(event.duration for event in group)
                voice_ends[voice_idx] = max(voice_ends[voice_idx], end)
                for event in group:
                    group_voice[event.index] = voice_idx + 1

        polyphonic_offsets = {
            onset for onset, onset_groups in onset_to_groups.items() if len(onset_groups) > 1
        }
        for event in staff_events:
            event.voice = group_voice.get(event.index, 1)
            if event.offset in polyphonic_offsets or event.voice > 1:
                event.stem = "up" if event.voice == 1 else "down"
            else:
                middle_line = 71 if staff == 0 else 50
                event.stem = "down" if event.pitch >= middle_line else "up"


def build_score(
    events: list[NoteEvent],
    measure_infos: list[MeasureInfo],
    key_fifths: list[int],
    title: str | None = None,
) -> stream.Score:
    assign_voices_and_stems(events)
    score = stream.Score()
    events_by_part_measure: dict[tuple[int, int], list[NoteEvent]] = {}
    for event in events:
        events_by_part_measure.setdefault((event.staff, event.measure), []).append(event)

    for part_idx in (0, 1):
        part = stream.Part()
        ins = instrument.Piano()
        ins.partId = f"P{part_idx + 1}"
        part.insert(0, ins)
        offset = 0.0
        last_ts: tuple[int, int] | None = None
        last_key: int | None = None
        for measure_idx, info in enumerate(measure_infos):
            m = stream.Measure(number=measure_idx + 1)
            if measure_idx == 0:
                m.insert(0, clef.TrebleClef() if part_idx == 0 else clef.BassClef())
            ts = (info.numerator, info.denominator)
            if ts != last_ts:
                m.insert(0, meter.TimeSignature(f"{ts[0]}/{ts[1]}"))
                last_ts = ts
            fifths = key_fifths[measure_idx] if measure_idx < len(key_fifths) else 0
            if fifths != last_key:
                m.insert(0, m21_key.KeySignature(fifths))
                last_key = fifths

            measure_events = sorted(
                events_by_part_measure.get((part_idx, measure_idx), []),
                key=lambda event: (event.voice, event.offset, event.pitch, event.duration),
            )
            voices: dict[int, stream.Voice] = {voice_id: stream.Voice(id=str(voice_id)) for voice_id in range(1, 5)}
            for event in measure_events:
                voice_id = min(max(int(event.voice), 1), 4)
                voice = voices[voice_id]
                n = note.Note()
                n.pitch = choose_spelling(event.pitch, fifths)
                if event.grace:
                    n.duration = m21_duration.Duration(0).getGraceDuration()
                else:
                    n.duration.quarterLength = opFrac(max(event.duration, SCORE_GRID))
                if event.trill:
                    n.expressions.append(expressions.Trill())
                if event.staccato:
                    n.articulations.append(articulations.Staccato())
                if event.stem:
                    n.stemDirection = event.stem
                voice.insert(opFrac(max(event.offset, 0.0)), n)
            for voice_id in sorted(voices):
                m.insert(0, voices[voice_id])
            part.insert(opFrac(offset), m)
            offset += info.length
        score.insert(0, part)
    if title:
        score.metadata = score.metadata or metadata.Metadata()
        score.metadata.title = title
    try:
        return postprocess_score(score, inPlace=False)
    except Exception:
        return score


def output_path_for_json(json_path: Path, output_mode: str) -> Path:
    if output_mode == "basename":
        name = json_path.name.replace(".node_a.json", ".extracted.mxl")
        return json_path.with_name(name)
    if output_mode == "fixed":
        return json_path.with_name("extracted.mxl")
    raise ValueError(f"unknown output mode: {output_mode}")


def sub(parent: ET.Element, tag: str, text: str | int | None = None, **attrs: str) -> ET.Element:
    child = ET.SubElement(parent, tag, {key: str(value) for key, value in attrs.items()})
    if text is not None:
        child.text = str(text)
    return child


def accidental_xml_name(alter: int) -> str:
    return {
        -2: "flat-flat",
        -1: "flat",
        0: "natural",
        1: "sharp",
        2: "double-sharp",
    }.get(alter, "natural")


def duration_ticks(quarter_length: float, divisions: int = 24) -> int:
    return max(1, int(round(max(float(quarter_length), SCORE_GRID) * divisions)))


def append_note_xml(
    measure: ET.Element,
    event: NoteEvent | None,
    fifths: int,
    voice_id: int,
    divisions: int,
    rest_duration: float | None = None,
    chord_tone: bool = False,
) -> None:
    n = sub(measure, "note")
    if chord_tone:
        sub(n, "chord")
    if event is None:
        sub(n, "rest")
        ticks = duration_ticks(rest_duration or SCORE_GRID, divisions)
        sub(n, "duration", ticks)
        sub(n, "voice", voice_id)
        return

    if event.grace:
        sub(n, "grace", **{"slash": "yes"})
    step, alter, octave_num, show_accidental = choose_spelling_parts(event.pitch, fifths)
    pitch_node = sub(n, "pitch")
    sub(pitch_node, "step", step)
    if alter:
        sub(pitch_node, "alter", alter)
    sub(pitch_node, "octave", octave_num)
    if not event.grace:
        sub(n, "duration", duration_ticks(event.duration, divisions))
    if event.tie_stop:
        sub(n, "tie", type="stop")
    if event.tie_start:
        sub(n, "tie", type="start")
    sub(n, "voice", voice_id)
    if show_accidental:
        sub(n, "accidental", accidental_xml_name(alter))
    if event.stem:
        sub(n, "stem", event.stem)
    if event.trill or event.staccato or event.tie_start or event.tie_stop:
        notations = sub(n, "notations")
        if event.tie_stop:
            sub(notations, "tied", type="stop")
        if event.tie_start:
            sub(notations, "tied", type="start")
        if event.trill:
            ornaments = sub(notations, "ornaments")
            sub(ornaments, "trill-mark")
        if event.staccato:
            articulations_node = sub(notations, "articulations")
            sub(articulations_node, "staccato")


def write_events_mxl(
    events: list[NoteEvent],
    measure_infos: list[MeasureInfo],
    key_fifths: list[int],
    out_path: Path,
    title: str | None,
) -> str:
    assign_voices_and_stems(events)
    divisions = 24
    root = ET.Element("score-partwise", {"version": "3.1"})
    if title:
        work = sub(root, "work")
        sub(work, "work-title", title)
    part_list = sub(root, "part-list")
    for part_idx in (0, 1):
        score_part = sub(part_list, "score-part", id=f"P{part_idx + 1}")
        sub(score_part, "part-name", "Piano")

    events_by_part_measure: dict[tuple[int, int], list[NoteEvent]] = {}
    for event in events:
        events_by_part_measure.setdefault((event.staff, event.measure), []).append(event)

    for part_idx in (0, 1):
        part = sub(root, "part", id=f"P{part_idx + 1}")
        last_key: int | None = None
        last_ts: tuple[int, int] | None = None
        for measure_idx, info in enumerate(measure_infos):
            measure = sub(part, "measure", number=measure_idx + 1)
            fifths = key_fifths[measure_idx] if measure_idx < len(key_fifths) else 0
            ts = (info.numerator, info.denominator)
            if measure_idx == 0 or fifths != last_key or ts != last_ts:
                attrs = sub(measure, "attributes")
                if measure_idx == 0:
                    sub(attrs, "divisions", divisions)
                if measure_idx == 0 or fifths != last_key:
                    key_node = sub(attrs, "key")
                    sub(key_node, "fifths", fifths)
                    last_key = fifths
                if measure_idx == 0 or ts != last_ts:
                    time_node = sub(attrs, "time")
                    sub(time_node, "beats", ts[0])
                    sub(time_node, "beat-type", ts[1])
                    last_ts = ts
                if measure_idx == 0:
                    clef_node = sub(attrs, "clef")
                    if part_idx == 0:
                        sub(clef_node, "sign", "G")
                        sub(clef_node, "line", 2)
                    else:
                        sub(clef_node, "sign", "F")
                        sub(clef_node, "line", 4)

            measure_events = events_by_part_measure.get((part_idx, measure_idx), [])
            by_voice: dict[int, list[NoteEvent]] = {}
            for event in measure_events:
                by_voice.setdefault(min(max(int(event.voice), 1), 4), []).append(event)
            if not by_voice:
                append_note_xml(measure, None, fifths, 1, divisions, rest_duration=info.length)
                continue

            first_voice = True
            measure_ticks = duration_ticks(info.length, divisions)
            for voice_id in sorted(by_voice):
                if not first_voice:
                    backup = sub(measure, "backup")
                    sub(backup, "duration", measure_ticks)
                first_voice = False
                voice_events = sorted(by_voice[voice_id], key=lambda item: (item.offset, item.grace, item.duration, item.pitch))
                groups: dict[tuple[float, float, bool], list[NoteEvent]] = {}
                for event in voice_events:
                    groups.setdefault((event.offset, event.duration, event.grace), []).append(event)
                cursor = 0.0
                for (offset, dur, is_grace), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
                    safe_offset = min(max(offset, 0.0), max(info.length - SCORE_GRID, 0.0))
                    actual_offset = max(safe_offset, cursor)
                    if actual_offset >= info.length - 1e-6 and not is_grace:
                        continue
                    safe_duration = min(max(dur, SCORE_GRID), max(info.length - actual_offset, SCORE_GRID))
                    if actual_offset > cursor + 1e-6:
                        append_note_xml(measure, None, fifths, voice_id, divisions, rest_duration=min(actual_offset - cursor, max(info.length - cursor, 0.0)))
                        cursor = actual_offset
                    group.sort(key=lambda item: item.pitch)
                    for note_idx, event in enumerate(group):
                        if event.offset != actual_offset or event.duration != safe_duration:
                            event = clone_event_segment(
                                event,
                                event.index,
                                event.measure,
                                actual_offset,
                                safe_duration,
                                event.tie_start,
                                event.tie_stop,
                            )
                        append_note_xml(measure, event, fifths, voice_id, divisions, chord_tone=note_idx > 0)
                    if not is_grace:
                        cursor = min(info.length, max(cursor, actual_offset + safe_duration))

    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>
  </rootfiles>
</container>
""".encode("utf-8")
    tmp_path = out_path.with_name(out_path.name + ".tmp.mxl")
    tmp_path.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp_path, "w") as archive:
        archive.writestr("mimetype", "application/vnd.recordare.musicxml", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("score.musicxml", xml_bytes, compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp_path, out_path)
    return "direct_musicxml"


def convert_one(task: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    json_path = Path(task["json_path"])
    refined_dir = Path(task["refined_dir"])
    output_mode = task["output_mode"]
    overwrite = bool(task["overwrite"])
    key_complexity = float(task["key_complexity"])
    change_penalty = float(task["change_penalty"])
    min_key_run = int(task["min_key_run_measures"])
    out_path = output_path_for_json(json_path, output_mode)
    result = {"json_path": str(json_path), "output_path": str(out_path)}
    if out_path.exists() and not overwrite:
        result.update({"status": "skipped", "reason": "exists", "elapsed_sec": 0.0})
        return result

    try:
        payload = json.load(json_path.open(encoding="utf-8"))
        input_note_count = len(payload.get("score", {}).get("pitch") or [])
        source = "midi"
        built = events_from_midi(payload, json_path, refined_dir)
        if built is None:
            source = "score_feature"
            built = events_from_score_feature(payload)
        events, measure_infos = built
        key_fifths = infer_key_fifths_by_measure(events, len(measure_infos), key_complexity, change_penalty, min_key_run)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_mode = write_events_mxl(events, measure_infos, key_fifths, out_path, title=json_path.parent.name)
        result.update(
            {
                "status": "ok",
                "source": source,
                "write_mode": write_mode,
                "note_count": input_note_count,
                "written_note_count": len(events),
                "measure_count": len(measure_infos),
                "key_change_count": sum(1 for a, b in zip(key_fifths, key_fifths[1:]) if a != b),
                "key_fifths_counts": dict(Counter(key_fifths)),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        tmp_path = out_path.with_name(out_path.name + ".tmp.mxl")
        tmp_path.unlink(missing_ok=True)
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "elapsed_sec": round(time.time() - started, 3)})
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(result.get("status", "unknown") for result in results)
    source_counts = Counter(result.get("source", "none") for result in results if result.get("status") == "ok")
    total_notes = sum(int(result.get("note_count") or 0) for result in results)
    total_written_notes = sum(int(result.get("written_note_count") or 0) for result in results)
    total_measures = sum(int(result.get("measure_count") or 0) for result in results)
    total_key_changes = sum(int(result.get("key_change_count") or 0) for result in results)
    return {
        "total": len(results),
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "total_notes": total_notes,
        "total_written_notes": total_written_notes,
        "total_measures": total_measures,
        "total_key_changes": total_key_changes,
        "avg_key_changes_per_score": total_key_changes / max(status_counts.get("ok", 0), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json-root", type=Path, default=Path("PianoCoRe/processed"))
    parser.add_argument("--refined-dir", type=Path, default=Path("PianoCoRe/refined"))
    parser.add_argument("--summary-path", type=Path, default=Path("PianoCoRe/processed/pianocore_a_extracted_mxl_summary.json"))
    parser.add_argument("--details-path", type=Path, default=Path("PianoCoRe/processed/pianocore_a_extracted_mxl_details.jsonl"))
    parser.add_argument("--num-proc", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-mode", choices=["basename", "fixed"], default="basename")
    parser.add_argument("--key-complexity", type=float, default=0.35)
    parser.add_argument("--change-penalty", type=float, default=12.0)
    parser.add_argument("--min-key-run-measures", type=int, default=4)
    args = parser.parse_args()

    json_paths = sorted(path for path in args.json_root.rglob("*.node_a.json"))
    if args.limit is not None:
        json_paths = json_paths[: args.limit]
    tasks = [
        {
            "json_path": str(path),
            "refined_dir": str(args.refined_dir),
            "output_mode": args.output_mode,
            "overwrite": args.overwrite,
            "key_complexity": args.key_complexity,
            "change_penalty": args.change_penalty,
            "min_key_run_measures": args.min_key_run_measures,
        }
        for path in json_paths
    ]

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.details_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with args.details_path.open("w", encoding="utf-8") as details_file:
        if args.num_proc > 1:
            with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
                futures = [executor.submit(convert_one, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting MXL"):
                    result = future.result()
                    results.append(result)
                    details_file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    details_file.flush()
        else:
            for task in tqdm(tasks, desc="Extracting MXL"):
                result = convert_one(task)
                results.append(result)
                details_file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                details_file.flush()

    summary = summarize(results)
    summary.update(
        {
            "json_root": str(args.json_root),
            "refined_dir": str(args.refined_dir),
            "summary_path": str(args.summary_path),
            "details_path": str(args.details_path),
            "num_proc": args.num_proc,
            "output_mode": args.output_mode,
            "key_complexity": args.key_complexity,
            "change_penalty": args.change_penalty,
            "min_key_run_measures": args.min_key_run_measures,
        }
    )
    with args.summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
        summary_file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
