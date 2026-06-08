#!/usr/bin/env python3
"""Helpers for projecting XML/MXL score features onto refined PianoCoRe notes."""

from __future__ import annotations

import contextlib
import difflib
import math
import os
import signal
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from miditoolkit import MidiFile


warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_ROOT = REPO_ROOT / "MIDI2ScoreTransformer" / "midi2scoretransformer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_ROOT))

from tokenizer import PARAMS, MultistreamTokenizer  # noqa: E402
from src.utils.node_midi import sorted_piano_notes  # noqa: E402


class TimeoutError(RuntimeError):
    pass


@contextlib.contextmanager
def time_limit(seconds: int | float | None):
    if not seconds or seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"operation exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class ZipResolver:
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        self.zip_file = zipfile.ZipFile(zip_path)
        self.names = self.zip_file.namelist()
        self.name_set = set(self.names)
        self.suffix_to_name: dict[str, str] = {}
        for name in self.names:
            normalized = name.replace("\\", "/")
            parts = normalized.split("/")
            for start in range(len(parts)):
                suffix = "/".join(parts[start:])
                self.suffix_to_name.setdefault(suffix, name)

    def close(self):
        self.zip_file.close()

    def resolve(self, relative_path: str) -> str:
        rel = str(relative_path).replace("\\", "/").lstrip("/")
        candidates = [
            rel,
            f"PianoCoRe/raw/{rel}",
            f"raw/{rel}",
        ]
        for candidate in candidates:
            if candidate in self.name_set:
                return candidate
            if candidate in self.suffix_to_name:
                return self.suffix_to_name[candidate]
        raise FileNotFoundError(f"{relative_path} not found in {self.zip_path}")

    def extract_to_temp(self, relative_path: str, suffix: str) -> str:
        member = self.resolve(relative_path)
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with self.zip_file.open(member) as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())
        return tmp_path


def tensor_values(streams: dict[str, Any], key: str) -> list[float]:
    value = streams[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).tolist()


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def normalize_affine(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return clamp01((float(value) - minimum) / (maximum - minimum))


def parse_xml_features(raw_zip: ZipResolver, score_xml_path: str, timeout_sec: float) -> dict[str, Any]:
    suffix = Path(score_xml_path).suffix or ".mxl"
    tmp_path = raw_zip.extract_to_temp(score_xml_path, suffix=suffix)
    try:
        with time_limit(timeout_sec):
            streams = MultistreamTokenizer.parse_mxl(tmp_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)

    pitches = [int(round(v)) for v in tensor_values(streams, "pitch")]
    offsets = tensor_values(streams, "offset")
    durations = tensor_values(streams, "duration")
    downbeats = tensor_values(streams, "downbeat")
    velocities = tensor_values(streams, "velocity") if "velocity" in streams else [64.0] * len(pitches)
    hands = [int(round(v)) for v in tensor_values(streams, "hand")]
    trills = [int(round(v)) for v in tensor_values(streams, "trill")]
    graces = [int(round(v)) for v in tensor_values(streams, "grace")]
    staccatos = [int(round(v)) for v in tensor_values(streams, "staccato")]

    downbeat_min = float(PARAMS["downbeat"]["min"])
    score_continuous = []
    score_structure = []
    score_annotation = []
    unknown_staff = 0
    for idx in range(len(pitches)):
        staff_raw = hands[idx] if idx < len(hands) else 2
        if staff_raw not in (0, 1):
            unknown_staff += 1
        staff = 1 if staff_raw == 1 else 0
        downbeat = float(downbeats[idx])
        first = 1 if downbeat > downbeat_min + 1e-6 else 0
        ml_raw = max(downbeat, 0.0)
        score_continuous.append([clamp01(float(velocities[idx]) / 127.0)])
        score_structure.append(
            [
                normalize_affine(offsets[idx], PARAMS["offset"]["min"], PARAMS["offset"]["max"]),
                normalize_affine(durations[idx], PARAMS["duration"]["min"], PARAMS["duration"]["max"]),
                normalize_affine(ml_raw, 0.0, PARAMS["downbeat"]["max"]),
                first,
            ]
        )
        score_annotation.append([staff, trills[idx], graces[idx], staccatos[idx]])

    return {
        "pitch": pitches,
        "score_continuous": score_continuous,
        "score_structure": score_structure,
        "score_annotation": score_annotation,
        "unknown_staff_count": unknown_staff,
        "trill_count": int(sum(trills)),
        "grace_count": int(sum(graces)),
        "staccato_count": int(sum(staccatos)),
    }


def load_midi_pitches_from_zip(raw_zip: ZipResolver, midi_path: str) -> list[int]:
    tmp_path = raw_zip.extract_to_temp(midi_path, suffix=".mid")
    try:
        midi = MidiFile(tmp_path)
        return [int(note.pitch) for note in sorted_piano_notes(midi)]
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)


def load_midi_pitches(path: Path) -> list[int]:
    midi = MidiFile(str(path))
    return [int(note.pitch) for note in sorted_piano_notes(midi)]


def subsequence_map(target: list[int], source: list[int]) -> dict[int, int] | None:
    """Return target-index -> source-index if target is a subsequence of source."""
    mapping: dict[int, int] = {}
    source_idx = 0
    for target_idx, pitch in enumerate(target):
        while source_idx < len(source) and source[source_idx] != pitch:
            source_idx += 1
        if source_idx >= len(source):
            return None
        mapping[target_idx] = source_idx
        source_idx += 1
    return mapping


def invert_mapping(mapping: dict[int, int]) -> dict[int, int]:
    return {value: key for key, value in mapping.items()}


def sequence_matcher_map(
    target: list[int],
    source: list[int],
    max_notes: int,
    timeout_sec: float,
) -> dict[int, int]:
    """Map equal blocks as target-index -> source-index."""
    if len(target) > max_notes or len(source) > max_notes:
        return {}
    with time_limit(timeout_sec):
        matcher = difflib.SequenceMatcher(None, target, source, autojunk=False)
        mapping: dict[int, int] = {}
        for target_start, source_start, size in matcher.get_matching_blocks():
            for offset in range(size):
                mapping[target_start + offset] = source_start + offset
        return mapping


def build_raw_to_refined_map(
    raw_pitches: list[int],
    refined_pitches: list[int],
    max_notes: int,
    timeout_sec: float,
    use_sequence_matcher: bool,
) -> tuple[str, dict[int, int]]:
    """Return relation and refined-index -> raw-index mapping."""
    if raw_pitches == refined_pitches:
        return "exact", {idx: idx for idx in range(len(refined_pitches))}

    mapping = subsequence_map(refined_pitches, raw_pitches)
    if mapping is not None:
        return "refined_subsequence_raw", mapping

    if use_sequence_matcher:
        try:
            partial = sequence_matcher_map(refined_pitches, raw_pitches, max_notes, timeout_sec)
        except TimeoutError:
            partial = {}
        if partial:
            return "sequence_matcher_partial", partial

    return "failed", {}


def build_raw_to_xml_map(
    raw_pitches: list[int],
    xml_pitches: list[int],
    max_notes: int,
    timeout_sec: float,
    use_sequence_matcher: bool,
) -> tuple[str, dict[int, int]]:
    """Return relation and raw-index -> XML-index mapping."""
    if raw_pitches == xml_pitches:
        return "exact", {idx: idx for idx in range(len(raw_pitches))}

    raw_to_xml = subsequence_map(raw_pitches, xml_pitches)
    if raw_to_xml is not None:
        return "raw_subsequence_xml", raw_to_xml

    xml_to_raw = subsequence_map(xml_pitches, raw_pitches)
    if xml_to_raw is not None:
        return "xml_subsequence_raw", invert_mapping(xml_to_raw)

    if len(raw_pitches) == len(xml_pitches):
        equal_positions = {
            idx: idx for idx, (raw_pitch, xml_pitch) in enumerate(zip(raw_pitches, xml_pitches)) if raw_pitch == xml_pitch
        }
        if equal_positions:
            return "same_length_equal_positions", equal_positions

    if use_sequence_matcher:
        try:
            raw_to_xml = sequence_matcher_map(raw_pitches, xml_pitches, max_notes, timeout_sec)
        except TimeoutError:
            raw_to_xml = {}
        if raw_to_xml:
            return "sequence_matcher_partial", raw_to_xml

    return "failed", {}
