#!/usr/bin/env python3
"""Helpers for projecting XML/MXL score features onto refined PianoCoRe notes."""

from __future__ import annotations

import contextlib
import difflib
import inspect
import math
import os
import signal
import sys
import tempfile
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from music21 import articulations, converter, expressions, stream as music21_stream
from music21.midi.translate import prepareStreamForMidi
from miditoolkit import MidiFile


warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_ROOTS = [
    REPO_ROOT / "MIDI2ScoreTransformer" / "midi2scoretransformer",
    REPO_ROOT / "backup" / "MIDI2ScoreTransformer" / "midi2scoretransformer",
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for tokenizer_root in reversed(TOKENIZER_ROOTS):
    if tokenizer_root.exists() and str(tokenizer_root) not in sys.path:
        sys.path.insert(0, str(tokenizer_root))


def patch_music21_strip_ties_preserve_voices() -> None:
    """Allow older MIDI2ScoreTransformer calls on newer music21 versions."""
    original_strip_ties = music21_stream.Stream.stripTies
    try:
        parameters = inspect.signature(original_strip_ties).parameters
    except (TypeError, ValueError):
        return
    if "preserveVoices" in parameters:
        return

    def strip_ties_compat(self, *args, preserveVoices=None, **kwargs):  # noqa: N803, ARG001
        return original_strip_ties(self, *args, **kwargs)

    music21_stream.Stream.stripTies = strip_ties_compat


patch_music21_strip_ties_preserve_voices()

from tokenizer import PARAMS, MultistreamTokenizer  # noqa: E402
from score_utils import realize_spanners  # noqa: E402
from src.utils.inr_midi import sorted_piano_notes  # noqa: E402


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


class FileResolver:
    def __init__(self, root: Path):
        self.root = Path(root)

    def close(self):
        return None

    def resolve(self, relative_path: str) -> Path:
        rel = Path(str(relative_path).replace("\\", "/").lstrip("/"))
        candidates = [
            self.root / rel,
            self.root / "PianoCoRe" / "raw" / rel,
            self.root / "raw" / rel,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"{relative_path} not found under {self.root}")

    def extract_to_temp(self, relative_path: str, suffix: str) -> str:
        source = self.resolve(relative_path)
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(source, "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())
        return tmp_path


def make_resolver(path: Path):
    path = Path(path)
    if path.is_dir():
        return FileResolver(path)
    return ZipResolver(path)


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


def clamp_score_grid_value(value: float, minimum: float, maximum: float) -> float:
    if math.isnan(value):
        return minimum
    return min(max(float(value), minimum), maximum)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def robust_xml_stream(mxl_path: str) -> music21_stream.Score:
    mxl = converter.parse(mxl_path, forceSource=True)
    with contextlib.suppress(Exception):
        mxl = realize_spanners(mxl)
    try:
        mxl = mxl.expandRepeats()
    except Exception:
        pass
    with contextlib.suppress(Exception):
        mxl.stripTies(preserveVoices=False, inPlace=True)
    with contextlib.suppress(Exception):
        mxl = prepareStreamForMidi(mxl)
    return mxl


def element_staff(element: Any) -> int:
    part = element.getContextByClass("Part")
    part_id = str(getattr(part, "id", "")).lower()
    if "staff1" in part_id:
        return 0
    if "staff2" in part_id:
        return 1
    return 2


def element_velocity(element: Any) -> int:
    velocity = getattr(getattr(element, "volume", None), "velocity", None)
    if velocity is not None:
        return int(max(0, min(127, round(float(velocity)))))
    cached = getattr(getattr(element, "volume", None), "cachedRealized", None)
    if cached is not None:
        return int(max(0, min(127, round(float(cached) * 127))))
    return 64


def robust_xml_note_entries(mxl: music21_stream.Score) -> list[dict[str, Any]]:
    entries = []
    trills = (expressions.Trill, expressions.InvertedMordent, expressions.Mordent, expressions.Turn)
    staccatos = (articulations.Staccatissimo, articulations.Staccato)

    for element in mxl.flatten().notes:
        if getattr(getattr(element, "style", None), "hideObjectOnPrint", False):
            continue
        if getattr(element, "isChord", False):
            pitches = list(getattr(element, "pitches", []))
        elif hasattr(element, "pitch"):
            pitches = [element.pitch]
        else:
            continue

        measure = element.getContextByClass("Measure")
        measure_offset = safe_float(getattr(measure, "offset", 0.0))
        duration = getattr(element, "duration", None)
        duration_quarter = safe_float(getattr(duration, "quarterLength", 0.0))
        is_grace = bool(getattr(duration, "isGrace", False))
        base_entry = {
            "offset": safe_float(getattr(element, "offset", 0.0)),
            "measure_offset": measure_offset,
            "duration": duration_quarter,
            "velocity": element_velocity(element),
            "hand": element_staff(element),
            "grace": int(is_grace),
            "trill": int(any(isinstance(item, trills) for item in getattr(element, "expressions", []))),
            "staccato": int(any(isinstance(item, staccatos) for item in getattr(element, "articulations", []))),
        }

        for pitch in pitches:
            with contextlib.suppress(Exception):
                entry = dict(base_entry)
                entry["pitch"] = int(round(float(pitch.midi)))
                entries.append(entry)

    entries.sort(key=lambda item: (item["offset"], not bool(item["grace"]), item["pitch"], item["duration"]))
    deduped = []
    for entry in entries:
        if not deduped or entry["offset"] != deduped[-1]["offset"] or entry["pitch"] != deduped[-1]["pitch"]:
            deduped.append(entry)
        elif bool(deduped[-1]["grace"]) or entry["duration"] > deduped[-1]["duration"]:
            deduped[-1] = entry
    return deduped


def pitch_step_to_midi(step: str, octave: str, alter: str | None = None) -> int:
    semitone = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step.upper()]
    return (int(octave) + 1) * 12 + semitone + int(round(float(alter or 0)))


def direct_musicxml_bytes(mxl_path: str) -> bytes:
    path = Path(mxl_path)
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as archive:
            container = "META-INF/container.xml"
            if container in archive.namelist():
                root = ET.fromstring(archive.read(container))
                full_path = root.find(".//{*}rootfile")
                if full_path is not None and full_path.get("full-path"):
                    return archive.read(full_path.get("full-path"))
            for name in archive.namelist():
                if name.lower().endswith((".xml", ".musicxml")) and not name.startswith("META-INF/"):
                    return archive.read(name)
        raise FileNotFoundError(f"no MusicXML score found in {mxl_path}")
    return path.read_bytes()


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return element.find(f"{{*}}{name}")


def direct_child_text(element: ET.Element, name: str, default: str | None = None) -> str | None:
    child = direct_child(element, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def direct_has_descendant(element: ET.Element, names: set[str]) -> bool:
    for child in element.iter():
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name in names:
            return True
    return False


def direct_repeat_direction(measure: ET.Element, direction: str) -> bool:
    for barline in measure.findall("{*}barline"):
        repeat = direct_child(barline, "repeat")
        if repeat is not None and repeat.get("direction") == direction:
            return True
    return False


def direct_ending_numbers(measure: ET.Element) -> set[int]:
    numbers: set[int] = set()
    for ending in measure.findall("{*}barline/{*}ending"):
        for number in (ending.get("number") or "").replace(",", " ").split():
            with contextlib.suppress(ValueError):
                numbers.add(int(number))
    return numbers


def direct_expand_repeats(measures: list[ET.Element]) -> list[ET.Element]:
    expanded: list[ET.Element] = []
    repeat_start = 0
    for idx, measure in enumerate(measures):
        expanded.append(measure)
        if direct_repeat_direction(measure, "backward"):
            for repeated in measures[repeat_start : idx + 1]:
                if 1 not in direct_ending_numbers(repeated):
                    expanded.append(repeated)
            repeat_start = idx + 1
        if direct_repeat_direction(measure, "forward"):
            repeat_start = idx
    return expanded


def parse_mxl_direct(mxl_path: str) -> dict[str, list[float]]:
    root = ET.fromstring(direct_musicxml_bytes(mxl_path))
    entries: list[dict[str, Any]] = []
    trill_tags = {"trill-mark", "inverted-mordent", "mordent", "turn"}
    staccato_tags = {"staccato", "staccatissimo"}

    for part in root.findall("{*}part"):
        divisions = 1.0
        absolute_measure_offset = 0.0
        measures = direct_expand_repeats(part.findall("{*}measure"))
        for measure in measures:
            cursor = 0.0
            measure_max = 0.0
            previous_onset = 0.0
            for item in list(measure):
                local_name = item.tag.rsplit("}", 1)[-1]
                if local_name == "attributes":
                    divisions_text = direct_child_text(item, "divisions")
                    if divisions_text:
                        divisions = max(safe_float(divisions_text, divisions), 1e-9)
                    continue
                if local_name == "backup":
                    duration = safe_float(direct_child_text(item, "duration"), 0.0) / divisions
                    cursor = max(0.0, cursor - duration)
                    continue
                if local_name == "forward":
                    duration = safe_float(direct_child_text(item, "duration"), 0.0) / divisions
                    cursor += duration
                    measure_max = max(measure_max, cursor)
                    continue
                if local_name != "note":
                    continue
                if item.get("print-object") == "no":
                    continue

                is_chord = direct_child(item, "chord") is not None
                is_grace = direct_child(item, "grace") is not None
                onset = previous_onset if is_chord else cursor
                duration = 0.0 if is_grace else safe_float(direct_child_text(item, "duration"), 0.0) / divisions

                pitch_node = direct_child(item, "pitch")
                if pitch_node is not None:
                    step = direct_child_text(pitch_node, "step")
                    octave = direct_child_text(pitch_node, "octave")
                    if step is not None and octave is not None:
                        try:
                            staff_text = direct_child_text(item, "staff")
                            staff = int(staff_text) - 1 if staff_text and staff_text.isdigit() else 2
                            entries.append(
                                {
                                    "offset": absolute_measure_offset + onset,
                                    "measure_offset": absolute_measure_offset,
                                    "duration": duration,
                                    "velocity": 64,
                                    "hand": staff if staff in (0, 1) else 2,
                                    "grace": int(is_grace),
                                    "trill": int(direct_has_descendant(item, trill_tags)),
                                    "staccato": int(direct_has_descendant(item, staccato_tags)),
                                    "pitch": pitch_step_to_midi(step, octave, direct_child_text(pitch_node, "alter")),
                                }
                            )
                        except Exception:
                            pass

                if not is_chord:
                    previous_onset = onset
                    if not is_grace:
                        cursor += duration
                        measure_max = max(measure_max, cursor)
                elif not is_grace:
                    measure_max = max(measure_max, onset + duration)

            absolute_measure_offset += max(measure_max, 0.0)

    entries.sort(key=lambda item: (item["offset"], not bool(item["grace"]), item["pitch"], item["duration"]))
    deduped = []
    for entry in entries:
        if not deduped or entry["offset"] != deduped[-1]["offset"] or entry["pitch"] != deduped[-1]["pitch"]:
            deduped.append(entry)
        elif bool(deduped[-1]["grace"]) or entry["duration"] > deduped[-1]["duration"]:
            deduped[-1] = entry

    measure_offsets = [entry["measure_offset"] for entry in deduped]
    downbeats = []
    previous = 0.0
    downbeat_min = float(PARAMS["downbeat"]["min"])
    for measure_offset in measure_offsets:
        value = measure_offset - previous
        downbeats.append(value if value > 0 else downbeat_min)
        previous = measure_offset

    return {
        "offset": [entry["offset"] - entry["measure_offset"] for entry in deduped],
        "downbeat": downbeats,
        "duration": [entry["duration"] for entry in deduped],
        "pitch": [entry["pitch"] for entry in deduped],
        "velocity": [entry["velocity"] for entry in deduped],
        "grace": [entry["grace"] for entry in deduped],
        "trill": [entry["trill"] for entry in deduped],
        "staccato": [entry["staccato"] for entry in deduped],
        "hand": [entry["hand"] for entry in deduped],
    }


def parse_mxl_robust(mxl_path: str) -> dict[str, list[float]]:
    mxl = robust_xml_stream(mxl_path)
    entries = robust_xml_note_entries(mxl)
    measure_offsets = [entry["measure_offset"] for entry in entries]
    downbeats = []
    previous = 0.0
    downbeat_min = float(PARAMS["downbeat"]["min"])
    for measure_offset in measure_offsets:
        value = measure_offset - previous
        downbeats.append(value if value > 0 else downbeat_min)
        previous = measure_offset
    return {
        "offset": [entry["offset"] - entry["measure_offset"] for entry in entries],
        "downbeat": downbeats,
        "duration": [entry["duration"] for entry in entries],
        "pitch": [entry["pitch"] for entry in entries],
        "velocity": [entry["velocity"] for entry in entries],
        "grace": [entry["grace"] for entry in entries],
        "trill": [entry["trill"] for entry in entries],
        "staccato": [entry["staccato"] for entry in entries],
        "hand": [entry["hand"] for entry in entries],
    }


def parse_mxl_with_fallback(mxl_path: str) -> dict[str, Any]:
    try:
        return MultistreamTokenizer.parse_mxl(mxl_path)
    except Exception:
        return parse_mxl_robust(mxl_path)


def parse_xml_features(raw_zip: ZipResolver, score_xml_path: str, timeout_sec: float) -> dict[str, Any]:
    suffix = Path(score_xml_path).suffix or ".mxl"
    tmp_path = raw_zip.extract_to_temp(score_xml_path, suffix=suffix)
    try:
        with time_limit(timeout_sec):
            streams = parse_mxl_with_fallback(tmp_path)
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
                clamp_score_grid_value(offsets[idx], PARAMS["offset"]["min"], PARAMS["offset"]["max"]),
                clamp_score_grid_value(durations[idx], PARAMS["duration"]["min"], PARAMS["duration"]["max"]),
                clamp_score_grid_value(ml_raw, 0.0, PARAMS["downbeat"]["max"]),
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
