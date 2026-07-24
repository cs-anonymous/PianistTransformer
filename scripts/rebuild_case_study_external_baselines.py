#!/usr/bin/env python3
"""Rebuild case-study crops for external baselines.

This script intentionally aligns external outputs by musical content instead of
reusing raw score-note indices. DExter and VirtuosoNet can add timing jitter or
use a different XML/MIDI note ordering, so their MIDI note index is not a stable
proxy for the score index.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pretty_midi


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "results" / "case_study"
DEXTER_MANIFEST = ROOT / "backup/results0722/external_eval_20260718_asap_processed/dexter/prediction_manifest.json"
VNET_MANIFEST = ROOT / "backup/results0722/external_eval_20260718_asap_processed/virtuosonet/prediction_manifest.json"


def all_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    return sorted(
        [note for inst in midi.instruments for note in inst.notes],
        key=lambda note: (note.start, note.pitch, note.end),
    )


def onset_groups(notes: list[pretty_midi.Note], eps: float) -> list[tuple[float, list[int]]]:
    groups: list[tuple[float, list[int]]] = []
    current: list[pretty_midi.Note] = []
    group_start = None
    last_start = None
    for note in sorted(notes, key=lambda n: (n.start, n.pitch, n.end)):
        if group_start is None or note.start - float(last_start) <= eps:
            current.append(note)
            if group_start is None:
                group_start = note.start
            last_start = note.start
        else:
            groups.append((float(group_start), sorted({n.pitch for n in current})))
            current = [note]
            group_start = note.start
            last_start = note.start
    if current:
        groups.append((float(group_start), sorted({n.pitch for n in current})))
    return groups


def jaccard(a: list[int], b: list[int]) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / len(aa | bb) if aa or bb else 1.0


def best_group_match(
    target: list[tuple[float, list[int]]],
    candidate: list[tuple[float, list[int]]],
    *,
    k: int = 12,
    min_group_index: int = 0,
) -> tuple[int, float]:
    best_index = -1
    best_score = -1.0
    if len(target) < k or len(candidate) < k:
        return best_index, best_score
    for index in range(min_group_index, len(candidate) - k + 1):
        score = sum(jaccard(target[offset][1], candidate[index + offset][1]) for offset in range(k))
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def exact_pitch_match(target_pitches: list[int], candidate_notes: list[pretty_midi.Note]) -> int:
    candidate_pitches = [note.pitch for note in candidate_notes]
    k = len(target_pitches)
    for index in range(0, len(candidate_pitches) - k + 1):
        if candidate_pitches[index : index + k] == target_pitches:
            return index
    return -1


def find_boundaries(
    score_notes: list[pretty_midi.Note],
    candidate_notes: list[pretty_midi.Note],
    start_index: int,
    end_index: int,
    method_tag: str,
) -> tuple[float, float, dict]:
    target_notes = score_notes[start_index : end_index + 1]
    if not target_notes:
        raise ValueError("empty target note span")

    # VirtuosoNet often preserves note order when using the processed XML path,
    # so an exact pitch sequence gives a cleaner boundary than fuzzy grouping.
    if method_tag == "virtuosonet_han":
        k = min(24, len(target_notes))
        note_start = exact_pitch_match([n.pitch for n in target_notes[:k]], candidate_notes)
        tail_start = max(0, note_start) + max(1, len(target_notes) // 2)
        tail_match = -1
        tail_k = min(24, len(target_notes))
        tail_pattern = [n.pitch for n in target_notes[-tail_k:]]
        candidate_pitches = [n.pitch for n in candidate_notes]
        if note_start >= 0:
            for idx in range(tail_start, len(candidate_pitches) - tail_k + 1):
                if candidate_pitches[idx : idx + tail_k] == tail_pattern:
                    tail_match = idx
                    break
        if note_start >= 0 and tail_match >= 0:
            start_time = candidate_notes[note_start].start
            end_note = candidate_notes[tail_match + tail_k - 1]
            return start_time, end_note.start, {
                "alignment_method": "exact pitch-sequence match on processed external output",
                "start_note_index": note_start,
                "end_note_index": tail_match + tail_k - 1,
                "start_match_len": k,
                "end_match_len": tail_k,
            }

    target_groups = onset_groups(target_notes, eps=0.005)
    candidate_groups = onset_groups(candidate_notes, eps=0.025)
    head_k = min(12, len(target_groups))
    tail_k = min(12, len(target_groups))
    start_group, head_score = best_group_match(target_groups[:head_k], candidate_groups, k=head_k)
    min_tail_group = max(0, start_group + len(target_groups) // 2) if start_group >= 0 else 0
    end_group, tail_score = best_group_match(target_groups[-tail_k:], candidate_groups, k=tail_k, min_group_index=min_tail_group)
    if start_group < 0 or end_group < 0:
        raise ValueError("could not find grouped onset match")
    start_time = candidate_groups[start_group][0]
    end_time = candidate_groups[end_group + tail_k - 1][0]
    if method_tag == "dexter":
        # DExter often jitters/chord-spreads onsets enough that fuzzy tail
        # matching can jump to a later repetition. Once the musical head is
        # found, the target note count gives a more stable segment end.
        notes_after_start = [note for note in candidate_notes if note.start >= start_time - 1e-6]
        target_count = len(target_notes)
        if len(notes_after_start) >= target_count:
            end_time = notes_after_start[target_count - 1].start
        else:
            end_time = notes_after_start[-1].start
    return start_time, end_time, {
        "alignment_method": "fuzzy onset-group pitch match on processed external output",
        "start_group_index": start_group,
        "end_group_index": end_group + tail_k - 1,
        "start_group_score": head_score,
        "end_group_score": tail_score,
        "start_match_groups": head_k,
        "end_match_groups": tail_k,
    }


def crop_midi(source: Path, output: Path, start: float, end: float) -> dict:
    midi = pretty_midi.PrettyMIDI(str(source))
    cropped = pretty_midi.PrettyMIDI(initial_tempo=120)
    note_count = 0
    cc_count = 0
    for inst in midi.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=inst.name,
        )
        for note in inst.notes:
            if start <= note.start <= end:
                new_note = deepcopy(note)
                new_note.start = max(0.0, note.start - start)
                new_note.end = max(new_note.start + 0.001, note.end - start)
                new_inst.notes.append(new_note)
                note_count += 1
        for cc in inst.control_changes:
            if start <= cc.time <= end:
                new_cc = deepcopy(cc)
                new_cc.time = max(0.0, cc.time - start)
                new_inst.control_changes.append(new_cc)
                cc_count += 1
        if new_inst.notes or new_inst.control_changes:
            cropped.instruments.append(new_inst)
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped.write(str(output))
    return {
        "notes": note_count,
        "control_changes": cc_count,
        "duration_s": pretty_midi.PrettyMIDI(str(output)).get_end_time(),
    }


def load_manifest_items(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)["items"]


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if resolved.exists():
        return resolved
    backup_path = Path(str(resolved).replace("/results/", "/backup/results0722/"))
    if backup_path.exists():
        return backup_path
    if not resolved.is_absolute():
        candidate = ROOT / resolved
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def item_for_case(items: list[dict], score_source: str) -> dict:
    suffix = score_source.replace("PianoCoRe/refined/", "")
    matches = [item for item in items if item["score_midi"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest match for {score_source}, found {len(matches)}")
    return matches[0]


def main() -> None:
    top_manifest_path = CASE_ROOT / "manifest.json"
    with top_manifest_path.open() as handle:
        cases = json.load(handle)
    manifests = {
        "dexter": load_manifest_items(DEXTER_MANIFEST),
        "virtuosonet_han": load_manifest_items(VNET_MANIFEST),
    }
    labels = {
        "dexter": ("DExter", "dexter"),
        "virtuosonet_han": ("VirtuosoNet-Han", "virtuosonet_han_gru"),
    }

    summary: list[dict] = []
    for case in cases:
        score_record = next(record for record in case["outputs"] if record["kind"] == "score")
        score_path = resolve_path(score_record["source"])
        score_notes = all_notes(pretty_midi.PrettyMIDI(str(score_path)))
        start_index, end_index = case["aligned_note_index_range_raw"]
        case_baselines = [
            record
            for record in case.get("baseline_outputs", [])
            if record.get("method_tag") not in {"dexter", "virtuosonet_han"}
        ]
        for method_tag in ("dexter", "virtuosonet_han"):
            item = item_for_case(manifests[method_tag], score_record["source"])
            label, protocol = labels[method_tag]
            for sample_no, source_string in enumerate(item["prediction_paths"][:2]):
                source = resolve_path(source_string)
                candidate_notes = all_notes(pretty_midi.PrettyMIDI(str(source)))
                start_time, end_time, align_info = find_boundaries(
                    score_notes, candidate_notes, start_index, end_index, method_tag
                )
                if end_time <= start_time:
                    raise ValueError(f"bad crop boundary for {case['key']} {method_tag} sample {sample_no}")
                segment = case["gt000_user_segment_s"]
                output = CASE_ROOT / case["key"] / (
                    f"{method_tag}_sample_{sample_no:03d}_{int(segment[0])}_{int(segment[1])}s_aligned.mid"
                )
                stats = crop_midi(source, output, start_time, end_time)
                record = {
                    "kind": "baseline",
                    "method_tag": method_tag,
                    "method_label": label,
                    "protocol": protocol,
                    "sample_id": f"sample_{sample_no:03d}",
                    "source": str(source.relative_to(ROOT)),
                    "output": str(output.relative_to(ROOT)),
                    "crop_time_s": [start_time, end_time],
                    "boundary_method": align_info["alignment_method"],
                    "aligned_note_index_range_raw": [start_index, end_index],
                    **align_info,
                    **stats,
                }
                case_baselines.append(record)
                summary.append({"case": case["key"], **record})
        case["baseline_outputs"] = case_baselines
        with (CASE_ROOT / case["key"] / "manifest.json").open("w") as handle:
            json.dump(case, handle, indent=2, ensure_ascii=False)

    with top_manifest_path.open("w") as handle:
        json.dump(cases, handle, indent=2, ensure_ascii=False)
    with (CASE_ROOT / "external_baseline_realign_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    for item in summary:
        print(
            item["case"],
            item["method_tag"],
            item["sample_id"],
            "crop",
            [round(x, 3) for x in item["crop_time_s"]],
            "dur",
            round(item["duration_s"], 3),
            "notes",
            item["notes"],
        )


if __name__ == "__main__":
    main()
