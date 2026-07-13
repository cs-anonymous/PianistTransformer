#!/usr/bin/env python
import argparse
import copy
import json
import math
import shutil
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from miditoolkit import MidiFile


REPAIRS = {
    "Albéniz,_Isaac/Recuerdos_de_viaje,_Op.71/6._Rumores_de_la_Caleta_-_Malagueña/score.mxl": {
        "directions": {150: 60.0, 207: 87.0},
    },
    "Schumann,_Robert/Arabeske,_Op.18/score.musicxml": {
        "directions": {330: 116.0},
    },
    "Schumann,_Robert/Kreisleriana,_Op.16/6._Sehr_langsam/score.musicxml": {
        "directions": {3: 42.0},
    },
    "Schumann,_Robert/Waldszenen,_Op.82/score.mxl": {
        "directions": {
            0: 132.0,
            60: 156.0,
            150: 96.0,
            215: 60.0,
            325: 160.0,
            457: 132.0,
            550: 63.0,
            661: 180.0,
            731: 80.0,
        },
    },
}

MIDI_REPAIRS = {
    "Albéniz,_Isaac/Recuerdos_de_viaje,_Op.71/6._Rumores_de_la_Caleta_-_Malagueña/score.mxl": {
        50400: 60.0,
        56400: 60.0,
        70800: 87.0,
    },
    "Schumann,_Robert/Arabeske,_Op.18/score.musicxml": {
        "tempo_value": (29.0, 116.0),
    },
    "Schumann,_Robert/Kreisleriana,_Op.16/6._Sehr_langsam/score.musicxml": {
        "tempo_value": (168.0, 42.0),
    },
    "Schumann,_Robert/Waldszenen,_Op.82/score.mxl": {
        "tempo_event_indices_full": {
            0: 132.0, 1: 156.0, 2: 96.0, 3: 60.0, 4: 160.0,
            9: 132.0, 16: 63.0, 19: 180.0, 35: 80.0,
        },
        "tempo_event_indices_mini": {
            0: 132.0, 1: 156.0, 2: 96.0, 3: 60.0, 4: 160.0,
            9: 132.0, 16: 63.0, 19: 180.0, 20: 80.0,
        },
    },
}


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def read_xml_payload(path):
    if path.suffix.lower() != ".mxl" or not zipfile.is_zipfile(path):
        return ET.parse(path), None, None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        candidates = [
            name for name in names
            if name.lower().endswith((".xml", ".musicxml")) and "meta-inf/" not in name.lower()
        ]
        candidates.sort(key=lambda name: (name.count("/"), len(name), name))
        if not candidates:
            raise ValueError(f"No score XML member in {path}")
        member = candidates[0]
        members = {name: archive.read(name) for name in names}
    return ET.ElementTree(ET.fromstring(members[member])), member, members


def write_xml_payload(tree, path, member, members):
    ET.indent(tree, space="  ")
    xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if member is None:
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return
    members[member] = xml_bytes
    tmp = path.with_name(path.name + ".tempo-repair.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    tmp.replace(path)


def repair_xml(path, spec):
    tree, member, members = read_xml_payload(path)
    directions = [node for node in tree.getroot().iter() if local_name(node.tag) == "direction"]
    changes = []
    if "all_sound_scale" in spec:
        scale = float(spec["all_sound_scale"])
        for sound in (node for node in tree.getroot().iter() if local_name(node.tag) == "sound"):
            if "tempo" in sound.attrib:
                old = float(sound.attrib["tempo"])
                sound.attrib["tempo"] = format(old * scale, ".12g")
                changes.append(("all", old, old * scale))
    for index, new_tempo in spec.get("directions", {}).items():
        sounds = [node for node in directions[index].iter() if local_name(node.tag) == "sound" and "tempo" in node.attrib]
        if len(sounds) != 1:
            raise ValueError(f"Expected one tempo sound at direction {index} in {path}, got {len(sounds)}")
        old = float(sounds[0].attrib["tempo"])
        sounds[0].attrib["tempo"] = format(float(new_tempo), ".12g")
        changes.append((index, old, float(new_tempo)))
    write_xml_payload(tree, path, member, members)
    return changes


def repair_midi(path, spec):
    midi = MidiFile(str(path))
    changes = []
    if "tempo_event_indices_full" in spec:
        index_map = spec["tempo_event_indices_mini"] if "_mini" in path.stem else spec["tempo_event_indices_full"]
        for index, new_tempo in index_map.items():
            event = midi.tempo_changes[index]
            old = float(event.tempo)
            event.tempo = float(new_tempo)
            changes.append((int(event.time), old, event.tempo))
    elif "all_tempo_scale" in spec:
        scale = float(spec["all_tempo_scale"])
        for event in midi.tempo_changes:
            old = float(event.tempo)
            event.tempo = old * scale
            changes.append((int(event.time), old, event.tempo))
    elif "tempo_value" in spec:
        old_target, new_tempo = spec["tempo_value"]
        for event in midi.tempo_changes:
            if math.isclose(float(event.tempo), float(old_target), rel_tol=1e-4, abs_tol=1e-3):
                old = float(event.tempo)
                event.tempo = float(new_tempo)
                changes.append((int(event.time), old, event.tempo))
    else:
        for tick, new_tempo in spec.items():
            matches = [event for event in midi.tempo_changes if int(event.time) == int(tick)]
            if len(matches) != 1:
                raise ValueError(f"Expected one tempo event at tick {tick} in {path}, got {len(matches)}")
            old = float(matches[0].tempo)
            if not math.isclose(old, float(new_tempo), rel_tol=1e-4, abs_tol=1e-3):
                matches[0].tempo = float(new_tempo)
                changes.append((int(tick), old, matches[0].tempo))
    if not changes:
        raise ValueError(f"No MIDI tempo changes applied to {path}")
    midi.dump(str(path))
    return changes


def backup_file(path, source_root, backup_root, prefix):
    relative = path.relative_to(source_root)
    destination = backup_root / prefix / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("../PianoCoRe/raw"))
    parser.add_argument("--refined-root", type=Path, default=Path("../PianoCoRe/refined"))
    parser.add_argument("--processed-root", type=Path, default=Path("../PianoCoRe/processed"))
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--score-xml-path", action="append", default=None)
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata)
    report = {"xml": [], "midi": []}
    for xml_rel, xml_spec in REPAIRS.items():
        if args.score_xml_path and xml_rel not in set(args.score_xml_path):
            continue
        for prefix, root in (("raw", args.raw_root), ("processed", args.processed_root)):
            path = root / xml_rel
            if not path.exists():
                continue
            backup_file(path, root, args.backup_root, prefix)
            changes = repair_xml(path, xml_spec)
            report["xml"].append({"path": str(path), "changes": changes})
        rows = metadata[metadata["score_xml_path"] == xml_rel]
        for kind, column, root in (
            ("raw", "score_midi_path", args.raw_root),
            ("refined", "refined_score_midi_path", args.refined_root),
        ):
            for midi_rel in sorted(rows[column].dropna().unique()):
                path = root / midi_rel
                backup_file(path, root, args.backup_root, kind)
                changes = repair_midi(path, MIDI_REPAIRS[xml_rel])
                report["midi"].append({"path": str(path), "changes": changes})
    output = args.backup_root / "repair_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"xml_files": len(report["xml"]), "midi_files": len(report["midi"]), "report": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
