#!/usr/bin/env python3
"""Build MusicXML score fragments and README image embeds for case studies."""

from __future__ import annotations

import copy
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from music21 import converter
import verovio


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "results" / "case_study"
OUT_DIR = CASE_DIR / "score_fragments"
README = CASE_DIR / "README.md"

RAW_SCORE_PATHS = {
    "beethoven_op26_i": ROOT
    / 'PianoCoRe/raw/Beethoven,_Ludwig_van/Piano_Sonata_No.12_in_A_flat_major,_Op.26_("Funeral_March")/1._Andante_con_variazioni_(A_flat_major)/score.mxl',
    "chopin_scherzo_no2": ROOT
    / "PianoCoRe/raw/Chopin,_Frédéric/Scherzo_No.2_in_B_flat_minor,_Op.31,_B.111/score.musicxml",
    "glinka_the_lark": ROOT
    / "PianoCoRe/raw/Glinka,_Mikhail/A_Farewell_to_Saint_Petersburg/10._The_Lark/score.mxl",
    "haydn_hob_xvi_31_i": ROOT
    / "PianoCoRe/raw/Haydn,_Joseph/Piano_Sonata_No.46,_Keyboard_Sonata_in_E_major,_Hob.XVI:31/1._Moderato_(E_major)/score.musicxml",
    "liszt_mephisto_waltz": ROOT
    / "PianoCoRe/raw/Liszt,_Franz/Mephisto_Waltz_No.1,_S.514/score.musicxml",
}

MEASURE_OVERRIDES = {
    # The paper case starts at the C-F-E gesture and ends at the final repeated E-flat gesture.
    "beethoven_op26_i": (175, 209),
    # Start at the left-hand broken chord / right-hand C-flat+F gesture; end at the D-flat octave.
    "chopin_scherzo_no2": (65, 129),
    "glinka_the_lark": (39, 55),
    "haydn_hob_xvi_31_i": (25, 41),
    "liszt_mephisto_waltz": (737, 791),
}


def read_musicxml(path: Path) -> bytes:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as zf:
            container = ET.fromstring(zf.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None:
                raise RuntimeError(f"Cannot find rootfile in {path}")
            return zf.read(rootfile.attrib["full-path"])
    return path.read_bytes()


def pitched_note_count(measure: ET.Element) -> int:
    count = 0
    for note in measure.findall("note"):
        if note.find("pitch") is not None and note.find("grace") is None:
            count += 1
    return count


def measure_counts_by_number(xml_path: Path) -> list[tuple[int, int]]:
    root = ET.fromstring(read_musicxml(xml_path))
    per_measure: dict[int, int] = {}
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            raw_number = measure.attrib.get("number", "0")
            try:
                number = int(re.sub(r"\D.*$", "", raw_number))
            except ValueError:
                continue
            per_measure[number] = per_measure.get(number, 0) + pitched_note_count(measure)
    return sorted(per_measure.items())


def infer_measure_range(xml_path: Path, note_start: int, note_end: int, pad: int = 1) -> tuple[int, int, int]:
    total = 0
    start_measure = None
    end_measure = None
    counts = measure_counts_by_number(xml_path)
    for number, count in counts:
        prev = total
        total += count
        if start_measure is None and prev <= note_start < total:
            start_measure = number
        if end_measure is None and prev <= note_end < total:
            end_measure = number
            break
    if start_measure is None or end_measure is None:
        raise RuntimeError(
            f"Cannot map note indices {note_start}-{note_end} to measures for {xml_path}; "
            f"counted {total} notes"
        )
    min_measure = min(n for n, _ in counts)
    max_measure = max(n for n, _ in counts)
    return max(min_measure, start_measure - pad), min(max_measure, end_measure + pad), sum(c for _, c in counts)


def extract_fragment(raw_xml: Path, out_xml: Path, start_measure: int, end_measure: int) -> None:
    score = converter.parse(raw_xml)
    fragment = score.measures(start_measure, end_measure)
    if not fragment.parts:
        raise RuntimeError(f"No parts found after extracting measures {start_measure}-{end_measure} from {raw_xml}")
    fragment.metadata = copy.deepcopy(score.metadata)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    fragment.write("musicxml", fp=str(out_xml))


def render_svg_pages(xml_path: Path, out_prefix: Path) -> list[Path]:
    tk = verovio.toolkit()
    tk.setOptions(
        {
            "adjustPageHeight": True,
            "breaks": "auto",
            "footer": "none",
            "header": "none",
            "pageHeight": 2200,
            "pageWidth": 3000,
            "scale": 32,
            "spacingSystem": 7,
        }
    )
    if not tk.loadFile(str(xml_path)):
        raise RuntimeError(f"Verovio failed to load {xml_path}")
    page_count = tk.getPageCount()
    pages = []
    for page_no in range(1, page_count + 1):
        svg = tk.renderToSVG(page_no)
        svg_path = out_prefix.with_name(f"{out_prefix.name}_p{page_no:02d}.svg")
        svg_path.write_text(svg, encoding="utf-8")
        pages.append(svg_path)
    return pages


def img_block(key: str, title: str, pages: list[Path], start_measure: int, end_measure: int) -> str:
    lines = [
        f"<!-- score-fragment: {key} -->",
        "",
        f"<p><strong>Score fragment, measures {start_measure}-{end_measure}</strong></p>",
        "",
    ]
    for i, page in enumerate(pages, 1):
        rel = page.relative_to(CASE_DIR).as_posix()
        lines.append(
            f'<p><img src="{rel}" alt="{title} score fragment page {i}" width="100%"></p>'
        )
        lines.append("")
    lines.append(f"<!-- /score-fragment: {key} -->")
    return "\n".join(lines)


def replace_readme_blocks(blocks: dict[str, str], titles: dict[str, str]) -> None:
    text = README.read_text(encoding="utf-8")
    for key, block in blocks.items():
        pattern = re.compile(
            rf"\n?<!-- score-fragment: {re.escape(key)} -->.*?<!-- /score-fragment: {re.escape(key)} -->\n?",
            flags=re.S,
        )
        text = pattern.sub("\n", text)
        heading = f"### {titles[key]}"
        pos = text.find(heading)
        if pos < 0:
            raise RuntimeError(f"Cannot find README heading: {heading}")
        line_end = text.find("\n", pos)
        text = text[: line_end + 1] + "\n" + block + "\n" + text[line_end + 1 :]
    README.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_out = []
    blocks = {}
    titles = {}

    for manifest_path in sorted(CASE_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = manifest["key"]
        title = manifest["title"]
        if key not in RAW_SCORE_PATHS:
            continue
        note_start, note_end = manifest["aligned_note_index_range_raw"]
        raw_xml = RAW_SCORE_PATHS[key]
        if key in MEASURE_OVERRIDES:
            start_measure, end_measure = MEASURE_OVERRIDES[key]
            xml_note_count = sum(count for _, count in measure_counts_by_number(raw_xml))
        else:
            start_measure, end_measure, xml_note_count = infer_measure_range(raw_xml, note_start, note_end)
        frag_xml = OUT_DIR / f"{key}_m{start_measure:03d}_{end_measure:03d}.musicxml"
        extract_fragment(raw_xml, frag_xml, start_measure, end_measure)
        pages = render_svg_pages(frag_xml, OUT_DIR / f"{key}_m{start_measure:03d}_{end_measure:03d}")
        blocks[key] = img_block(key, title, pages, start_measure, end_measure)
        titles[key] = title
        manifest_out.append(
            {
                "key": key,
                "title": title,
                "raw_score": str(raw_xml.relative_to(ROOT)),
                "fragment_musicxml": str(frag_xml.relative_to(ROOT)),
                "svg_pages": [str(p.relative_to(ROOT)) for p in pages],
                "aligned_note_index_range_raw": [note_start, note_end],
                "xml_pitched_note_count": xml_note_count,
                "measure_range_with_padding": [start_measure, end_measure],
            }
        )

    replace_readme_blocks(blocks, titles)
    (OUT_DIR / "score_fragments_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for item in manifest_out:
        print(
            f"{item['key']}: measures {item['measure_range_with_padding'][0]}-"
            f"{item['measure_range_with_padding'][1]}, pages={len(item['svg_pages'])}"
        )


if __name__ == "__main__":
    main()
