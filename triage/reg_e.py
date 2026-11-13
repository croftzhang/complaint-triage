"""Fetch 12 CFR Part 1005 (Regulation E) from the eCFR API and split it into
section-level chunks for retrieval.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

ECFR_TITLES = "https://www.ecfr.gov/api/versioner/v1/titles.json"
ECFR_FULL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-12.xml"

SECTION_RE = re.compile(r"§\s*(1005\.\d+)")      # citations in free text, e.g. "§ 1005.6"
N_RE = re.compile(r"(1005\.\d+)")                # eCFR DIV8 N attribute is "1005.6" (no § sign)


@dataclass
class Chunk:
    id: str
    section: str        # e.g. "1005.6"
    heading: str        # e.g. "§ 1005.6 Liability of consumer for unauthorized transfers."
    chunk_idx: int
    text: str


def latest_title12_date() -> str:
    r = requests.get(ECFR_TITLES, timeout=60)
    r.raise_for_status()
    for t in r.json()["titles"]:
        if t["number"] == 12:
            return t["up_to_date_as_of"]
    raise RuntimeError("title 12 not found in eCFR titles.json")


def fetch_part_1005_xml(date: str | None = None) -> str:
    date = date or latest_title12_date()
    r = requests.get(ECFR_FULL.format(date=date), params={"part": "1005"}, timeout=120)
    r.raise_for_status()
    return r.text


def _text(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_sections(xml_text: str) -> list[dict]:
    """Return [{section, heading, paragraphs:[...]}] for every <DIV8 TYPE="SECTION">."""
    root = ET.fromstring(xml_text)
    sections = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        m = N_RE.search(div.get("N", ""))
        if not m:
            continue
        head_el = div.find("HEAD")
        heading = _text(head_el) if head_el is not None else div.get("N", "")
        paragraphs = [_text(p) for p in div.iter("P")]
        paragraphs = [p for p in paragraphs if p]
        sections.append({"section": m.group(1), "heading": heading, "paragraphs": paragraphs})
    return sections


def chunk_sections(sections: list[dict], chunk_chars: int = 1200) -> list[Chunk]:
    """Greedy paragraph packing: paragraphs are appended to a chunk until it
    exceeds chunk_chars. Every chunk is prefixed with its section heading so the
    embedding carries the section context.
    """
    chunks: list[Chunk] = []
    for s in sections:
        buf: list[str] = []
        size = 0
        idx = 0

        def flush():
            nonlocal buf, size, idx
            if buf:
                text = s["heading"] + "\n" + "\n".join(buf)
                chunks.append(Chunk(id=f"{s['section']}#{idx}", section=s["section"], heading=s["heading"], chunk_idx=idx, text=text))
                idx += 1
                buf, size = [], 0

        for p in s["paragraphs"]:
            if size + len(p) > chunk_chars and buf:
                flush()
            buf.append(p)
            size += len(p)
        flush()
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def sections_in_text(text: str) -> set[str]:
    """Extract '1005.x' citations from free text."""
    return set(SECTION_RE.findall(text or ""))
