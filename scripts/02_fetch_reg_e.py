#!/usr/bin/env python
"""Step 2 - Download Regulation E (12 CFR Part 1005) from eCFR and chunk it.

  python scripts/02_fetch_reg_e.py                       # fetch latest version from eCFR
  python scripts/02_fetch_reg_e.py --date 2026-10-01     # a specific point-in-time version
  python scripts/02_fetch_reg_e.py --local-xml data/fixtures/reg_e_fixture.xml   # offline / smoke test

Output: data/raw/reg_e.xml, data/processed/reg_e_chunks.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, REPO_ROOT, path  # noqa: E402
from triage.reg_e import chunk_sections, fetch_part_1005_xml, parse_sections, save_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="eCFR point-in-time date YYYY-MM-DD (default: latest)")
    ap.add_argument("--local-xml", type=Path, help="parse a local eCFR XML file instead of downloading")
    args = ap.parse_args()

    raw_dir, processed_dir = path("raw_dir"), path("processed_dir")

    if args.local_xml is None and CFG.get("smoke", {}).get("tiny_models"):
        args.local_xml = REPO_ROOT / "data/fixtures/reg_e_fixture.xml"

    if args.local_xml:
        xml_text = Path(args.local_xml).read_text(encoding="utf-8")
        print(f"parsed local XML {args.local_xml}")
    else:
        xml_text = fetch_part_1005_xml(args.date)
        (raw_dir / "reg_e.xml").write_text(xml_text, encoding="utf-8")
        print(f"downloaded {len(xml_text):,} chars -> {raw_dir / 'reg_e.xml'}")

    sections = parse_sections(xml_text)
    chunks = chunk_sections(sections, CFG["retrieval"]["chunk_chars"])
    out = processed_dir / "reg_e_chunks.jsonl"
    save_chunks(chunks, out)
    print(f"{len(sections)} sections -> {len(chunks)} chunks -> {out}")
    for s in sections[:5]:
        print("  ", s["heading"][:90])


if __name__ == "__main__":
    main()
