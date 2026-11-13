#!/usr/bin/env python
"""Step 1 - Build a balanced complaint dataset (train/test parquet files).

Three sources, pick one:

  python scripts/01_download_cfpb.py                 # CFPB public API (default, ~1-2k rows per product group)
  python scripts/01_download_cfpb.py --from-csv complaints.csv   # full CSV download from files.consumerfinance.gov
  python scripts/01_download_cfpb.py --synthetic     # templated fake data for the smoke test

Output: data/processed/complaints{,_train,_test}.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, path  # noqa: E402
from triage.data_io import API_MAP, CSV_MAP, balanced_sample, clean, normalise, save_splits  # noqa: E402
from triage.taxonomy import PRODUCT_GROUPS  # noqa: E402

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
PAGE = 500   # rows per API request


def fetch_product(raw_product: str, date_min: str, want: int) -> list[dict]:
    """Page through the CFPB search API for one raw product name."""
    rows: list[dict] = []
    frm = 0
    while len(rows) < want and frm < 10_000:  # ES result window limit
        params = {
            "product": raw_product,
            "has_narrative": "true",
            "date_received_min": date_min,
            "size": PAGE,
            "frm": frm,
            "sort": "created_date_desc",
            "format": "json",
            "no_aggs": "true",
            "no_highlight": "true",
        }
        for attempt in range(4):
            try:
                r = requests.get(API_URL, params=params, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:  # network hiccups are common on this API
                wait = 5 * (attempt + 1)
                print(f"    retry {attempt + 1} after error: {e} (sleep {wait}s)")
                time.sleep(wait)
        else:
            print(f"    giving up on {raw_product} at frm={frm}")
            break
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            break
        rows.extend(h["_source"] for h in hits)
        frm += PAGE
        time.sleep(1.0)  # be polite
    return rows


def from_api(n_per_class: int, date_min: str) -> pd.DataFrame:
    frames = []
    for group, raw_names in PRODUCT_GROUPS.items():
        want = int(n_per_class * 1.5)  # over-fetch, then sample
        got = 0
        for raw in raw_names:
            if got >= want:
                break
            print(f"[{group}] fetching '{raw}' ...")
            rows = fetch_product(raw, date_min, want - got)
            print(f"    {len(rows)} rows")
            if rows:
                frames.append(pd.DataFrame(rows))
                got += len(rows)
    df = pd.concat(frames, ignore_index=True)
    return normalise(df, API_MAP)


def from_csv(csv_path: Path, date_min: str, n_per_class: int, seed: int) -> pd.DataFrame:
    """Stream the full CSV (several GB uncompressed) in chunks, keep rows with narratives."""
    keep = []
    usecols = list(CSV_MAP.keys())
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=200_000, low_memory=False)):
        chunk = chunk[chunk["Consumer complaint narrative"].notna()]
        chunk = chunk[chunk["Date received"] >= date_min]
        keep.append(normalise(chunk, CSV_MAP))
        print(f"  chunk {i}: kept {sum(len(k) for k in keep)} rows so far")
    df = pd.concat(keep, ignore_index=True)
    # Sub-sample early to keep memory small: 5x what we need per group.
    return balanced_sample(clean(df), n_per_class * 5, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", type=Path, help="path to complaints.csv from files.consumerfinance.gov")
    ap.add_argument("--synthetic", action="store_true", help="generate templated fake data (smoke test)")
    ap.add_argument("--n-per-class", type=int, default=CFG["data"]["n_per_class"])
    args = ap.parse_args()

    date_min, seed, test_size = CFG["data"]["date_min"], CFG["data"]["seed"], CFG["data"]["test_size"]
    raw_dir, processed_dir = path("raw_dir"), path("processed_dir")

    if args.synthetic or CFG.get("smoke", {}).get("tiny_models"):
        from triage.synthetic import make_synthetic_complaints
        df = make_synthetic_complaints(args.n_per_class, seed)
        df["product_group"] = df["product"].map(lambda p: next(g for g, raws in PRODUCT_GROUPS.items() if p in raws))
        print(f"synthetic data: {len(df)} rows")
    elif args.from_csv:
        df = from_csv(args.from_csv, date_min, args.n_per_class, seed)
    else:
        df = from_api(args.n_per_class, date_min)
        df.to_parquet(raw_dir / "cfpb_api_raw.parquet", index=False)

    df = clean(df)
    df = balanced_sample(df, args.n_per_class, seed)
    save_splits(df, processed_dir, test_size, seed)


if __name__ == "__main__":
    main()
