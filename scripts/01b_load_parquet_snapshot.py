#!/usr/bin/env python
"""Step 1 (alternative) - Build the balanced complaint dataset from a Parquet
SNAPSHOT of the CFPB Consumer Complaint Database.

Why this script exists
----------------------
On 2026-08-14 the CFPB stopped publishing consumer complaint narratives. The
live search API and the current complaints.csv.zip no longer contain the
"Consumer complaint narrative" column, so scripts/01_download_cfpb.py cannot
obtain training text any more. Snapshots taken before that date still have the
narratives. One that is easy to download is the Hugging Face dataset
`Mouwiya/cfpb-consumer-complaints` (about 16.6M rows, Parquet, snake_case
column names, snapshot taken around January 2026):

    hf download Mouwiya/cfpb-consumer-complaints --repo-type dataset --local-dir D:\\data\\cfpb_parquet
    # older huggingface_hub: huggingface-cli download ... (same arguments)

Then, from the repository root with the main environment activated:

    python scripts/01b_load_parquet_snapshot.py --parquet-dir D:\\data\\cfpb_parquet
    python scripts/01b_load_parquet_snapshot.py --parquet-dir D:\\data\\cfpb_parquet --n-per-class 500

Output is identical to step 1: data/processed/complaints{,_train,_test}.parquet.
Any Parquet export of the CFPB database works as long as it has a narrative,
a product and a company-response column; column names are matched loosely
(snake_case, original CSV headers, or "Complaint Text" style).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, path  # noqa: E402
from triage.data_io import balanced_sample, clean, normalise, save_splits  # noqa: E402

# normalised column name (lower-case, alphanumerics only) -> canonical name used by triage.data_io
ALIASES: dict[str, str] = {
    "complaintid": "complaint_id",
    "datereceived": "date_received",
    "product": "product",
    "subproduct": "sub_product",
    "issue": "issue",
    "subissue": "sub_issue",
    "consumercomplaintnarrative": "narrative",
    "complainttext": "narrative",
    "narrative": "narrative",
    "company": "company",
    "state": "state",
    "tags": "tags",
    "submittedvia": "submitted_via",
    "companyresponsetoconsumer": "company_response",
    "companyresponse": "company_response",
    "timelyresponse": "timely",
    "consumerdisputed": "consumer_disputed",
}
REQUIRED = {"narrative", "product", "company_response"}
MIN_CHARS = 80  # same threshold as triage.data_io.clean


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def column_mapping(schema_names: list[str]) -> dict[str, str]:
    """Map the file's actual column names to canonical names."""
    mapping: dict[str, str] = {}
    for col in schema_names:
        canon = ALIASES.get(_norm(col))
        if canon and canon not in mapping.values():
            mapping[col] = canon
    missing = REQUIRED - set(mapping.values())
    if missing:
        raise SystemExit(
            f"parquet files lack required column(s) {sorted(missing)}.\n"
            f"columns found: {schema_names}\n"
            "If the snapshot was taken after 2026-08-14 it has no narratives; use an earlier snapshot."
        )
    return mapping


def iter_frames(files: list[Path], mapping: dict[str, str], batch_size: int):
    """Yield canonical DataFrames (narrative present, >= MIN_CHARS) batch by batch."""
    inv = {v: k for k, v in mapping.items()}
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in mapping if c in pf.schema_arrow.names]
        n_file = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            chunk = batch.to_pandas()
            narr = chunk[inv["narrative"]]
            chunk = chunk[narr.notna() & (narr.astype(str).str.len() >= MIN_CHARS)]
            if chunk.empty:
                continue
            if "date_received" in inv:
                d = pd.to_datetime(chunk[inv["date_received"]], errors="coerce")
                chunk = chunk.assign(**{inv["date_received"]: d.dt.strftime("%Y-%m-%d").fillna("")})
            part = normalise(chunk, mapping)
            part = part[part["product_group"].notna()]
            n_file += len(part)
            yield part
        print(f"  {f.name}: {n_file:,} usable rows (narrative + known product group)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", type=Path, required=True, help="folder containing the snapshot's .parquet files (searched recursively)")
    ap.add_argument("--n-per-class", type=int, default=CFG["data"]["n_per_class"])
    ap.add_argument("--date-min", default=CFG["data"]["date_min"], help="keep complaints received on/after this date (YYYY-MM-DD)")
    ap.add_argument("--batch-size", type=int, default=100_000)
    args = ap.parse_args()

    seed, test_size = CFG["data"]["seed"], CFG["data"]["test_size"]
    processed_dir = path("processed_dir")

    files = sorted(p for p in args.parquet_dir.rglob("*.parquet") if p.is_file())
    if not files:
        raise SystemExit(f"no .parquet files found under {args.parquet_dir}")
    mapping = column_mapping(pq.ParquetFile(files[0]).schema_arrow.names)
    print(f"{len(files)} parquet file(s); column mapping: {mapping}")

    # Accumulate rows, but keep memory bounded: whenever the pool grows past
    # cap_total, re-sample it to at most `cap_per_group` rows per product group.
    cap_per_group = max(args.n_per_class * 20, 5_000)
    cap_total = cap_per_group * 9 * 2
    pool: pd.DataFrame | None = None
    for part in iter_frames(files, mapping, args.batch_size):
        part = part[part["date_received"].fillna("") >= args.date_min] if "date_received" in mapping.values() else part
        if part.empty:
            continue
        pool = part if pool is None else pd.concat([pool, part], ignore_index=True)
        if len(pool) > cap_total:
            pool = balanced_sample(pool, cap_per_group, seed)

    if pool is None or pool.empty:
        raise SystemExit(f"no rows with narratives on/after {args.date_min}; check --date-min or the snapshot")

    print(f"pooled {len(pool):,} rows with narratives on/after {args.date_min}")
    df = clean(pool)
    df = balanced_sample(df, args.n_per_class, seed)
    save_splits(df, processed_dir, test_size, seed)


if __name__ == "__main__":
    main()
