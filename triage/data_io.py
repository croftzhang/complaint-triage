"""Normalise CFPB records (API JSON or CSV download) into one canonical schema
and build a balanced, stratified train/test split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .taxonomy import LABELS, to_group

CANONICAL = [
    "complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
    "narrative", "company", "state", "tags", "submitted_via", "company_response",
    "timely", "consumer_disputed",
]

# API field names  -> canonical
API_MAP = {
    "complaint_id": "complaint_id",
    "date_received": "date_received",
    "product": "product",
    "sub_product": "sub_product",
    "issue": "issue",
    "sub_issue": "sub_issue",
    "complaint_what_happened": "narrative",
    "company": "company",
    "state": "state",
    "tags": "tags",
    "submitted_via": "submitted_via",
    "company_response": "company_response",
    "timely": "timely",
    "consumer_disputed": "consumer_disputed",
}

# CSV header names (files.consumerfinance.gov/ccdb/complaints.csv.zip) -> canonical
CSV_MAP = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Company response to consumer": "company_response",
    "Timely response?": "timely",
    "Consumer disputed?": "consumer_disputed",
}


def normalise(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.rename(columns=mapping)
    for col in CANONICAL:
        if col not in out.columns:
            out[col] = None
    out = out[CANONICAL].copy()
    out["complaint_id"] = out["complaint_id"].astype(str)
    out["narrative"] = out["narrative"].fillna("").astype(str).str.strip()
    out["product_group"] = out["product"].map(to_group)
    return out


def clean(df: pd.DataFrame, min_chars: int = 80) -> pd.DataFrame:
    df = df[df["product_group"].notna()]
    df = df[df["narrative"].str.len() >= min_chars]
    df = df.drop_duplicates(subset="complaint_id")
    df = df.drop_duplicates(subset="narrative")
    return df.reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for label in LABELS:
        sub = df[df["product_group"] == label]
        if len(sub) == 0:
            print(f"  warning: no rows for group '{label}'")
            continue
        parts.append(sub.sample(n=min(n_per_class, len(sub)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test_idx = []
    for label, sub in df.groupby("product_group"):
        n_test = max(1, int(round(len(sub) * test_size)))
        test_idx.extend(rng.choice(sub.index.to_numpy(), size=n_test, replace=False).tolist())
    test_mask = df.index.isin(test_idx)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def save_splits(df: pd.DataFrame, processed_dir: Path, test_size: float, seed: int) -> None:
    train, test = stratified_split(df, test_size, seed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / "complaints.parquet", index=False)
    train.to_parquet(processed_dir / "complaints_train.parquet", index=False)
    test.to_parquet(processed_dir / "complaints_test.parquet", index=False)
    print(f"saved {len(df)} rows -> train {len(train)}, test {len(test)} in {processed_dir}")
    print(df["product_group"].value_counts().to_string())


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / f"complaints_{name}.parquet")
