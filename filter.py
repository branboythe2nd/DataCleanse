
#!/usr/bin/env python3
"""
Improved HS Code Filter (retains HS column + configurable output dir)
--------------------------------------------------------------------
Scans CSVs under --input-dir (recursively), auto-detects the HS code column,
filters rows where ANY HS code in the cell begins with:
    - "03"  (allow missing leading zero cases)
    - "1604" or "1605"
and writes a slimmed CSV to --output-dir while retaining the HS column and the
requested columns.

Key behavior:
- Auto-detect HS column by name preference or content heuristic.
- Supports multiple HS codes per cell (comma-separated).
- Robust normalization:
    * strips non-digits,
    * restores missing leading zero for odd digit-lengths (5→6, 7→8, 9→10),
    * accepts only 6/8/10-digit tokens.
- Chunked processing for large files.
- Output includes HS column (original values) + specified business columns.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Columns to retain in the output (plus the HS code column)
KEEP_COLS = [
    "ShipperName",
    "ShipperFullAddress",
    "ShipperLocalDUNS",
    "ShipperPanjivaID",
    "ShipperCleanedStockTickers",
    "ShipperOriginalFormat",
    "NetWeightKg",
    "ValueOfGoodsFOBUSD",
]

# Preferred HS column name candidates (checked in order)
HS_CANDIDATE_NAMES = [
    "HSCode", "HS Code", "HS_Code", "HS", "HSCODE", "HTS", "HTSCode", "HTS Code",
]

# Chunk size for large CSVs
CHUNK_ROWS = 150_000

DIGIT_RE = re.compile(r"\D+")

def normalize_hs_token(token: str) -> Optional[str]:
    """Normalize a HS token to 6/8/10 digits, fixing missing leading zero if needed."""
    if token is None:
        return None
    digits = DIGIT_RE.sub("", str(token))
    if not digits:
        return None
    if len(digits) in (5, 7, 9):
        digits = "0" + digits
    if len(digits) not in (6, 8, 10):
        return None
    return digits

def any_token_matches_target(cell: str) -> bool:
    """True if ANY comma-separated token normalizes and starts with 03/1604/1605."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return False
    for raw in str(cell).split(","):
        norm = normalize_hs_token(raw.strip())
        if not norm:
            continue
        if norm.startswith(("03", "1604", "1605")):
            return True
    return False

def score_as_hs_column(series: pd.Series, sample: int = 10_000) -> float:
    """Heuristic fraction of sampled values that contain a valid 6/8/10-digit HS token."""
    if series.empty:
        return 0.0
    s = series.dropna().astype(str).head(sample)
    if s.empty:
        return 0.0
    total = len(s)
    good = 0
    for val in s:
        tokens = str(val).split(",")
        if any(normalize_hs_token(tok.strip()) for tok in tokens):
            good += 1
    return good / max(total, 1)

def detect_hs_column(df: pd.DataFrame) -> Optional[str]:
    """Pick HS column by name preference, else by content score (>=0.30)."""
    for name in HS_CANDIDATE_NAMES:
        if name in df.columns:
            return name
    scores = {col: score_as_hs_column(df[col]) for col in df.columns}
    best_col, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_col if best_score >= 0.30 else None

def write_filtered_csv(
    src_path: Path,
    dest_path: Path,
    hs_col: str,
) -> Tuple[int, int]:
    """Stream, filter by HS criteria on hs_col, and write HS + KEEP_COLS. Returns (kept, total)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    header_written = False

    head = pd.read_csv(src_path, nrows=0, dtype=str, low_memory=False)
    # Build read/use columns: HS + those KEEP_COLS that exist
    in_cols = [c for c in KEEP_COLS if c in head.columns]
    if hs_col not in in_cols:
        in_cols = [hs_col] + in_cols

    out_cols = in_cols[:]  # we will write the same set we read (HS + keepers)

    for chunk in pd.read_csv(src_path, usecols=in_cols, chunksize=CHUNK_ROWS, dtype=str, low_memory=False):
        total += len(chunk)
        mask = chunk[hs_col].apply(any_token_matches_target)
        sub = chunk.loc[mask, out_cols].copy()
        if not sub.empty:
            sub.to_csv(dest_path, mode="a", header=not header_written, index=False)
            header_written = True
            kept += len(sub)

    return kept, total

def process_csv(csv_path: Path, input: Path, output: Path) -> None:
    rel = csv_path.relative_to(input)
    out_path = output / rel.parent / f"{csv_path.stem}_filtered.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {csv_path} ...")
    try:
        sample_df = pd.read_csv(csv_path, nrows=50_000, dtype=str, low_memory=False)
    except Exception as e:
        print(f"  Skipping (read error): {e}")
        return

    hs_col = detect_hs_column(sample_df)
    if not hs_col:
        print("  Skipping: could not detect HS code column.")
        return

    kept, total = write_filtered_csv(csv_path, out_path, hs_col=hs_col)
    print(f"  Done: kept {kept:,} / {total:,} rows  ->  {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Filter CSVs by HS code prefix (03/1604/1605), retaining HS column.")
    ap.add_argument("--input", required=True, help="Folder containing input CSV files (scanned recursively).")
    ap.add_argument("--output", required=True, help="Folder to write filtered CSVs (mirrors input structure).")
    args = ap.parse_args()

    input = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not input.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input}")
    output.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input.rglob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in: {input}")
        return

    for p in csv_paths:
        process_csv(p, input=input, output=output)

    print("All processing complete.")

if __name__ == "__main__":
    main()
