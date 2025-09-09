#!/usr/bin/env python3
"""
Script Name: cluster.py

Description:
    Build two cluster summaries from a single CSV:
      1) Consignee clusters using OR across reference IDs
         (ConsigneeLocalDUNS OR ConsigneePanjivaID)
      2) Shipper clusters using ShipperPanjivaID

    Each cluster summarizes:
        - <Prefix>_ClusterID
        - <Prefix>_ClusterSize (total rows in the cluster)
        - <Prefix>_NumUniqueNames
        - <Prefix>_CanonicalName  (most frequent; tie-breaker = longest)
        - <Prefix>_Names          (comma-separated unique names)
        - For Consignee: lists of unique DUNS and Panjiva IDs in the cluster
        - For Shipper: list of unique ShipperPanjivaID values in the cluster

Input CSV must include:
    - ConsigneeName, ConsigneeLocalDUNS, ConsigneePanjivaID
    - ShipperName,   ShipperPanjivaID

Output:
    A single CSV named "<input_basename>_combined_cluster_tables.csv" with two
    sections:
        "Consignee Clusters"
        (CSV table)
        [blank line]
        "Shipper Clusters"
        (CSV table)

Usage:
    python cluster.py --input <path_to_csv>

Notes:
    - Consignee clustering uses OR across IDs via an iterative DFS over
      (column, value) adjacency. This avoids recursion depth issues.
    - Both tables are sorted by <Prefix>_NumUniqueNames descending.
"""

import pandas as pd
import argparse
import os
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple


def _cluster_or(df: pd.DataFrame, name_col: str, id_cols: List[str], prefix: str) -> pd.DataFrame:
    """
    Cluster rows where records are connected if they share ANY one of the id_cols (OR logic).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least name_col and id_cols.
    name_col : str
        Column with the entity names (e.g., ConsigneeName or ShipperName).
    id_cols : list[str]
        Reference ID columns (one or more). OR logic connects rows sharing any one of them.
    prefix : str
        Prefix for output columns ("Consignee" or "Shipper").

    Returns
    -------
    pd.DataFrame
        Cluster summary sorted by <prefix>_NumUniqueNames (descending).
    """
    # Validate columns
    missing = [c for c in [name_col, *id_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Work on the necessary columns only and ensure 0..N-1 index
    df = df[[name_col, *id_cols]].copy().reset_index(drop=True)

    # Build inverted index: (col, value) -> set(row_indices)
    id_to_rows: Dict[Tuple[str, object], Set[int]] = defaultdict(set)
    for r in range(len(df)):
        for col in id_cols:
            val = df.at[r, col]
            if pd.notna(val):
                id_to_rows[(col, val)].add(r)

    visited: Set[int] = set()
    clusters = []
    cluster_id = 1

    # Iterative DFS for connected components under OR condition
    for start in range(len(df)):
        if start in visited:
            continue

        stack = [start]
        comp: Set[int] = set()

        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            comp.add(i)

            # Add neighbors sharing any ID value
            for col in id_cols:
                val = df.at[i, col]
                if pd.notna(val):
                    for n in id_to_rows.get((col, val), set()):
                        if n not in visited:
                            stack.append(n)

        comp_df = df.loc[list(comp)]
        names = comp_df[name_col].dropna().astype(str)
        if names.empty:
            continue

        # Canonical = most frequent; tie-breaker = longest string
        name_counts = Counter(names)
        canonical = max(name_counts.keys(), key=lambda x: (name_counts[x], len(x)))

        row = {
            f"{prefix}_ClusterID": cluster_id,
            f"{prefix}_ClusterSize": int(names.size),
            f"{prefix}_NumUniqueNames": int(names.nunique()),
            f"{prefix}_CanonicalName": canonical,
            f"{prefix}_Names": ", ".join(sorted(names.unique())),
        }

        # Include unique ID value lists (helpful diagnostics)
        for col in id_cols:
            col_vals = comp_df[col].dropna().astype(str).unique()
            row[f"{prefix}_{col}_Values"] = ", ".join(sorted(col_vals))

        clusters.append(row)
        cluster_id += 1

    # Sort by NumUniqueNames descending for readability
    out = (
        pd.DataFrame(clusters)
        .sort_values(by=f"{prefix}_NumUniqueNames", ascending=False)
        .reset_index(drop=True)
    )
    return out


def cluster_consignee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consignee clustering: OR across ConsigneeLocalDUNS and ConsigneePanjivaID.
    """
    return _cluster_or(
        df,
        name_col="ConsigneeName",
        id_cols=["ConsigneeLocalDUNS", "ConsigneePanjivaID"],
        prefix="Consignee",
    )


def cluster_shipper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shipper clustering: by ShipperPanjivaID (OR across one column = same as grouping by it).
    """
    return _cluster_or(
        df,
        name_col="ShipperName",
        id_cols=["ShipperPanjivaID"],
        prefix="Shipper",
    )


def main():
    """
    Command-line entry point:
      - Reads input CSV
      - Builds Consignee (OR) clusters and Shipper clusters
      - Writes both tables into one CSV with section headers
    """
    parser = argparse.ArgumentParser(description="Cluster Consignee (OR across IDs) and Shipper names from a CSV.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    consignee_clusters = cluster_consignee(df)
    shipper_clusters = cluster_shipper(df)

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    output_file = f"{base_filename}_combined_cluster_tables.csv"

    # Write both tables into a single CSV-like file with section headers
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write("Consignee Clusters\n")
        consignee_clusters.to_csv(f, index=False)
        f.write("\n\n")
        f.write("Shipper Clusters\n")
        shipper_clusters.to_csv(f, index=False)

    print(f"Combined cluster tables saved to {output_file}")


if __name__ == "__main__":
    main()
