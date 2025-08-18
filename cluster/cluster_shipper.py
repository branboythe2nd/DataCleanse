#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from collections import Counter

NAME_COLUMN = "ShipperName"
REFERENCE_IDS = ["ShipperPanjivaID"]

def cluster_names(df, name_col, id_cols):
    clusters = []

    # Ensure reference column exists
    for col in id_cols:
        if col not in df.columns:
            raise ValueError(f"Reference column '{col}' not found in CSV")

    # Group by reference ID: every group is one cluster
    grouped = df.groupby(id_cols)

    for _, group in grouped:
        names = group[name_col].dropna().unique().tolist()
        if not names:
            continue

        # Count occurrences
        name_counts = Counter(group[name_col])

        # Canonical name = most frequent, tie-breaker = longest
        canonical_name = max(names, key=lambda x: (name_counts[x], len(x)))

        clusters.append({
            "Shipper_ClusterSize": sum(name_counts[name] for name in names),
            "Shipper_NumUniqueNames": len(names),
            "Shipper_CanonicalName": canonical_name,
            "Shipper_Names": ", ".join(names)
        })

    return pd.DataFrame(clusters)

def main():
    parser = argparse.ArgumentParser(description="Cluster Shipper names in CSV based on reference IDs")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cluster_df = cluster_names(df, NAME_COLUMN, REFERENCE_IDS)

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    output_file = f"{base_filename}_shipper_cluster_summary.csv"
    cluster_df.to_csv(output_file, index=False)
    print(f"Shipper cluster summary saved to {output_file}")

if __name__ == "__main__":
    main()
