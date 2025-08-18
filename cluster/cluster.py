#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from collections import Counter

def cluster_names(df, name_col, id_cols, prefix):
    clusters = []

    for col in id_cols:
        if col not in df.columns:
            raise ValueError(f"Reference column '{col}' not found in CSV")

    grouped = df.groupby(id_cols)

    for _, group in grouped:
        names = group[name_col].dropna().unique().tolist()
        if not names:
            continue

        name_counts = Counter(group[name_col])
        canonical_name = max(names, key=lambda x: (name_counts[x], len(x)))

        clusters.append({
            f"{prefix}_ClusterSize": sum(name_counts[name] for name in names),
            f"{prefix}_NumUniqueNames": len(names),
            f"{prefix}_CanonicalName": canonical_name,
            f"{prefix}_Names": ", ".join(names)
        })

    return pd.DataFrame(clusters)

def main():
    parser = argparse.ArgumentParser(description="Cluster Consignee and Shipper names in CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Consignee clustering
    consignee_clusters = cluster_names(df, "ConsigneeName", ["ConsigneeLocalDUNS", "ConsigneePanjivaID"], "Consignee")

    # Shipper clustering
    shipper_clusters = cluster_names(df, "ShipperName", ["ShipperPanjivaID"], "Shipper")

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    output_file = f"{base_filename}_combined_cluster_tables.csv"

    # Write to CSV with two tables
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write("Consignee Clusters\n")
        consignee_clusters.to_csv(f, index=False)
        f.write("\n\n")  # empty line separator
        f.write("Shipper Clusters\n")
        shipper_clusters.to_csv(f, index=False)

    print(f"Combined cluster tables saved to {output_file}")

if __name__ == "__main__":
    main()
