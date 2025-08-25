#!/usr/bin/env python3
"""
Script Name: cluster_consignee.py

Description:
    This script reads a CSV file of trade/import data and clusters consignee
    names based on shared reference IDs. If two consignee rows have the same
    combination of ConsigneeLocalDUNS and ConsigneePanjivaID, they are treated
    as belonging to the same cluster (i.e., technically the same entity).
    For each cluster, the script summarizes:
        - Cluster size (total rows)
        - Number of unique consignee names
        - Canonical name (most frequent, tie-broken by length)
        - List of all unique names in the cluster

Input:
    A CSV file containing at least the following columns:
        - ConsigneeName
        - ConsigneeLocalDUNS
        - ConsigneePanjivaID

Output:
    A new CSV file with the suffix "_consignee_cluster_summary.csv"
    containing the summarized cluster information.

Usage:
    python cluster_consignee.py --input <path_to_csv>

Example:
    python cluster_consignee.py --input us_imports_2015.csv

Dependencies:
    - pandas
    - rapidfuzz (installed but not used in this version)
    - collections (standard library)
    - argparse, os (standard library)

Notes:
    - SIMILARITY_THRESHOLD is defined but not applied in this version.
    - Clustering is strict: only rows sharing both reference IDs fall into
      the same cluster.
    - To extend functionality, fuzzy matching can be added using rapidfuzz.

"""

import pandas as pd
import argparse
import os
from collections import Counter
from rapidfuzz import fuzz

SIMILARITY_THRESHOLD = 85
NAME_COLUMN = "ConsigneeName"
REFERENCE_IDS = ["ConsigneeLocalDUNS", "ConsigneePanjivaID"]


def cluster_names(df, name_col, id_cols):
    """
    Cluster consignee names based on reference IDs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing consignee information.
    name_col : str
        Column name containing consignee names.
    id_cols : list of str
        Reference ID columns (e.g., ConsigneeLocalDUNS, ConsigneePanjivaID).

    Returns
    -------
    pandas.DataFrame
        A dataframe summarizing clusters with columns:
            - Consignee_ClusterSize: total number of rows in the cluster
            - Consignee_NumUniqueNames: count of distinct names
            - Consignee_CanonicalName: most representative name
            - Consignee_Names: comma-separated list of unique names
    """
    clusters = []

    # Ensure reference columns exist
    for col in id_cols:
        if col not in df.columns:
            raise ValueError(f"Reference column '{col}' not found in CSV")

    # Group by reference IDs: every group is one cluster
    grouped = df.groupby(id_cols)

    for _, group in grouped:
        names = group[name_col].dropna().unique().tolist()
        if not names:
            continue

        # Count occurrences of each name
        name_counts = Counter(group[name_col])

        # Canonical name = most frequent; tie-breaker = longest string
        canonical_name = max(names, key=lambda x: (name_counts[x], len(x)))

        clusters.append({
            "Consignee_ClusterSize": sum(name_counts[name] for name in names),
            "Consignee_NumUniqueNames": len(names),
            "Consignee_CanonicalName": canonical_name,
            "Consignee_Names": ", ".join(names)
        })

    return pd.DataFrame(clusters)


def main():
    """
    Command-line entry point.

    - Parses arguments for input CSV path.
    - Calls cluster_names() to generate clusters.
    - Saves the cluster summary as a new CSV file in the same directory.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Cluster Consignee names in CSV based on reference IDs")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cluster_df = cluster_names(df, NAME_COLUMN, REFERENCE_IDS)

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    output_file = f"{base_filename}_consignee_cluster_summary.csv"
    cluster_df.to_csv(output_file, index=False)
    print(f"Consignee cluster summary saved to {output_file}")


if __name__ == "__main__":
    main()
