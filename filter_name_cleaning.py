#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import csv
import sys

# Increase maximum CSV field size
csv.field_size_limit(sys.maxsize)

# Columns to keep in final output
columns_to_keep = [
    "HSCode",
    "ConsigneeName",
    "ConsigneeLocalDUNS",
    "ConsigneePanjivaID",
    "ShipperName",
    "ShipperPanjivaID",
]

# Output directory
output_dir = "./datasets/filtered"  # current directory by default

# --- FUNCTIONS ---

def clean_and_filter(df):
    """Normalize HSCode column and filter rows by HSCode."""
    original_columns = df.columns.copy()

    # Normalize columns
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True).str.lower()

    if 'hscode' not in df.columns:
        raise ValueError("HSCode column not found after normalization.")

    # Clean HSCode column
    df['hscode'] = df['hscode'].astype(str).str.strip().str.replace('.', '', regex=False)
    df['hscode'] = df['hscode'].apply(lambda x: '0' + x if x.startswith('3') and not x.startswith('03') else x)

    # Filter function
    def hscode_matches(cell):
        codes = [code.strip() for code in cell.split(';')]
        for code in codes:
            if code.startswith('03') or code.startswith('1604') or code.startswith('1605'):
                return True
        return False

    filtered_df = df[df['hscode'].apply(hscode_matches)]
    filtered_df.columns = original_columns
    return filtered_df

# --- MAIN PROCESSING ---

def main():
    input_files = sys.argv[1:]  # get files from command-line arguments

    if not input_files:
        print("Usage: python filter_hscode.py file1.csv file2.csv ...")
        sys.exit(1)

    filtered_dataframes = {}

    for filepath in input_files:
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"Processing {model_name} from {filepath}...")

        try:
            df = pd.read_csv(filepath, engine="python")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        print(f"{model_name}: Loaded {len(df)} rows")

        try:
            filtered_df = clean_and_filter(df)
        except ValueError as ve:
            print(f"Skipping {model_name}: {ve}")
            continue

        # Save filtered file
        out_path = os.path.join(output_dir, f"{model_name}_filtered_HSCode.csv")
        filtered_df.to_csv(out_path, index=False)
        print(f"Saved filtered HSCode to {out_path} ({filtered_df.shape[0]} rows)")

        filtered_dataframes[model_name] = filtered_df

    # Select columns and save final output
    for model, df_filtered in filtered_dataframes.items():
        missing_cols = [col for col in columns_to_keep if col not in df_filtered.columns]
        if missing_cols:
            print(f"Warning: Missing columns in {model}: {missing_cols}")
            final_df = df_filtered[[col for col in columns_to_keep if col in df_filtered.columns]]
        else:
            final_df = df_filtered[columns_to_keep]

        out_path = os.path.join(output_dir, f"{model}_filtered_HSCode_selected_columns.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved {model} selected columns to {out_path} ({final_df.shape[0]} rows)")

if __name__ == "__main__":
    main()
