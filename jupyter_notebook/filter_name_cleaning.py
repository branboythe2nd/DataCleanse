import pandas as pd
import os
import csv
import sys

csv.field_size_limit(sys.maxsize)

columns_to_keep = [
    "HSCode",
    "ConsigneeName",
    "ConsigneeLocalDUNS",
    "ConsigneePanjivaID",
    "ShipperName",
    "ShipperPanjivaID",
]

output_dir = "./datasets/filtered"
os.makedirs(output_dir, exist_ok=True)

def clean_and_filter_chunk(df_chunk):
    df_chunk.columns = df_chunk.columns.str.strip().str.replace('\ufeff', '', regex=True).str.lower()
    if 'hscode' not in df_chunk.columns:
        return pd.DataFrame()  # skip if no HSCode
    df_chunk['hscode'] = df_chunk['hscode'].astype(str).str.strip().str.replace('.', '', regex=False)
    df_chunk['hscode'] = df_chunk['hscode'].apply(lambda x: '0' + x if x.startswith('3') and not x.startswith('03') else x)

    def hscode_matches(cell):
        codes = [code.strip() for code in cell.split(';')]
        for code in codes:
            if code.startswith('03') or code.startswith('1604') or code.startswith('1605'):
                return True
        return False

    return df_chunk[df_chunk['hscode'].apply(hscode_matches)]

def main():
    input_files = sys.argv[1:]
    if not input_files:
        print("Usage: python filter_hscode.py file1.csv file2.csv ...")
        sys.exit(1)

    for filepath in input_files:
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(output_dir, f"{model_name}_filtered_HSCode_selected_columns.csv")
        print(f"Processing {model_name} in chunks from {filepath}...")

        # Initialize CSV writer
        first_chunk = True
        try:
            for chunk in pd.read_csv(filepath, engine='python', chunksize=100000):
                filtered_chunk = clean_and_filter_chunk(chunk)
                if filtered_chunk.empty:
                    continue

                # Select available columns
                available_cols = [col for col in columns_to_keep if col in filtered_chunk.columns]
                filtered_chunk = filtered_chunk[available_cols]

                # Append or write header
                filtered_chunk.to_csv(out_path, mode='w' if first_chunk else 'a', index=False, header=first_chunk)
                first_chunk = False
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

        print(f"Saved filtered data to {out_path}")

if __name__ == "__main__":
    main()
