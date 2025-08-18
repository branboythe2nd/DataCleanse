import os
import glob
import pandas as pd
import re

def combine_csv(input_folder, output_file):
    # Find all CSV files in the folder
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # Sort files numerically based on the number in the filename
    def numerical_sort(value):
        # Extract numbers from filename
        numbers = re.findall(r'\d+', os.path.basename(value))
        return [int(num) for num in numbers]

    all_files.sort(key=numerical_sort)

    # Read and combine them
    df_list = []
    for file in all_files:
        print(f"Reading {file} ...")
        df = pd.read_csv(file)
        df_list.append(df)
    
    # Concatenate into one DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Save to output CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined {len(all_files)} files into {output_file}")

if __name__ == "__main__":
    input_folder = "datasets"              # change this to your folder
    output_file = "combined_output.csv"    # change this to your output file
    combine_csv(input_folder, output_file)
