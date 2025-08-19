DataCleanse.

A toolkit for cleaning, preprocessing, and clustering trade datasets (imports/exports).  
This repository contains Python scripts and Jupyter notebooks to normalize HS codes, filter noisy records, and group similar entities (e.g., consignees and shippers).






Project Structure 
- cluster 
        - This folder contains code to cluster and group similar entities of the trade dataset (based on consignee name and shipper name).
- jupyter notebook
        - This folder contains jupyter notebooks that allow basic filtering of the trade dataset and initial cluster code.
- input 
        - This folder contains trade input data. Data is split based on country.
- output
        - This folder contains data outputted from running preprocessing and cluster functions.
- preprocessing
        - This folder contrains code to preprocess and clean input data. Specifically, the output data contains cleaned up relevant HSCodes (03,1604,1605) and only relevant columns.


Execution Steps
1. Install dependencies
   1. Start a virtual environment :
            > python3 -m venv tradesweep
            > (Mac/Linux) : source venv/bin/activate    (Windows) : venv\Scripts\activate
   2. Download requirement.txt : pip install -r requirements.txt
      
2. Preprocessing Steps
   1. filter.py (retains only the rows with relevant HSCodes, i.e 03, 1604, 1605) : python filter.py --input <input_folder/> --output <output_folder/>
   2. filter_hscode.py (normalizes the HSCodes and ensures correct format) : python filter_hscode.py --input <input_folder/> --output <output_folder/>
   3. combine_csv.py (combines the individual chronological month csv files to output a final csv file by year) : python combine_csv.py --input <input_folder/> --output <output_folder/>

   
4. Cluster
    - cluster based on consignee names : python cluster_consignee.py --input <input_folder/> --output <output_folder/>
    - cluster based on shipper names : cluster_shipper.py --input <input_folder/> --output <output_folder/>
    - cluster based on both consignee names and shipper names : python cluster.py --input <input_folder/> --output <output_folder/>
