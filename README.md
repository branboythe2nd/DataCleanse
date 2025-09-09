<<<<<<< HEAD

# CSV Cluster Review — Full Stack App

A minimal full‑stack web app for reviewing clusters on CSV rows:

- **Upload a CSV** and display it as a table
- **Apply Clustering Strategy 1** (exact duplicates on a chosen text column)
- **Apply Clustering Strategy 2** (fuzzy clusters by Jaccard token similarity on a chosen text column)
- **Approve/Disapprove** each highlighted row + select a **reason** ("Underclustered", "Overclustered", "Other" + free text)
- **Lock** approved rows (locked rows are frozen and excluded from later clustering; unlock is available)
- **Provenance log** records which strategy ran and every decision
- **Download** state as JSON and **export annotated CSV**
- **Save/Load** state to a tiny Node/Express server

## Quick start

### 1) Server
```bash
cd server
npm install
npm start         # starts on http://localhost:5050
```

### 2) Client
```bash
cd client
npm install
npm run dev       # Vite on http://localhost:5173
```

You can change the API base by setting `VITE_API_BASE` for the client:
```bash
VITE_API_BASE="http://localhost:5050" npm run dev
```

## Notes

- Strategy 1 groups rows by *exact* normalized text of the selected column (lowercased, spaces normalized). Clusters with >1 member highlight all member rows.
- Strategy 2 groups by Jaccard token similarity >= 0.5 on the selected text column using a simple union‑find. You can tweak the threshold in `App.jsx`.
- Row metadata (cluster id, decision, reason, locked) are carried in a `_meta` object attached to each row in memory and exported into the annotated CSV.
- The server stores state at `server/data/state.json`.


---

## Implement your clustering
Open **client/src/strategies.js** and implement `strategy1` / `strategy2`.

Each should **return an array of clusters** like:
```js
[{ id: "S1-group-1", members: [0, 3, 8] }]
```
- `id` will be shown in the **Cluster** column.
- `members` are **row indices** (0-based) into the current rows array.
- The UI **ignores locked rows** automatically; please avoid including them yourself too.

Use `rows`, `columns`, and `options.textColumn` as needed.


---

## Delegate clustering to Python (your implementation)

Buttons **Apply Strategy 1 / 2** now POST to the server endpoints:

- `POST /cluster/S1`
- `POST /cluster/S2`

The server reads `server/config.json` to find your Python interpreter and
strategy scripts:
```json
{
  "python": "python3",
  "strategies": {
    "S1": "strategies/strategy1.py",
    "S2": "strategies/strategy2.py"
  }
}
```

### Python contract
Your script must:
- **Read** a single JSON object from STDIN:
  ```json
  { "rows": [...], "columns": [...], "textColumn": "name", "params": {} }
  ```
- **Write** a JSON object to STDOUT:
  ```json
  {
    "clusters": [
      {"id": "cluster-1", "members": [0,2,5]},
      {"id": "cluster-2", "members": [1,4]}
    ],
    "meta": { "strategy": "S1", "notes": "optional" }
  }
  ```

> `members` are 0-based **row indices** in the incoming array. The UI will highlight
> those rows and display `id` in the “Cluster” column. **Locked rows** are left unchanged.

Two stub files are provided for you to implement:
- `server/strategies/strategy1.py`
- `server/strategies/strategy2.py`

Make them executable and implement your logic. Example run check:
```bash
cd server
echo '{"rows":[],"columns":[]}' | python3 strategies/strategy1.py
```

If you prefer running a separate Python web service, you can modify
`server/index.js` in `runPythonStrategy()` to call your service instead of spawning a process.


### Returning a CSV from your Python strategy
Your Python may also return a CSV. The server will attach it back to the UI if you include **ONE** of:

- `"csv": "<raw CSV string>"` (preferred)
- `"csv_path": "relative/or/absolute/path.csv"`
- `"csv_b64": "<base64 text>"`

The UI will parse the CSV and **replace the table** with the returned data *before* highlighting rows by the provided cluster memberships.


### Using the attached Entity Clusterer (CLI)
We integrated a strategy **EC** that calls your Python script with CLI args
(`--input_dir`, `--output_dir`, `--feedback_dir`, `--name_cols`, `--helper_cols`, `--id_col`).  
Put your script at `server/strategies/entity_clusterer.py` (already copied from your upload).  
In the UI, fill **name_cols**, **helper_cols**, and **id_col** (optional), then click
**Apply Entity Clusterer (Python)**. The backend writes the current table to a
temporary `input/input.csv`, executes your script, and returns the resulting
`out/clusters_latest.csv` to the UI, which becomes the displayed table. Rows with
the same `cluster_id` are highlighted as clusters.


### Simplified flow (your Python strategy only)
- Upload CSV → click **Apply Entity Clusterer (Python)**.
- The backend runs your script and returns `clusters_latest.csv`.
- The UI shows that CSV, highlights clusters, and enables approve/disapprove + reasons, lock/unlock, provenance logging, and download/save.
=======
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
>>>>>>> origin/nk
