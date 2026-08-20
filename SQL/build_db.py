import pandas as pd
from sqlalchemy import create_engine
import os

# --- 1. USER INPUT SECTION ---
# Update these paths to point to your actual .csv.gz files
file_1_path = "/home/conrad.brabec/mimic-iv-3.1//hosp/d_icd_diagnoses.csv.gz"
file_2_path = "/home/conrad.brabec/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz"

# Where you want the final SQLite database saved
db_path = "/home/conrad.brabec/mimic_data.db"
# -----------------------------

def ingest_file(file_path, table_name, engine):
    print(f"--- Starting ingestion for: {table_name} ---")
    
    # Process in chunks of 100,000 rows to keep memory usage low
    # 'compression="gzip"' tells pandas to handle the .gz automatically
    chunk_iterator = pd.read_csv(file_path, chunksize=100000, compression='gzip')
    
    for i, chunk in enumerate(chunk_iterator):
        # First chunk replaces (or creates) table; subsequent chunks append
        mode = 'replace' if i == 0 else 'append'
        chunk.to_sql(table_name, engine, if_exists=mode, index=False)
        print(f"Processed chunk {i+1}...")
    
    print(f"Finished {table_name}.")

# Create database connection
engine = create_engine(f"sqlite:///{db_path}")

# Ingest your two files
ingest_file(file_1_path, "table_name_1", engine)
ingest_file(file_2_path, "table_name_2", engine)

print("All files successfully imported into the SQLite database.")
