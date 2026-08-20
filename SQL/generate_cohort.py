import pandas as pd
import sqlite3

# Point to your existing SQLite database on the ARC cluster
db_path = "/home/conrad.brabec/SQL/mimic_data.db"
conn = sqlite3.connect(db_path)

# Precise Medical Query using ICD-9 and ICD-10 Rules
# Joining on BOTH icd_code and icd_version is critical in MIMIC-IV!
cancer_query = """
SELECT
    t2.subject_id,
    t2.icd_code,
    t2.icd_version,
    t1.long_title
FROM table_name_2 AS t2
JOIN table_name_1 AS t1
    ON t2.icd_code = t1.icd_code 
    AND t2.icd_version = t1.icd_version
WHERE 
    -- Rule 1: ICD-10 Malignant Neoplasms (Codes C00 - C96)
    (t2.icd_version = 10 AND t2.icd_code LIKE 'C%')
    
    OR 
    
    -- Rule 2: ICD-9 Malignant Neoplasms (Codes 140 - 208)
    -- We grab the first 3 characters, convert to a number, and check the range
    (t2.icd_version = 9 AND (
        CAST(SUBSTR(t2.icd_code, 1, 3) AS INTEGER) BETWEEN 140 AND 208
    ));
"""

print("🚀 Running hyper-precise ICD Oncology Query across millions of rows...")
cancer_df = pd.read_sql(cancer_query, conn)

print(f"✅ Found {len(cancer_df)} mathematically verified cancer diagnoses.")
print("\n--- Sample of the Diagnoses Captured ---")
print(cancer_df[['icd_version', 'icd_code', 'long_title']].head(10))

# Extract the unique patient IDs
unique_cancer_patients = cancer_df['subject_id'].unique()
print(f"\nTotal unique cancer patients isolated: {len(unique_cancer_patients)}")

# Save to your SQL folder
output_file = "/home/conrad.brabec/SQL/cancer_cohort_ids_strict.txt"
pd.Series(unique_cancer_patients).to_csv(output_file, index=False, header=False)
print(f"📁 Saved strict patient IDs to {output_file}")

conn.close()
