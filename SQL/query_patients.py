
import pandas as pd
import sqlite3

# Point to the database INSIDE the SQL folder
conn = sqlite3.connect("/home/conrad.brabec/SQL/mimic_data.db")

# Use table_name_2 (Patient Log) and table_name_1 (Dictionary)
cancer_query = """
SELECT 
    t2.subject_id, 
    t2.icd_code, 
    t1.long_title
FROM table_name_2 AS t2
JOIN table_name_1 AS t1 
    ON t2.icd_code = t1.icd_code
WHERE LOWER(t1.long_title) LIKE '%cancer%'
   OR LOWER(t1.long_title) LIKE '%malignant%'
   OR LOWER(t1.long_title) LIKE '%neoplasm%';
"""

print("Running SQL Query across millions of rows...")
cancer_df = pd.read_sql(cancer_query, conn)

print(f"Found {len(cancer_df)} cancer-related diagnoses.")
print(cancer_df.head())

unique_cancer_patients = cancer_df['subject_id'].unique()
print(f"Total unique cancer patients: {len(unique_cancer_patients)}")

output_file = "/home/conrad.brabec/SQL/cancer_cohort_ids.txt"
pd.Series(unique_cancer_patients).to_csv(output_file, index=False, header=False)
print(f"✅ Saved unique patient IDs to {output_file}")

conn.close()
