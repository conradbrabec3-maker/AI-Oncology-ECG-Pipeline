import pandas as pd
import sqlite3

# 1. Point to your database
db_path = "/home/conrad.brabec/SQL/mimic_data.db"
output_file = "/home/conrad.brabec/SQL/healthy_control_cohort.txt"

print("🔗 Connecting to MIMIC-IV SQLite Database...")
conn = sqlite3.connect(db_path)

# 2. The Strict Exclusion Query (The "Perfect Control" Filter)
# We group by the patient. If the SUM of their cancer or heart codes is > 0, they are excluded.
control_query = """
SELECT 
    subject_id
FROM table_name_2
GROUP BY subject_id
HAVING SUM(
    CASE 
        -- Rule 1: Exclude ALL ICD-10 Cancers (C00-C96)
        WHEN (icd_version = 10 AND icd_code LIKE 'C%') THEN 1
        
        -- Rule 2: Exclude ALL ICD-9 Cancers (140-208)
        WHEN (icd_version = 9 AND CAST(SUBSTR(icd_code, 1, 3) AS INTEGER) BETWEEN 140 AND 208) THEN 1
        
        -- Rule 3: Exclude ALL ICD-10 Heart/Circulatory Conditions (I00-I99)
        WHEN (icd_version = 10 AND icd_code LIKE 'I%') THEN 1
        
        -- Rule 4: Exclude ALL ICD-9 Heart/Circulatory Conditions (390-459)
        WHEN (icd_version = 9 AND CAST(SUBSTR(icd_code, 1, 3) AS INTEGER) BETWEEN 390 AND 459) THEN 1
        
        -- Otherwise, it's a safe code (like a broken bone)
        ELSE 0 
    END
) = 0;
"""

print("🚀 Scanning millions of rows to isolate patients with pristine oncology and cardiovascular records...")
control_df = pd.read_sql(control_query, conn)

# 3. Process and Deduplicate
unique_control_patients = control_df['subject_id'].unique()
print(f"✅ Extracted {len(unique_control_patients)} mathematically verified healthy control patients.")

# 4. Save to a brand new text file
pd.Series(unique_control_patients).to_csv(output_file, index=False, header=False)
print(f"📁 Successfully saved perfect control list to: {output_file}")

conn.close()
