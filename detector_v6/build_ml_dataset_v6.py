import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
CANCER_FILE = "/home/conrad.brabec/SQL/cancer_cohort_ids_strict.txt"
HEALTHY_FILE = "/home/conrad.brabec/SQL/healthy_control_cohort.txt"
PHYSICAL_MAP = "/home/conrad.brabec/ecg_file_map.csv"
OUTPUT_FILE = "/home/conrad.brabec/detector_v6/master_dataset_index_v6.csv"

print("⏳ Loading SQL cohorts and physical file map...")
cancer_ids = pd.read_csv(CANCER_FILE, header=None, names=['subject_id'], dtype=str)
healthy_ids = pd.read_csv(HEALTHY_FILE, header=None, names=['subject_id'], dtype=str)

df_map = pd.read_csv(PHYSICAL_MAP)
path_col = next((col for col in df_map.columns if col.lower() in ['path', 'record_path', 'record_name', 'file_path']), None)

# Extract 8-digit IDs to find out who actually has an ECG file
df_map['clean_id'] = df_map[path_col].astype(str).str.extract(r'(?:p)?(\d{8})', expand=False)
valid_ecg_ids = df_map['clean_id'].dropna().unique()

print(f"Raw SQL Counts -> Cancer: {len(cancer_ids)}, Healthy: {len(healthy_ids)}")

# --- THE V6 FIX: Filter Ghost Patients ---
print("🧹 Removing patients who do not have physical ECG files...")
cancer_ids = cancer_ids[cancer_ids['subject_id'].isin(valid_ecg_ids)]
healthy_ids = healthy_ids[healthy_ids['subject_id'].isin(valid_ecg_ids)]

print(f"Verified ECG Counts -> Cancer: {len(cancer_ids)}, Healthy: {len(healthy_ids)}")

# Assign labels and balance 50/50
cancer_ids['label'] = 1.0
healthy_ids['label'] = 0.0

min_count = min(len(cancer_ids), len(healthy_ids))
print(f"⚖️ Balancing dataset to exactly {min_count} VERIFIED patients per class...")

cancer_balanced = cancer_ids.sample(n=min_count, random_state=42)
healthy_balanced = healthy_ids.sample(n=min_count, random_state=42)
master_df = pd.concat([cancer_balanced, healthy_balanced]).reset_index(drop=True)

# 70/15/15 Split
print("🪓 Splitting into Train, Validation, and Test sets...")
train_df, temp_df = train_test_split(master_df, test_size=0.30, random_state=42, stratify=master_df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

final_dataset = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
final_dataset.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Master V6 Dataset saved: {OUTPUT_FILE}")
