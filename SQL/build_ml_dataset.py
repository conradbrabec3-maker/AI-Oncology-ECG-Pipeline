import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. File Paths
cancer_file = "/home/conrad.brabec/SQL/cancer_cohort_ids_strict.txt"
healthy_file = "/home/conrad.brabec/SQL/healthy_control_cohort.txt"
output_file = "/home/conrad.brabec/detector_v5/master_dataset_index.csv"

print("⏳ Loading patient cohorts...")
cancer_ids = pd.read_csv(cancer_file, header=None, names=['subject_id'])
healthy_ids = pd.read_csv(healthy_file, header=None, names=['subject_id'])

# 2. Assign AI Labels (1 = Cancer, 0 = Healthy)
cancer_ids['label'] = 1
healthy_ids['label'] = 0

print(f"Original counts -> Cancer: {len(cancer_ids)}, Healthy: {len(healthy_ids)}")

# 3. Balance the Dataset (Strict 50/50 Split)
# We find whichever group is smaller, and match the larger group to that size.
min_count = min(len(cancer_ids), len(healthy_ids))
print(f"⚖️ Balancing dataset to exactly {min_count} patients per class...")

cancer_balanced = cancer_ids.sample(n=min_count, random_state=42)
healthy_balanced = healthy_ids.sample(n=min_count, random_state=42)

# Combine into one master dataframe
master_df = pd.concat([cancer_balanced, healthy_balanced]).reset_index(drop=True)

# 4. Perform the Train / Validation / Test Split (70% / 15% / 15%)
# The 'stratify' command ensures that every split has exactly 50% cancer and 50% healthy!
print("🪓 Splitting into Train, Validation, and Test sets...")

# First, split off 30% for Val/Test combined
train_df, temp_df = train_test_split(master_df, test_size=0.30, random_state=42, stratify=master_df['label'])

# Next, split that 30% strictly in half (15% Val, 15% Test)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

# Mark the splits in the dataframe so PyTorch knows how to use them
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# 5. Combine and Save
final_dataset = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
final_dataset.to_csv(output_file, index=False)

print(f"\n✅ Master ML Dataset successfully saved to: {output_path}")
print("--- FINAL DATASET METRICS ---")
print(f"🧠 Training Patients:   {len(train_df)}")
print(f"📊 Validation Patients: {len(val_df)}")
print(f"🎓 Test Exam Patients:  {len(test_df)}")
print(f"Total AI Roster:        {len(final_dataset)}")
