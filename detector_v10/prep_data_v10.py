import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configurations ---
PATIENTS_CSV_PATH = "/home/conrad.brabec/mimic-iv-3.1/hosp/patients.csv.gz"
ECG_MAP_PATH = "/home/conrad.brabec/ecg_file_map.csv" # Your existing balanced cohort map
OUTPUT_TRAIN = "/home/conrad.brabec/detector_v10/train_v10.csv"
OUTPUT_TEST = "/home/conrad.brabec/detector_v10/test_v10.csv"

def prep_v10_data():
    print("🚀 Loading Datasets...")
    # 1. Load the existing balanced cohort map (contains subject_id, record_path, is_cancer)
    ecg_df = pd.read_csv(ECG_MAP_PATH)
    
    # 2. Load the MIMIC-IV patients table to get Age and Gender
    patients_df = pd.read_csv(PATIENTS_CSV_PATH, usecols=['subject_id', 'gender', 'anchor_age'])
    
    # 3. Merge demographics into the ECG dataset
    print("🧬 Merging Demographics...")
    
    # Force both ID columns to be strings so they match perfectly
    ecg_df['patient_id'] = ecg_df['patient_id'].astype(str)
    patients_df['subject_id'] = patients_df['subject_id'].astype(str)
    
    merged_df = pd.merge(ecg_df, patients_df, left_on='patient_id', right_on='subject_id', how='inner')
    
    # 4. Demographic Encoding & Normalization
    # Encode Gender: Male (M) = 0, Female (F) = 1
    merged_df['gender_encoded'] = merged_df['gender'].apply(lambda x: 1 if x == 'F' else 0)
    
    # Normalize Age: Min-Max scaling to compress ages (e.g., 18-100) into a [0.0, 1.0] range
    # This ensures an age of 85 doesn't mathematically overwhelm a normalized ECG tensor.
    min_age = merged_df['anchor_age'].min()
    max_age = merged_df['anchor_age'].max()
    merged_df['age_normalized'] = (merged_df['anchor_age'] - min_age) / (max_age - min_age)
    
    # 5. The 70/30 Split
    # We use 'stratify' to guarantee both the 70% and 30% sets have perfectly balanced Cancer/Healthy ratios
    print("🪓 Executing 70/30 Stratified Split...")
    train_df, test_df = train_test_split(
        merged_df, 
        test_size=0.30, 
        random_state=42, 
        stratify=merged_df['record_path']
    )
    
    # Save the splits
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)
    
    print(f"✅ Success! Training Set: {len(train_df)} patients. Test Set: {len(test_df)} patients.")

if __name__ == "__main__":
    prep_v10_data()
