import os
import torch
import torch.nn as nn
import pandas as pd
import wfdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import zipfile
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
ZIP_PATH = '/tmp/local_dataset.zip' # Updated to match SLURM staging name
TEST_CSV = '/home/conrad.brabec/detector_v4/test_cohort.csv'
INDEX_MAP = '/home/conrad.brabec/ecg_file_map.csv' 
MODEL_PATH = '/home/conrad.brabec/detector_v4/ecg_cancer_detector.pt' 
BATCH_SIZE = 128
NUM_WORKERS = 4

# ==========================================
# 2. V4 AI ARCHITECTURE (Must Match Training)
# ==========================================
class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        super(CancerDetector1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 312, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ==========================================
# 3. DATASET LOADER (WITH Z-SCORE MATH)
# ==========================================
class ECGInferenceDataset(Dataset):
    def __init__(self, cohort_csv, index_csv, zip_path):
        self.zip_path = zip_path
        self.tmp_dir = "/tmp/ecg_test_buffers_v4"
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        print("⏳ Loading labels and master file map...")
        df_labels = pd.read_csv(cohort_csv)
        df_labels['patient_id'] = df_labels['patient_id'].astype(str)
        
        if 'label' in df_labels.columns:
            df_labels = df_labels.rename(columns={'label': 'true_label'})

        df_map = pd.read_csv(index_csv)
        df_map['clean_id'] = df_map['record_path'].str.split('/').str[3].str.replace('p', '', case=False)

        self.merged_df = pd.merge(df_labels, df_map, left_on='patient_id', right_on='clean_id', how='inner')
        self.merged_df = self.merged_df.drop_duplicates(subset=['clean_id'])
        
        self.records = self.merged_df.to_dict('records')
        print(f"✅ Successfully matched {len(self.records)} ECG records for the Final Exam!")
        self.zf = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, 'r')
            
        record_data = self.records[idx]
        record_path = record_data['record_path']
        true_label = record_data['true_label']
        
        micro_folder = os.path.join(self.tmp_dir, f"test_{idx}")
        os.makedirs(micro_folder, exist_ok=True)

        try:
            self.zf.extract(record_path + '.hea', micro_folder)
            self.zf.extract(record_path + '.dat', micro_folder)

            local_record = os.path.join(micro_folder, record_path)
            record = wfdb.rdrecord(local_record)

            signal = record.p_signal
            signal = np.nan_to_num(signal)
            
            # --- V4 MATH FIX: Z-SCORE NORMALIZATION ---
            mean = np.mean(signal, axis=0)
            std = np.std(signal, axis=0) + 1e-8
            signal = (signal - mean) / std

            signal = np.transpose(signal)

            return torch.tensor(signal, dtype=torch.float32), true_label
        finally:
            shutil.rmtree(micro_folder, ignore_errors=True)

# ==========================================
# 4. EVALUATION PIPELINE
# ==========================================
def evaluate_model():
    print("⏳ Initializing V4 Evaluation Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference Engine Target: {device}")

    model = CancerDetector1DCNN()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.to(device)
    model.eval() 

    dataset = ECGInferenceDataset(TEST_CSV, INDEX_MAP, ZIP_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds = []
    all_labels = []

    print(f"🔍 Running {len(dataset)} patients through the AI...")
    
    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Evaluating"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            preds = (outputs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n==================================================")
    print("🏥 AI ONCOLOGY DIAGNOSTIC REPORT (V4)")
    print("==================================================")
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly identified Healthy): {cm[0][0]}")
    print(f"False Positives (Falsely called Cancer):     {cm[0][1]}")
    print(f"False Negatives (Missed the Cancer):         {cm[1][0]}")
    print(f"True Positives (Correctly identified Cancer):  {cm[1][1]}\n")
    
    print("Detailed Accuracy Report:")
    print(classification_report(all_labels, all_preds, target_names=['Healthy (0.0)', 'Cancer (1.0)']))

if __name__ == '__main__':
    evaluate_model()
