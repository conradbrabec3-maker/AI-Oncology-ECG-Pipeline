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
ZIP_PATH = '/tmp/local_dataset.zip' 
TMP_DIR = '/tmp/ecg_v5_test_buffers'

ML_INDEX_CSV = '/home/conrad.brabec/detector_v5/dataset_index_v5.csv'
PHYSICAL_MAP_CSV = '/home/conrad.brabec/ecg_file_map.csv' 
MODEL_PATH = '/home/conrad.brabec/detector_v5/ecg_cancer_detector.pt' 

BATCH_SIZE = 128
NUM_WORKERS = 4

# ==========================================
# 2. V5 AI ARCHITECTURE
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
# 3. DATASET LOADER (DYNAMIC EXTRACTION)
# ==========================================
class ECGEvaluationDataset(Dataset):
    def __init__(self):
        self.zip_path = ZIP_PATH
        self.tmp_dir = TMP_DIR
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.zf = None

        print("⏳ Loading TEST cohort from ML Index...")
        df_ml = pd.read_csv(ML_INDEX_CSV)
        df_ml['split'] = df_ml['split'].astype(str).str.lower().str.strip()
        
        df_test = df_ml[df_ml['split'] == 'test'].copy()
        df_test['subject_id'] = df_test['subject_id'].astype(str)
        print(f"📊 Found {len(df_test)} patients explicitly assigned to TEST.")

        print("⏳ Fusing with physical ZIP paths...")
        df_map = pd.read_csv(PHYSICAL_MAP_CSV)
        
        path_col = next((col for col in df_map.columns if col.lower() in ['path', 'record_path', 'record_name', 'file_path']), None)
        if path_col is None:
            raise KeyError(f"Could not find path column in {PHYSICAL_MAP_CSV}")
            
        df_map.rename(columns={path_col: 'record_path'}, inplace=True)
        df_map = df_map.dropna(subset=['record_path'])

        df_map['clean_id'] = df_map['record_path'].astype(str).str.extract(r'(?:p)?(\d{8})', expand=False)
        df_map['clean_id'] = df_map['clean_id'].astype(str)

        self.merged_df = pd.merge(df_test, df_map, left_on='subject_id', right_on='clean_id', how='inner')
        self.merged_df = self.merged_df.drop_duplicates(subset=['clean_id'])
        self.records = self.merged_df.to_dict('records')

        if len(self.records) == 0:
            raise ValueError("CRITICAL: Merge resulted in 0 test patients!")
            
        print(f"✅ Successfully matched {len(self.records)} ECG records for the Final Exam!")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # Open ZIP safely for multiprocessing
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, 'r')
            
        record_data = self.records[idx]
        raw_path = str(record_data['record_path'])
        clean_zip_path = raw_path.replace('.dat', '').replace('.hea', '')
        true_label = record_data['label']
        
        # Create unique micro-folder for this specific file
        micro_folder = os.path.join(self.tmp_dir, f"test_{idx}")
        os.makedirs(micro_folder, exist_ok=True)

        try:
            # Extract on the fly
            self.zf.extract(clean_zip_path + '.hea', micro_folder)
            self.zf.extract(clean_zip_path + '.dat', micro_folder)

            local_record = os.path.join(micro_folder, clean_zip_path)
            record = wfdb.rdrecord(local_record)

            signal = record.p_signal
            signal = np.nan_to_num(signal)
            
            # Z-Score Normalization
            mean = np.mean(signal, axis=0)
            std = np.std(signal, axis=0) + 1e-8
            signal = (signal - mean) / std

            signal = np.transpose(signal)

            return torch.tensor(signal, dtype=torch.float32), torch.tensor([true_label], dtype=torch.float32)
        finally:
            # Immediately delete the micro-folder to save space
            shutil.rmtree(micro_folder, ignore_errors=True)

# ==========================================
# 4. EVALUATION PIPELINE
# ==========================================
def evaluate_model():
    print("⏳ Initializing V5 Evaluation Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference Engine Target: {device}")

    model = CancerDetector1DCNN()
    
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.to(device)
    model.eval() 

    dataset = ECGEvaluationDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds = []
    all_labels = []

    print(f"\n🔍 Running {len(dataset)} patients through the AI...")
    
    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Evaluating"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            preds = (outputs >= 0.5).float()
            
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    print("\n==================================================")
    print("🏥 AI ONCOLOGY DIAGNOSTIC REPORT (V5)")
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
