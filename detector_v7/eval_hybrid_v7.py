import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import zipfile
import wfdb
import scipy.signal
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
ZIP_PATH = '/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
TMP_DIR = '/tmp/ecg_v7_eval_buffers'
ML_INDEX_CSV = '/home/conrad.brabec/detector_v6/master_dataset_index_v6.csv'
PHYSICAL_MAP_CSV = '/home/conrad.brabec/ecg_file_map.csv'
MODEL_SAVE_PATH = '/home/conrad.brabec/detector_v7/ecg_cancer_detector_v7.pt'

BATCH_SIZE = 128

# --- 1. IDENTICAL DATASET CLASS ---
class MasterECGDataset(Dataset):
    def __init__(self, split_target='test'):
        df_ml = pd.read_csv(ML_INDEX_CSV)
        df_ml['split'] = df_ml['split'].astype(str).str.lower().str.strip()
        df_split = df_ml[df_ml['split'] == split_target].copy()
        df_split['subject_id'] = df_split['subject_id'].astype(str)

        df_map = pd.read_csv(PHYSICAL_MAP_CSV)
        path_col = next((col for col in df_map.columns if col.lower() in ['path', 'record_path']), None)
        df_map.rename(columns={path_col: 'record_path'}, inplace=True)
        df_map['clean_id'] = df_map['record_path'].astype(str).str.extract(r'(?:p)?(\d{8})', expand=False).astype(str)
        
        merged_df = pd.merge(df_split, df_map, left_on='subject_id', right_on='clean_id', how='inner').drop_duplicates(subset=['clean_id'])
        self.records = merged_df.to_dict('records')

        os.makedirs(TMP_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            for r in tqdm(self.records, desc=f"Unpacking {split_target.upper()}"):
                clean_zip_path = str(r['record_path']).replace('.dat', '').replace('.hea', '')
                base_name = clean_zip_path.split('/')[-1]
                local_hea = os.path.join(TMP_DIR, base_name + '.hea')
                if not os.path.exists(local_hea):
                    with open(local_hea, 'wb') as f_out: f_out.write(zf.read(clean_zip_path + '.hea'))
                    with open(os.path.join(TMP_DIR, base_name + '.dat'), 'wb') as f_out: f_out.write(zf.read(clean_zip_path + '.dat'))
                r['local_base'] = os.path.join(TMP_DIR, base_name)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        raw_signal = np.nan_to_num(wfdb.rdrecord(self.records[idx]['local_base']).p_signal)
        
        lead_II = raw_signal[:, 1]
        peaks, _ = scipy.signal.find_peaks(lead_II, distance=150, prominence=0.3)
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks)
            rr_ms = rr_intervals * 2.0
            mean_rr = np.mean(rr_ms)
            sdnn = np.std(rr_ms)
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_ms))))
        else:
            mean_rr, sdnn, rmssd = 0.0, 0.0, 0.0
            
        hrv_features = np.array([mean_rr/1000.0, sdnn/100.0, rmssd/100.0], dtype=np.float32)

        mean, std = np.mean(raw_signal, axis=0), np.std(raw_signal, axis=0) + 1e-8
        norm_signal = (raw_signal - mean) / std
        
        sig_tensor = torch.tensor(np.transpose(norm_signal), dtype=torch.float32)
        hrv_tensor = torch.tensor(hrv_features, dtype=torch.float32)
        lbl_tensor = torch.tensor([self.records[idx]['label']], dtype=torch.float32)
        
        return sig_tensor, hrv_tensor, lbl_tensor

# --- 2. IDENTICAL MODEL CLASS ---
class HybridCancerDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear((64*312) + 3, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(128, 1) # NO SIGMOID HERE
        )
        
    def forward(self, x_sig, x_hrv): 
        cnn_out = self.conv(x_sig)
        combined = torch.cat((cnn_out, x_hrv), dim=1)
        return self.fc(combined)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 V7 Evaluation Target: {device}")
    
    # Target 'test' split
    test_loader = DataLoader(MasterECGDataset('test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize and load model
    model = HybridCancerDetector()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("\n🚀 Beginning V7 Test Cohort Evaluation...")
    with torch.no_grad():
        progress = tqdm(test_loader, desc="Evaluating")
        for sig, hrv, lbl in progress:
            sig, hrv, lbl = sig.to(device), hrv.to(device), lbl.to(device)
            
            # Get raw logits from model
            logits = model(sig, hrv)
            
            # Manually apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)
            
            # Threshold at 0.5 for binary classification
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    print("\n" + "="*60)
    print("🏥 V7 AI DIAGNOSTIC REPORT (HYBRID MULTIMODAL)")
    print("="*60)
    print(f"True Negatives (Healthy): {tn:<5} |  False Positives: {fp}")
    print(f"False Negatives (Missed): {fn:<5} |  True Positives (Caught Cancer): {tp}")
    print("-" * 60)
    print(classification_report(all_labels, all_preds, target_names=["Healthy (0.0)", "Cancer (1.0)"], digits=4))
    print("="*60 + "\n")
