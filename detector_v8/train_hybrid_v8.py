import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import zipfile
import wfdb
import scipy.signal
from tqdm import tqdm

# --- CONFIGURATION ---
ZIP_PATH = '/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
TMP_DIR = '/tmp/ecg_v8_training_buffers' # Bumped to V8
ML_INDEX_CSV = '/home/conrad.brabec/detector_v6/master_dataset_index_v6.csv'
PHYSICAL_MAP_CSV = '/home/conrad.brabec/ecg_file_map.csv'
MODEL_SAVE_PATH = '/home/conrad.brabec/detector_v8/ecg_cancer_detector_v8.pt'

BATCH_SIZE = 128
EPOCHS = 10

class MasterECGDataset(Dataset):
    def __init__(self, split_target='train'):
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
        # 1. Load Raw Signal
        raw_signal = np.nan_to_num(wfdb.rdrecord(self.records[idx]['local_base']).p_signal)
        
        # 2. HRV MATH (With NaN Safeguards)
        lead_II = raw_signal[:, 1]
        peaks, _ = scipy.signal.find_peaks(lead_II, distance=150, prominence=0.3)
        
        if len(peaks) > 2:
            rr_intervals = np.diff(peaks)
            rr_ms = rr_intervals * 2.0  
            mean_rr = np.mean(rr_ms)
            sdnn = np.std(rr_ms)
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_ms))))
        elif len(peaks) == 2:
            rr_intervals = np.diff(peaks)
            rr_ms = rr_intervals * 2.0
            mean_rr = np.mean(rr_ms)
            sdnn, rmssd = 0.0, 0.0
        else:
            mean_rr, sdnn, rmssd = 0.0, 0.0, 0.0
            
        hrv_features = np.array([mean_rr/1000.0, sdnn/100.0, rmssd/100.0], dtype=np.float32)
        hrv_features = np.nan_to_num(hrv_features, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Normalize Signal for the CNN
        mean, std = np.mean(raw_signal, axis=0), np.std(raw_signal, axis=0) + 1e-8
        norm_signal = (raw_signal - mean) / std
        
        # 4. Return Tensors
        sig_tensor = torch.tensor(np.transpose(norm_signal), dtype=torch.float32)
        hrv_tensor = torch.tensor(hrv_features, dtype=torch.float32)
        lbl_tensor = torch.tensor([self.records[idx]['label']], dtype=torch.float32)
        
        return sig_tensor, hrv_tensor, lbl_tensor

class HybridCancerDetectorV8(nn.Module):
    def __init__(self):
        super().__init__()
        # BRANCH 1: CNN Vision
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten()
        )
        
        # V8 UPGRADE: The Feature Squeezer
        # Compresses 19,968 wave features down to 64 highly concentrated clues
        self.cnn_squeezer = nn.Linear(64 * 312, 64)
        
        # V8 UPGRADE: The Balanced Fusion Brain
        # Merges the 64 squeezed wave features with the 3 timing features (64 + 3 = 67)
        self.fc = nn.Sequential(
            nn.Linear(67, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(128, 1)
        )
        
    def forward(self, x_sig, x_hrv): 
        cnn_raw = self.conv(x_sig)
        cnn_squeezed = self.cnn_squeezer(cnn_raw)
        
        # The true multimodal fusion happens here
        combined = torch.cat((cnn_squeezed, x_hrv), dim=1)
        return self.fc(combined)

if __name__ == '__main__':
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 V8 Multimodal Training Target: {device}")
    
    train_loader = DataLoader(MasterECGDataset('train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = nn.DataParallel(HybridCancerDetectorV8()).to(device) if torch.cuda.device_count() > 1 else HybridCancerDetectorV8().to(device)
    
    criterion, optimizer = nn.BCEWithLogitsLoss(), optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for sig, hrv, lbl in progress:
            optimizer.zero_grad()
            loss = criterion(model(sig.to(device), hrv.to(device)), lbl.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix({'loss': f"{(running_loss/(progress.n+1)):.4f}"})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ V8 Multimodal Model Saved: {MODEL_SAVE_PATH}")
