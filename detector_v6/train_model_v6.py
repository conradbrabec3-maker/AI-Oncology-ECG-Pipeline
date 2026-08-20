import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import zipfile
import wfdb
from tqdm import tqdm

ZIP_PATH = '/tmp/local_dataset_v6.zip'
TMP_DIR = '/tmp/ecg_v6_training_buffers'
ML_INDEX_CSV = '/home/conrad.brabec/detector_v6/master_dataset_index_v6.csv'
PHYSICAL_MAP_CSV = '/home/conrad.brabec/ecg_file_map.csv'
MODEL_SAVE_PATH = '/home/conrad.brabec/detector_v6/ecg_cancer_detector_v6.pt'

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
        signal = np.nan_to_num(wfdb.rdrecord(self.records[idx]['local_base']).p_signal)
        mean, std = np.mean(signal, axis=0), np.std(signal, axis=0) + 1e-8
        return torch.tensor(np.transpose((signal - mean) / std), dtype=torch.float32), torch.tensor([self.records[idx]['label']], dtype=torch.float32)

class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(64*312, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): return self.fc(self.conv(x))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 V6 Training Target: {device}")
    
    train_loader = DataLoader(MasterECGDataset('train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = nn.DataParallel(CancerDetector1DCNN()).to(device) if torch.cuda.device_count() > 1 else CancerDetector1DCNN().to(device)
    criterion, optimizer = nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for sig, lbl in progress:
            optimizer.zero_grad()
            loss = criterion(model(sig.to(device)), lbl.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix({'loss': f"{(running_loss/(progress.n+1)):.4f}"})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ V6 Model Saved: {MODEL_SAVE_PATH}")
