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

# ==========================================
# 1. CONFIGURATION
# ==========================================
ZIP_PATH = '/tmp/local_dataset.zip'
TMP_DIR = '/tmp/ecg_v5_training_buffers'

# FIXED: Now points directly to your V5 index file!
ML_INDEX_CSV = '/home/conrad.brabec/detector_v5/dataset_index_v5.csv'
PHYSICAL_MAP_CSV = '/home/conrad.brabec/ecg_file_map.csv'

BATCH_SIZE = 128
EPOCHS = 10

# ==========================================
# 2. DATASET LOADER (USING THE ML SPLIT CSV)
# ==========================================
class MasterECGDataset(Dataset):
    def __init__(self, split_target='train'):
        print(f"⏳ Loading {split_target.upper()} cohort from ML Index...")
        
        # Load the balanced 1/0 labels and splits
        df_ml = pd.read_csv(ML_INDEX_CSV)
        
        # Bulletproof the split names (forces all lowercase, strips hidden spaces)
        df_ml['split'] = df_ml['split'].astype(str).str.lower().str.strip()
        
        # Filter down to just the requested split (e.g., 'train')
        df_split = df_ml[df_ml['split'] == split_target].copy()
        df_split['subject_id'] = df_split['subject_id'].astype(str)
        
        print(f"📊 Found {len(df_split)} patients explicitly assigned to {split_target.upper()}.")

        # Load the physical file map
        print("⏳ Fusing with physical ZIP paths...")
        df_map = pd.read_csv(PHYSICAL_MAP_CSV)
        
        # Safely find the path column
        path_col = next((col for col in df_map.columns if col.lower() in ['path', 'record_path', 'record_name', 'file_path']), None)
        if path_col is None:
            raise KeyError(f"Could not find path column in {PHYSICAL_MAP_CSV}")
            
        df_map.rename(columns={path_col: 'record_path'}, inplace=True)
        df_map = df_map.dropna(subset=['record_path'])

        # Extract the 8-digit patient ID using Regex
        df_map['clean_id'] = df_map['record_path'].astype(str).str.extract(r'(?:p)?(\d{8})', expand=False)
        df_map['clean_id'] = df_map['clean_id'].astype(str)
        
        # Merge them together!
        merged_df = pd.merge(df_split, df_map, left_on='subject_id', right_on='clean_id', how='inner')
        merged_df = merged_df.drop_duplicates(subset=['clean_id'])
        self.records = merged_df.to_dict('records')

        if len(self.records) == 0:
            raise ValueError(f"CRITICAL: Merge resulted in 0 patients for the {split_target} split!")

        # Extract to NVMe (Only extracts the exact files needed for this split)
        print(f"📦 Extracting {len(self.records)} {split_target} files to NVMe storage...")
        os.makedirs(TMP_DIR, exist_ok=True)
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            for r in tqdm(self.records, desc=f"Unpacking {split_target}"):
                raw_path = str(r['record_path'])
                clean_zip_path = raw_path.replace('.dat', '').replace('.hea', '')
                base_name = clean_zip_path.split('/')[-1]
                
                # Check if we already extracted it
                local_hea = os.path.join(TMP_DIR, base_name + '.hea')
                if not os.path.exists(local_hea):
                    with open(local_hea, 'wb') as f_out:
                        f_out.write(zf.read(clean_zip_path + '.hea'))
                    with open(os.path.join(TMP_DIR, base_name + '.dat'), 'wb') as f_out:
                        f_out.write(zf.read(clean_zip_path + '.dat'))
                
                r['local_base'] = os.path.join(TMP_DIR, base_name)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_data = self.records[idx]
        local_base = record_data['local_base']
        true_label = record_data['label']

        # Load the physical waves
        record = wfdb.rdrecord(local_base)
        signal = record.p_signal
        signal = np.nan_to_num(signal)

        # Z-Score Normalization
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0) + 1e-8
        signal = (signal - mean) / std

        signal = np.transpose(signal)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor([true_label], dtype=torch.float32)

# ==========================================
# 3. V5 AI ARCHITECTURE
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
# 4. TRAINING ENGINE
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 V5 Training Engine Target: {device}")

    # Explicitly call only the 'train' split!
    train_dataset = MasterECGDataset(split_target='train')
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = CancerDetector1DCNN().to(device)
    if torch.cuda.device_count() > 1:
        print(f"⚡ Utilizing {torch.cuda.device_count()} GPUs for DataParallel!")
        model = nn.DataParallel(model)

    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n📁 --- INITIATING BALANCED EPOCH EXECUTION ---")
    
    # --- THE DEBUG CHECK ---
    print(f"DEBUG: Dataloader has successfully mapped {len(dataloader)} batches!")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for signals, labels in progress:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({'loss': f"{(running_loss / (progress.n + 1)):.4f}"})

    print("✅ Training Complete. Saving V5 Model Weights...")
    torch.save(model.state_dict(), '/home/conrad.brabec/detector_v5/ecg_cancer_detector.pt')
    print("✅ Pipeline safely terminated.")
