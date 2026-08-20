import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import zipfile
import wfdb
import random
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
ZIP_PATH = '/tmp/local_dataset.zip'
INDEX_MAP = '/home/conrad.brabec/ecg_file_map.csv'
CANCER_TXT = '/home/conrad.brabec/cohort_1_cancer.txt'
HEALTHY_TXT = '/home/conrad.brabec/cohort_3_controls.txt' # <-- Verify this name!
TMP_DIR = '/tmp/ecg_v4_training_buffers'

BATCH_SIZE = 128
EPOCHS = 10

# ==========================================
# 2. DATASET LOADER (50/50 SPLIT + NORMALIZATION)
# ==========================================
class BalancedECGDataset(Dataset):
    def __init__(self):
        print("⏳ Loading patient lists...")
        with open(CANCER_TXT, 'r') as f:
            cancer_ids = [line.strip() for line in f if line.strip()]
        with open(HEALTHY_TXT, 'r') as f:
            healthy_ids = [line.strip() for line in f if line.strip()]

        # FORCE 50/50 SPLIT
        min_size = min(len(cancer_ids), len(healthy_ids))
        random.seed(42)
        balanced_cancer = random.sample(cancer_ids, min_size)
        balanced_healthy = random.sample(healthy_ids, min_size)

        # Create master labels
        records = [{'patient_id': pid, 'label': 1.0} for pid in balanced_cancer] + \
                  [{'patient_id': pid, 'label': 0.0} for pid in balanced_healthy]
        
        df_labels = pd.DataFrame(records)
        print(f"⚖️ Dataset Balanced: {min_size} Cancer vs {min_size} Healthy patients.")

        # Map to physical files
        print("⏳ Mapping balanced cohorts to ZIP archive...")
        df_map = pd.read_csv(INDEX_MAP)
        df_map['clean_id'] = df_map['record_path'].str.split('/').str[3].str.replace('p', '', case=False)
        
        merged_df = pd.merge(df_labels, df_map, left_on='patient_id', right_on='clean_id', how='inner')
        merged_df = merged_df.drop_duplicates(subset=['clean_id'])
        self.records = merged_df.to_dict('records')

        # One-Time Fast Extraction to NVMe
        print(f"📦 Extracting {len(self.records)} balanced patients to NVMe storage ONCE...")
        os.makedirs(TMP_DIR, exist_ok=True)
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            for r in tqdm(self.records, desc="Unpacking to /tmp"):
                path = r['record_path']
                base_name = path.split('/')[-1]
                
                # Extract flat to avoid folder nesting
                with open(os.path.join(TMP_DIR, base_name + '.hea'), 'wb') as f_out:
                    f_out.write(zf.read(path + '.hea'))
                with open(os.path.join(TMP_DIR, base_name + '.dat'), 'wb') as f_out:
                    f_out.write(zf.read(path + '.dat'))
                
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

        # ----------------------------------------------------
        # THE MATH FIX: Z-Score Normalization
        # Prevents the -2000 to +2000 voltages from exploding
        # ----------------------------------------------------
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0) + 1e-8
        signal = (signal - mean) / std

        signal = np.transpose(signal)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor([true_label], dtype=torch.float32)

# ==========================================
# 3. V4 AI ARCHITECTURE (WITH BATCH NORM)
# ==========================================
class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        super(CancerDetector1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), # <-- Shock Absorber 1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), # <-- Shock Absorber 2
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 312, 128),
            nn.BatchNorm1d(128), # <-- Shock Absorber 3
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
    print(f"🚀 V4 Training Engine Target: {device}")

    # Load balanced dataset
    dataset = BalancedECGDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize AI
    model = CancerDetector1DCNN().to(device)
    if torch.cuda.device_count() > 1:
        print(f"⚡ Utilizing {torch.cuda.device_count()} GPUs for DataParallel!")
        model = nn.DataParallel(model)

    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n📁 --- INITIATING BALANCED EPOCH EXECUTION ---")
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

    print("✅ Training Complete. Saving V4 Model Weights...")
    torch.save(model.state_dict(), '/home/conrad.brabec/detector_v4/ecg_cancer_detector.pt')
    print("✅ Pipeline safely terminated.")
