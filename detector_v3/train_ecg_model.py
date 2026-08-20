import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import zipfile
import wfdb
import numpy as np
import shutil
from tqdm import tqdm

# --- 1. THE DATASET BUILDER ---
class DynamicECGDataset(Dataset):
    def __init__(self, cohort_file, zip_path, tmp_workspace="/tmp/ecg_extracted"):
        self.zip_path = zip_path
        self.workspace = tmp_workspace
        os.makedirs(self.workspace, exist_ok=True)
        
        with open(cohort_file, 'r') as f:
            raw_text = f.read().split()
        self.patient_ids = [p.strip() for p in raw_text if p.strip()]
        
        self.valid_records = []
        
        # 1. Map the files
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            all_files = zf.namelist()
            hea_files = [f for f in all_files if f.endswith('.hea')]
            
            for pid in tqdm(self.patient_ids, desc="Mapping IDs to Files", file=sys.stdout):
                target_str = f"p{pid}"
                matches = [f for f in hea_files if target_str in f]
                for match in matches:
                    self.valid_records.append(match.replace('.hea', ''))
                    
        # 2. Extract ONLY our 59k patients ONCE before training starts
        print(f"\n📦 Extracting {len(self.valid_records)} patients to NVMe storage ONCE...")
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for record_path in tqdm(self.valid_records, desc="Unpacking to /tmp", file=sys.stdout):
                hea_file = record_path + ".hea"
                dat_file = record_path + ".dat"
                
                # Check if it's already extracted to avoid duplicate work
                local_hea = os.path.join(self.workspace, hea_file)
                if not os.path.exists(local_hea):
                    zf.extract(hea_file, self.workspace)
                    zf.extract(dat_file, self.workspace)

    def __len__(self):
        return len(self.valid_records)

    def __getitem__(self, idx):
        # Now we just read the already extracted files directly from the fast NVMe disk!
        record_path = self.valid_records[idx]
        local_base_path = os.path.join(self.workspace, record_path)
        
        record = wfdb.rdrecord(local_base_path)
        signal = record.p_signal 
        
        signal = np.nan_to_num(signal).astype(np.float32)
        signal = np.transpose(signal)
        
        label = torch.tensor(1.0, dtype=torch.float32)
        
        return torch.tensor(signal), label

# --- 2. THE DEEP LEARNING ARCHITECTURE (1D-CNN) ---
class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        super(CancerDetector1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 312, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 3. THE EXECUTION LOOP ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Engine Target: {device}")
    
    HOME_DIR = "/home/conrad.brabec"
    COHORT_TXT = os.path.join(HOME_DIR, "cohort_1_cancer.txt")
    LOCAL_ZIP = "/tmp/local_dataset.zip" 
    
    dataset = DynamicECGDataset(cohort_file=COHORT_TXT, zip_path=LOCAL_ZIP)
    
    # num_workers=4 lets the CPU fetch files in parallel to feed the GPU faster
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    model = CancerDetector1DCNN().to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n📁 --- INITIATING EPOCH MATRIX EXECUTION ---")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        batch_loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", file=sys.stdout, leave=True)
        
        for batch_idx, (signals, labels) in enumerate(batch_loop):
            signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(signals)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_loop.set_postfix(loss=f"{loss.item():.4f}")
                
    print("\n✅ Training Pipeline Complete. Saving Model Weights...")
    torch.save(model.state_dict(), os.path.join(HOME_DIR, "detector_v3", "ecg_cancer_detector.pt"))
