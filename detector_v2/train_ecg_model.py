import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wfdb
import numpy as np
import zipfile
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
HOME_DIR = "/home/conrad.brabec"
TRAIN_CSV = os.path.join(HOME_DIR, "train_cohort.csv")
INDEX_MAP = os.path.join(HOME_DIR, "ecg_file_map.csv")
LOCAL_ZIP = "/tmp/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip"
MODEL_SAVE_PATH = os.path.join(HOME_DIR, "ecg_cancer_detector_v2.pt") # Saving as v2!

# --- 1. THE DATASET BUILDER (Now uses the lightning-fast CSVs) ---
class ECGDataset(Dataset):
    def __init__(self, cohort_csv, index_csv, zip_path):
        self.zip_path = zip_path
        self.tmp_dir = "/tmp/ecg_training_buffers"
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Load the Answer Key (Patient ID -> Cancer/Healthy Label)
        print("⏳ Loading labels and map...")
        df_labels = pd.read_csv(cohort_csv)
        df_labels['patient_id'] = df_labels['patient_id'].astype(str)
        
        # Load the 90GB ZIP Map
        df_map = pd.read_csv(index_csv)
        # Extract the pure ID: e.g. files/p1000/p10000032/s40689238/40689238 -> 10000032
        df_map['clean_id'] = df_map['record_path'].str.split('/').str[3].str.replace('p', '', case=False)
        
        # Merge the Map with the Answer Key (Inner Join drops any missing files automatically)
        self.merged_df = pd.merge(df_labels, df_map, left_on='patient_id', right_on='clean_id', how='inner')
        self.records = self.merged_df.to_dict('records')
        
        print(f"🔍 Successfully matched {len(self.records)} ECG records for training!")
        
        # Multiprocessing safe lock
        self.zf = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, 'r')
            
        record_data = self.records[idx]
        record_path = record_data['record_path']
        label = record_data['label']
        
        micro_folder = os.path.join(self.tmp_dir, f"train_{idx}")
        os.makedirs(micro_folder, exist_ok=True)
        
        try:
            self.zf.extract(record_path + '.hea', micro_folder)
            self.zf.extract(record_path + '.dat', micro_folder)
            
            local_record = os.path.join(micro_folder, record_path)
            record = wfdb.rdrecord(local_record)
            
            signal = record.p_signal
            signal = np.nan_to_num(signal)
            signal = np.transpose(signal) # Shape becomes (12, 5000)
            
            # We now return the raw signal AND the true label!
            return torch.tensor(signal, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)
        finally:
            shutil.rmtree(micro_folder, ignore_errors=True)

# --- 2. THE AI ARCHITECTURE (1D-CNN) ---
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
            nn.Sigmoid() # Squishes the final guess to exactly 0.0 to 1.0
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 3. THE TRAINING ENGINE ---
if __name__ == "__main__":
    # 1. Setup Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Engine Target: {device}")
    
    # 2. Build Model
    model = CancerDetector1DCNN().to(device)
    if torch.cuda.device_count() > 1:
        print(f"🔥 Detected {torch.cuda.device_count()} GPUs. Engaging DataParallel!")
        model = nn.DataParallel(model)

    # 3. Setup Math
    criterion = nn.BCELoss() # Binary Cross Entropy (Perfect for Yes/No questions)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Load Dataset
    dataset = ECGDataset(cohort_csv=TRAIN_CSV, index_csv=INDEX_MAP, zip_path=LOCAL_ZIP)
    # Increased num_workers to feed the GPU faster
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    # 5. Start Training Loop
    EPOCHS = 3 # 3 sweeps through the 117,000 patients is plenty to start
    
    print("⏳ Commencing Phase 3 AI Training...")
    
    for epoch in range(EPOCHS):
        model.train() # Tell PyTorch we are doing calculus, not just inferencing
        total_loss = 0.0
        
        # tqdm creates the progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for signals, labels in progress_bar:
            # Send data to GPU
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Step 1: Clear old math
            optimizer.zero_grad()
            
            # Step 2: The AI guesses
            predictions = model(signals)
            
            # Step 3: Compare guess vs True Label to get the error (loss)
            loss = criterion(predictions, labels)
            
            # Step 4: The heavy backwards calculus to find the bad neurons
            loss.backward()
            
            # Step 5: Update weights
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{(total_loss / (progress_bar.n + 1)):.4f}"})

    print("✅ Training Complete. Saving brain weights...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🔒 AI Brain safely saved to: {MODEL_SAVE_PATH}")
