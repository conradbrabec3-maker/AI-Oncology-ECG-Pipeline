import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import wfdb
import numpy as np
import zipfile
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
HOME_DIR = "/home/conrad.brabec"
CONTROL_TXT = os.path.join(HOME_DIR, "cohort_3_controls.txt")
LOCAL_ZIP = "/tmp/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip"
MODEL_PATH = os.path.join(HOME_DIR, "ecg_cancer_detector.pt")
OUTPUT_CSV = os.path.join(HOME_DIR, "ai_control_predictions.csv")

# --- 1. THE OPTIMIZED DATASET BUILDER ---
class ECGInferenceDataset(Dataset):
    def __init__(self, cohort_file, zip_path):
        self.zip_path = zip_path
        self.tmp_dir = "/tmp/ecg_inference_buffers"
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        print("⏳ Loading pre-indexed CSV map...")
        df = pd.read_csv('/home/conrad.brabec/ecg_file_map.csv')
        
        # Grab the 4th chunk (p10000032), and strip the 'p' so it becomes '10000032'
        clean_csv_ids = df['record_path'].str.split('/').str[3].str.replace('p', '', case=False)
        
        # Load targets from your text file
        with open(cohort_file, 'r') as f:
            raw_lines = f.readlines()
            targets = [line.strip().replace('p', '').replace('P', '') for line in raw_lines]
            
        # Filter map to only include patients in our control cohort
        self.valid_records = df[clean_csv_ids.isin(targets)]['record_path'].tolist()
        
        print(f"🔍 Successfully matched {len(self.valid_records)} ECG records!")
        
        # FIX: Do NOT open the ZIP file here. Multiprocessing workers will fight over it.
        # We set it to None, and let each worker open its own handle in __getitem__
        self.zf = None

    def __len__(self):
        return len(self.valid_records)

    def __getitem__(self, idx):
        # FIX: Lazy-load the zip file. This ensures each of the 4 workers gets its own safe handle.
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, 'r')
            
        record_path = self.valid_records[idx]
        patient_id = str(record_path).split('/')[-2] 
        
        micro_folder = os.path.join(self.tmp_dir, f"inf_{idx}")
        os.makedirs(micro_folder, exist_ok=True)
        
        try:
            self.zf.extract(record_path + '.hea', micro_folder)
            self.zf.extract(record_path + '.dat', micro_folder)
            
            local_record = os.path.join(micro_folder, record_path)
            record = wfdb.rdrecord(local_record)
            
            signal = record.p_signal
            signal = np.nan_to_num(signal)
            signal = np.transpose(signal)
            
            return torch.tensor(signal, dtype=torch.float32), patient_id
        finally:
            shutil.rmtree(micro_folder, ignore_errors=True)

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

# --- 3. THE INFERENCE LOOP ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference Engine Target: {device}")
    
    # Load Model
    model = CancerDetector1DCNN().to(device)
    
    # Safely load weights and strip 'module.' prefix from DataParallel
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval() # Tell PyTorch we are testing, not training
    
    # Load Data
    dataset = ECGInferenceDataset(cohort_file=CONTROL_TXT, zip_path=LOCAL_ZIP)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    results = []
    
    print("⏳ Running healthy controls through the AI...")
    with torch.no_grad(): # Turn off gradient math to save memory and maximize speed
        for signals, patient_ids in tqdm(dataloader, desc="Predicting"):
            signals = signals.to(device)
            predictions = model(signals).squeeze()
            
            # Save predictions
            if predictions.dim() == 0:
                results.append({"Patient_ID": patient_ids[0], "Cancer_Probability": f"{predictions.item()*100:.2f}%"})
            else:
                for i in range(len(patient_ids)):
                    results.append({"Patient_ID": patient_ids[i], "Cancer_Probability": f"{predictions[i].item()*100:.2f}%"})
                
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Predictions saved to {OUTPUT_CSV}")
