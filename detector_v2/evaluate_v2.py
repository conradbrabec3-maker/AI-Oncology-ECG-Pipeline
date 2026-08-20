import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wfdb
import numpy as np
import zipfile
import shutil
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
HOME_DIR = "/home/conrad.brabec"
V2_DIR = "/home/conrad.brabec/detector_v2"

# Updated to look inside detector_v2
TEST_CSV = os.path.join(V2_DIR, "test_cohort.csv") 
MODEL_PATH = os.path.join(V2_DIR, "ecg_cancer_detector_v2.pt")

# Kept the same (these are still in your main home folder)
INDEX_MAP = os.path.join(HOME_DIR, "ecg_file_map.csv")
LOCAL_ZIP = "/tmp/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip"

# --- 1. THE DATASET BUILDER ---
class ECGTestDataset(Dataset):
    def __init__(self, cohort_csv, index_csv, zip_path):
        self.zip_path = zip_path
        self.tmp_dir = "/tmp/ecg_test_buffers"
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        print("⏳ Loading labels and map...")
        df_labels = pd.read_csv(cohort_csv)
        df_labels['patient_id'] = df_labels['patient_id'].astype(str)
        
        df_map = pd.read_csv(index_csv)
        df_map['clean_id'] = df_map['record_path'].str.split('/').str[3].str.replace('p', '', case=False)
        
        self.merged_df = pd.merge(df_labels, df_map, left_on='patient_id', right_on='clean_id', how='inner')
        self.records = self.merged_df.to_dict('records')
        
        print(f"🔍 Successfully matched {len(self.records)} ECG records for the Final Exam!")
        self.zf = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, 'r')
            
        record_data = self.records[idx]
        record_path = record_data['record_path']
        true_label = record_data['label']
        
        micro_folder = os.path.join(self.tmp_dir, f"test_{idx}")
        os.makedirs(micro_folder, exist_ok=True)
        
        try:
            self.zf.extract(record_path + '.hea', micro_folder)
            self.zf.extract(record_path + '.dat', micro_folder)
            
            local_record = os.path.join(micro_folder, record_path)
            record = wfdb.rdrecord(local_record)
            
            signal = record.p_signal
            signal = np.nan_to_num(signal)
            signal = np.transpose(signal)
            
            return torch.tensor(signal, dtype=torch.float32), true_label
        finally:
            shutil.rmtree(micro_folder, ignore_errors=True)

# --- 2. THE AI ARCHITECTURE ---
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

# --- 3. THE FINAL EXAM ENGINE ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluation Engine Target: {device}")
    
    # Load Model
    model = CancerDetector1DCNN().to(device)
    
    # Safely load weights and strip 'module.' prefix from DataParallel (since we trained it on multi-GPU but evaluate on single)
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.eval() # Tell PyTorch we are testing, NOT training. Lock the weights!
    
    # Load Data
    dataset = ECGTestDataset(cohort_csv=TEST_CSV, index_csv=INDEX_MAP, zip_path=LOCAL_ZIP)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    all_predictions = []
    all_true_labels = []
    
    print("⏳ Administering Final Exam to the AI...")
    with torch.no_grad(): # Turn off gradient math to save memory and maximize speed
        for signals, true_labels in tqdm(dataloader, desc="Testing"):
            signals = signals.to(device)
            
            # The AI guesses (outputs a probability between 0.0 and 1.0)
            probabilities = model(signals).squeeze().cpu().numpy()
            
            # Convert probabilities to a hard guess (If > 50%, guess Cancer (1.0), else guess Healthy (0.0))
            if probabilities.ndim == 0:
                hard_guesses = [1.0 if probabilities > 0.5 else 0.0]
                true_labels_list = [true_labels.item()]
            else:
                hard_guesses = [1.0 if p > 0.5 else 0.0 for p in probabilities]
                true_labels_list = true_labels.numpy().tolist()
                
            all_predictions.extend(hard_guesses)
            all_true_labels.extend(true_labels_list)
                
    # --- CALCULATE FINAL MEDICAL METRICS ---
    print("\n" + "="*50)
    print(" 🏥 AI ONCOLOGY DIAGNOSTIC REPORT")
    print("="*50)
    
    # Confusion Matrix gives us True Positives, False Positives, etc.
    cm = confusion_matrix(all_true_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly identified Healthy): {cm[0][0]}")
    print(f"False Positives (Falsely called Cancer):     {cm[0][1]}")
    print(f"False Negatives (Missed the Cancer):         {cm[1][0]}")
    print(f"True Positives (Correctly identified Cancer):  {cm[1][1]}")
    
    print("\nDetailed Accuracy Report:")
    target_names = ['Healthy (0.0)', 'Cancer (1.0)']
    print(classification_report(all_true_labels, all_predictions, target_names=target_names))
    print("="*50 + "\n")
