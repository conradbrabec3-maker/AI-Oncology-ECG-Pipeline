import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import zipfile
import wfdb
import os
import io
from scipy.signal import find_peaks

class MultimodalECGDatasetV10(Dataset):
    def __init__(self, csv_path, zip_path):
        """
        csv_path: Path to train_v10.csv or test_v10.csv
        zip_path: Path to the node-local zipped archive (e.g., in /tmp/)
        """
        self.df = pd.read_csv(csv_path)
        self.zip_path = zip_path
        self.zip_ref = None # Will be opened per-worker to prevent I/O thrashing

    def __len__(self):
        return len(self.df)

    def extract_hrv_features(self, signal):
        # Using Lead II (index 1) for peak detection
        lead_ii = signal[:, 1]
        
        # Calculate peaks (adjust distance/prominence to your baseline sampling rate)
        peaks, _ = find_peaks(lead_ii, distance=150, prominence=0.5)
        
        if len(peaks) < 2:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
        # Calculate N-N intervals (in index steps, can multiply by ms conversion if needed)
        nn_intervals = np.diff(peaks)
        
        mean_nn = np.mean(nn_intervals)
        sdnn = np.std(nn_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(nn_intervals)))) if len(nn_intervals) > 1 else 0.0
        
        return np.array([mean_nn, sdnn, rmssd], dtype=np.float32)

    def __getitem__(self, idx):
        if self.zip_ref is None:
            self.zip_ref = zipfile.ZipFile(self.zip_path, 'r')
            
        row = self.df.iloc[idx]
        record_path = row['record_path'] # Assumes path looks like "files/p100/p1000.../record"
        
        # 1. Dynamically read .dat and .hea from the zip (The Storage Hack)
        # Using WFDB's physical file reader by staging to a fast memory buffer
        # Note: Depending on your exact WFDB setup, you might need to extract the 2 files temporarily
        # For brevity, this assumes you extract the current specific record to /tmp/current_record
        
        # [Implementation of signal reading goes here, returning a (5000, 12) numpy array]
        # Example dummy signal:
        signal = np.random.randn(5000, 12).astype(np.float32) 
        
        # Transpose signal for PyTorch 1D-CNN (Requires Channels First: 12, 5000)
        signal_tensor = torch.tensor(signal.T, dtype=torch.float32)
        
        # 2. Extract HRV from the raw signal
        hrv_array = self.extract_hrv_features(signal)
        hrv_tensor = torch.tensor(hrv_array, dtype=torch.float32)
        
        # 3. Extract Demographics (Age and Gender)
        demo_array = np.array([row['age_normalized'], row['gender_encoded']], dtype=np.float32)
        demo_tensor = torch.tensor(demo_array, dtype=torch.float32)
        
        # 4. Target Label
        label = torch.tensor([row['is_cancer']], dtype=torch.float32)
        
        return signal_tensor, hrv_tensor, demo_tensor, label
