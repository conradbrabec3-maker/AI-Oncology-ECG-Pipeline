import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wfdb
import zipfile
from scipy.signal import find_peaks
from captum.attr import IntegratedGradients
from tqdm import tqdm

# --- VERIFY THIS ZIP PATH ---
ZIP_PATH = '/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
TMP_DIR = '/tmp/xai_batch'
OUTPUT_CSV = '/home/conrad.brabec/xai/global_xai_stats_1.csv'

# --- 1. Define V4 Model Architecture ---
class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        super(CancerDetector1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
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
        return self.fc_layers(self.conv_layers(x))

def extract_biological_weights(ecg_signal, attribution_scores, sampling_rate=500):
    """
    Slices the 5000 XAI values into biological buckets based on R-Peak locations.
    """
    peaks, _ = find_peaks(ecg_signal, height=0.5, distance=int(0.3 * sampling_rate))
    
    if len(peaks) == 0:
        return 0, 0, 0 
        
    qrs_window = int(0.06 * sampling_rate) 
    t_wave_start = int(0.15 * sampling_rate) 
    t_wave_end = int(0.35 * sampling_rate)   
    
    total_attribution = np.sum(np.abs(attribution_scores))
    if total_attribution == 0:
        return 0, 0, 0
        
    qrs_attr, t_wave_attr = 0, 0
    
    for peak in peaks:
        start_qrs = max(0, peak - qrs_window)
        end_qrs = min(len(attribution_scores), peak + qrs_window)
        qrs_attr += np.sum(np.abs(attribution_scores[start_qrs:end_qrs]))
        
        start_t = min(len(attribution_scores), peak + t_wave_start)
        end_t = min(len(attribution_scores), peak + t_wave_end)
        t_wave_attr += np.sum(np.abs(attribution_scores[start_t:end_t]))
        
    rr_attr = total_attribution - (qrs_attr + t_wave_attr)
    
    qrs_pct = (qrs_attr / total_attribution) * 100
    t_wave_pct = (t_wave_attr / total_attribution) * 100
    rr_pct = (rr_attr / total_attribution) * 100
    
    return qrs_pct, t_wave_pct, rr_pct

def main():
    print("⏳ Initializing Global XAI Batch Processing...")
    os.makedirs(TMP_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Math Engine: {device}")

    model = CancerDetector1DCNN()
    model.load_state_dict(torch.load('/home/conrad.brabec/detector_v4/ecg_cancer_detector.pt', map_location=device))
    model.to(device)
    model.eval()
    ig = IntegratedGradients(model)

    cohort_file = '/home/conrad.brabec/cohort_1_cancer.txt'
    target_ids = []
    with open(cohort_file, 'r') as f:
        for line in f:
            match = re.search(r'(\d{8})', line)
            if match:
                target_ids.append(match.group(1))

    # Using first 500 for testing speed.
    target_ids = target_ids[:500] 
    print(f"📂 Scanning {len(target_ids)} patients for Global XAI Stats...")
    
    df_map = pd.read_csv('/home/conrad.brabec/ecg_file_map.csv')
    df_map['clean_id'] = df_map['record_path'].astype(str).str.extract(r'p(\d{8})')[0]
    
    # ---------------------------------------------------------
    # THE FIX: Drop duplicate recordings so we only have 1 per patient
    # ---------------------------------------------------------
    df_map = df_map.drop_duplicates(subset=['clean_id'], keep='first')
    matched_records = df_map[df_map['clean_id'].isin(target_ids)]
    
    results = []
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        
        for index, row in tqdm(matched_records.iterrows(), total=len(matched_records), desc="Extracting Biological Weights"):
            patient_id = row['clean_id']
            raw_path = str(row['record_path'])
            
            try:
                z.extract(raw_path + '.dat', TMP_DIR)
                z.extract(raw_path + '.hea', TMP_DIR)
                
                file_path_base = os.path.join(TMP_DIR, raw_path)
                record = wfdb.rdrecord(file_path_base) 
                signal = np.nan_to_num(record.p_signal) 
                
                sample_ecg = torch.tensor(signal, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
                sample_ecg.requires_grad_()
                
                with torch.no_grad():
                    pred_prob = model(sample_ecg).item() * 100
                
                attributions = ig.attribute(sample_ecg, target=0)
                
                ecg_np = sample_ecg.squeeze().cpu().detach().numpy()[1]
                attr_np = attributions.squeeze().cpu().detach().numpy()[1]
                
                qrs_pct, t_pct, rr_pct = extract_biological_weights(ecg_np, attr_np)
                
                results.append({
                    'Patient_ID': patient_id,
                    'Specific_ECG_Record': raw_path, # <--- Added this so you can verify exactly which ECG was used
                    'Cancer_Probability': round(pred_prob, 2),
                    'QRS_Weight_Pct': round(qrs_pct, 2),
                    'T_Wave_Weight_Pct': round(t_pct, 2),
                    'RR_Interval_Weight_Pct': round(rr_pct, 2)
                })
                
                os.remove(file_path_base + '.dat')
                os.remove(file_path_base + '.hea')

            except Exception as e:
                continue 

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Global XAI Statistics successfully saved to {OUTPUT_CSV}")

    print("==========================================")
    print("📊 MACRO BIOLOGICAL ATTRIBUTION SUMMARY")
    print(f"Average QRS Complex Importance: {df_results['QRS_Weight_Pct'].mean():.2f}%")
    print(f"Average T-Wave Importance:      {df_results['T_Wave_Weight_Pct'].mean():.2f}%")
    print(f"Average R-R Interval Importance: {df_results['RR_Interval_Weight_Pct'].mean():.2f}%")
    print("==========================================")

if __name__ == "__main__":
    main()
