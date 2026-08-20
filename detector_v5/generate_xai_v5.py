import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import zipfile
import shutil
from captum.attr import IntegratedGradients

# ==========================================
# 1. CONFIGURATION
# ==========================================
ZIP_PATH = '/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
ML_INDEX_CSV = '/home/conrad.brabec/SQL/master_dataset_index.csv'
PHYSICAL_MAP_CSV = '/home/conrad.brabec/ecg_file_map.csv' 
MODEL_PATH = '/home/conrad.brabec/detector_v5/ecg_cancer_detector.pt'
OUT_DIR = '/home/conrad.brabec/xaiImg_v5'

# ==========================================
# 2. V5 MODEL ARCHITECTURE
# ==========================================
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

def main():
    print("⏳ Initializing V5 12-Lead XAI Pipeline...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # --- Load the V5 Model Safely ---
    model = CancerDetector1DCNN()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    # --- Pick a "TEST" Patient Safely ---
    print("⏳ Searching for a valid TEST patient...")
    df_ml = pd.read_csv(ML_INDEX_CSV)
    df_ml['split'] = df_ml['split'].astype(str).str.lower().str.strip()
    
    # We strictly want a Cancer patient (label == 1.0) who is in the 'test' set
    df_test_cancer = df_ml[(df_ml['split'] == 'test') & (df_ml['label'] == 1.0)]
    
    if df_test_cancer.empty:
        print("❌ ERROR: No cancer patients found in the test split!")
        return
        
    target_id = str(df_test_cancer.sample(n=1).iloc[0]['subject_id'])
    
    # Map physical path
    df_map = pd.read_csv(PHYSICAL_MAP_CSV)
    path_col = next((col for col in df_map.columns if col.lower() in ['path', 'record_path', 'record_name', 'file_path']), None)
    df_map['clean_id'] = df_map[path_col].astype(str).str.extract(r'(?:p)?(\d{8})', expand=False)
    
    patient_record = df_map[df_map['clean_id'] == target_id]
    if patient_record.empty:
        print(f"❌ ERROR: Patient {target_id} not found in physical map.")
        return
        
    raw_path = str(patient_record.iloc[0][path_col]).replace('.dat', '').replace('.hea', '')

    # --- Extract from ZIP ---
    print(f"📦 Extracting Patient {target_id} directly from ZIP archive...")
    tmp_dir = f'/tmp/xai_patient_{target_id}'
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            # Flattens the file path so it unzips neatly into our tmp folder
            base_name = raw_path.split('/')[-1]
            with open(os.path.join(tmp_dir, base_name + '.hea'), 'wb') as f_out:
                f_out.write(z.read(raw_path + '.hea'))
            with open(os.path.join(tmp_dir, base_name + '.dat'), 'wb') as f_out:
                f_out.write(z.read(raw_path + '.dat'))
    except Exception as e:
        print(f"❌ ERROR reading ZIP: {e}")
        return

    # --- Load and NORMALIZE (The V5 Math Fix) ---
    print(f"✅ Found Match! Loading physical ECG waveform for Patient {target_id}...")
    local_record = os.path.join(tmp_dir, base_name)
    record = wfdb.rdrecord(local_record)

    signal = np.nan_to_num(record.p_signal)
    
    # Z-SCORE NORMALIZATION (CRITICAL FOR V5 XAI)
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0) + 1e-8
    signal = (signal - mean) / std

    sample_ecg = torch.tensor(signal, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    sample_ecg.requires_grad_()

    # --- Get AI Prediction ---
    with torch.no_grad():
        raw_output = model(sample_ecg).item()
        pred_prob = raw_output * 100
        ai_diagnosis = "CANCER" if raw_output > 0.5 else "HEALTHY"

    print("==================================================")
    print(f"🩺 GROUND TRUTH:    CANCER")
    print(f"🤖 AI PREDICTION:   {ai_diagnosis} ({pred_prob:.2f}% Probability)")
    print("==================================================")

    # --- Apply XAI (Integrated Gradients) ---
    print("🔍 Calculating Backward Gradients across all 12 Leads...")
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(sample_ecg, target=0, return_convergence_delta=True)

    attributions_np = attributions.squeeze().detach().numpy()
    ecg_np = sample_ecg.squeeze().detach().numpy()

    # --- Plot the Image ---
    print("🎨 Generating 12-Panel Image Grid...")
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(6, 2, figsize=(22, 18), sharex=True)
    fig.suptitle(f"Patient Record: p{target_id}\nGround Truth: CANCER  |  AI Prediction: {ai_diagnosis} ({pred_prob:.1f}%)", 
                 fontsize=22, fontweight='bold')

    for idx, ax in enumerate(axes.flatten()):
        ax.plot(ecg_np[idx], color='black', linewidth=1, alpha=0.3)
        scatter = ax.scatter(np.arange(5000), ecg_np[idx], 
                             c=attributions_np[idx], 
                             cmap='coolwarm', s=8, alpha=0.9)
        ax.set_title(f"Lead {lead_names[idx]}")
        ax.set_ylabel("Standardized mV")
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Time (Samples)")
    axes[-1, 1].set_xlabel("Time (Samples)")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label="Attribution Score (Red = Pushed toward prediction)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    output_image = os.path.join(OUT_DIR, f'xai_12lead_p{target_id}.png')
    plt.savefig(output_image, dpi=300)
    print(f"✅ Full 12-Lead Visualization saved to: {output_image}")
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
