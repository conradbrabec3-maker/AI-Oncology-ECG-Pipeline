import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import zipfile
from captum.attr import IntegratedGradients

# --- VERIFY THIS ZIP PATH ---
ZIP_PATH = '/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'

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

def main():
    print("⏳ Initializing Full 12-Lead XAI Pipeline...")
    
    out_dir = '/home/conrad.brabec/xaiImg'
    os.makedirs(out_dir, exist_ok=True)
    
    # --- 2. Load the Model ---
    model = CancerDetector1DCNN()
    model.load_state_dict(torch.load('/home/conrad.brabec/detector_v4/ecg_cancer_detector.pt', map_location='cpu'))
    model.eval()

    # --- 3. Read Cohort and Extract a Real Patient ---
    TEST_TYPE = "CANCER" 

    if TEST_TYPE == "CANCER":
        cohort_file = '/home/conrad.brabec/cohort_1_cancer.txt'
        true_label_text = "CANCER"
    else:
        cohort_file = '/home/conrad.brabec/cohort_3_controls.txt'
        true_label_text = "HEALTHY"

    target_ids = []
    with open(cohort_file, 'r') as f:
        for line in f:
            match = re.search(r'(\d{8})', line)
            if match:
                target_ids.append(match.group(1))

    print(f"📂 Loaded {len(target_ids)} valid 8-digit patient IDs from cohort file.")
    
    df_map = pd.read_csv('/home/conrad.brabec/ecg_file_map.csv')
    df_map['clean_id'] = df_map['record_path'].astype(str).str.extract(r'p(\d{8})')[0]
    
    matched_records = df_map[df_map['clean_id'].isin(target_ids)]
    
    if matched_records.empty:
        print("❌ ERROR: Could not map IDs.")
        return

    # Randomly pick ONE patient to visualize
    sample = matched_records.sample(n=1).iloc[0]
    raw_path = str(sample['record_path'])
    patient_id = 10001186

    # --- NEW: EXTRACT DIRECTLY FROM ZIP TO /tmp ---
    print(f"📦 Extracting Patient {patient_id} directly from ZIP archive...")
    tmp_dir = '/tmp/xai_patient'
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extract(raw_path + '.dat', tmp_dir)
            z.extract(raw_path + '.hea', tmp_dir)
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the ZIP file at {ZIP_PATH}!")
        print("Update the ZIP_PATH variable on line 12 of generate_xai.py to match your zip file name.")
        return
    except KeyError:
        print(f"❌ ERROR: Patient files not found inside the ZIP archive. Check path: {raw_path}")
        return

    # Point wfdb to the extracted files in the tmp directory
    file_path_base = os.path.join(tmp_dir, raw_path)

    print(f"✅ Found Match! Loading physical ECG waveform for Patient {patient_id}...")
    
    try:
        record = wfdb.rdrecord(file_path_base) 
    except FileNotFoundError:
         print(f"❌ ERROR: wfdb could not read the extracted files at {file_path_base}.")
         return

    signal = np.nan_to_num(record.p_signal) # Shape: (5000, 12)
    sample_ecg = torch.tensor(signal, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    sample_ecg.requires_grad_()

    # --- 4. Get AI Prediction ---
    with torch.no_grad():
        raw_output = model(sample_ecg).item()
        pred_prob = raw_output * 100
        ai_diagnosis = "CANCER" if raw_output > 0.5 else "HEALTHY"

    print("==================================================")
    print(f"🩺 GROUND TRUTH:    {true_label_text}")
    print(f"🤖 AI PREDICTION:   {ai_diagnosis} ({pred_prob:.2f}% Probability)")
    print("==================================================")

    # --- 5. Apply XAI (Integrated Gradients) ---
    print("🔍 Calculating Backward Gradients across all 12 Leads...")
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(sample_ecg, target=0, return_convergence_delta=True)

    attributions_np = attributions.squeeze().detach().numpy()
    ecg_np = sample_ecg.squeeze().detach().numpy()

    # --- 6. Generate the Complete 12-Lead Picture ---
    print("🎨 Generating 12-Panel Image Grid...")
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(6, 2, figsize=(22, 18), sharex=True)
    fig.suptitle(f"Patient Record: p{patient_id}\nGround Truth: {true_label_text}  |  AI Prediction: {ai_diagnosis} ({pred_prob:.1f}%)", 
                 fontsize=22, fontweight='bold')

    for idx, ax in enumerate(axes.flatten()):
        ax.plot(ecg_np[idx], color='black', linewidth=1, alpha=0.3)
        scatter = ax.scatter(np.arange(5000), ecg_np[idx], 
                             c=attributions_np[idx], 
                             cmap='coolwarm', s=8, alpha=0.9)
        ax.set_title(f"Lead {lead_names[idx]}")
        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Time (Samples)")
    axes[-1, 1].set_xlabel("Time (Samples)")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label="Attribution Score (Red = Pushed toward prediction)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    output_image = os.path.join(out_dir, f'xai_12lead_p{patient_id}.png')
    plt.savefig(output_image, dpi=300)
    print(f"✅ Full 12-Lead Visualization saved to: {output_image}")
    
    # --- Clean up /tmp to be polite to the cluster ---
    os.remove(file_path_base + '.dat')
    os.remove(file_path_base + '.hea')

if __name__ == "__main__":
    main()
