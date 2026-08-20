import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wfdb
import zipfile
import matplotlib
matplotlib.use('Agg') # CRITICAL for ARC Cluster: Prevents display rendering crashes
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# --- 1. System & Path Configurations ---
ZIP_PATH = '/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
TMP_DIR = '/tmp/xai_highres_4lead'
MODEL_PATH = '/home/conrad.brabec/detector_v4/ecg_cancer_detector.pt'
OUTPUT_IMAGE = 'high_res_4lead_xai.png'

# --- 2. V4 Model Architecture (Must Accept 12 Leads) ---
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

# --- 3. Filtered High-Resolution 4-Lead Plotting Function ---
def plot_4lead_heatmap(ecg_signal, attributions, probability, save_path):
    # SLICE DATA: Keep only the first 4 leads to maximize vertical spacing
    ecg_4 = ecg_signal[:4, :]
    attr_4 = attributions[:4, :]
    
    lead_names = ['Lead I', 'Lead II', 'Lead III', 'aVR']
    time_axis = np.arange(5000) / 500.0  # Assumes 500Hz sampling rate (10 seconds)
    
    cmap = plt.get_cmap('RdBu_r')
    vmax = np.max(np.abs(attr_4))  
    vmin = -vmax

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 16), dpi=600, sharex=True)
    fig.suptitle(f"1D-CNN Significant Autonomic Signatures (|Attribution| >= 0.002)\nAI Cancer Probability: {probability:.2f}%", 
                 fontsize=24, fontweight='bold', y=0.94)

    for i in range(4):
        ax = axes[i]
        
        # 1. Plot the clean base ECG waveform as a solid black line
        ax.plot(time_axis, ecg_4[i], color='black', alpha=0.85, linewidth=1.2, zorder=1)
        
        # 2. Isolate ONLY significant attribution points where absolute attribution >= 0.002
        lead_abs_attr = np.abs(attr_4[i])
        sig_mask = lead_abs_attr >= 0.002
        
        sig_time = time_axis[sig_mask]
        sig_ecg = ecg_4[i][sig_mask]
        sig_attr = attr_4[i][sig_mask]
        
        # 3. Overlay colored scatter markers exclusively on these critical points
        sc = ax.scatter(sig_time, sig_ecg, c=sig_attr, cmap=cmap, vmin=vmin, vmax=vmax, 
                        s=35, zorder=2, edgecolor='none')
        
        # Professional formatting
        ax.set_ylabel(lead_names[i], fontsize=18, fontweight='bold', labelpad=15)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Global X-Axis formatting
    axes[-1].set_xlabel("Time (Seconds)", fontsize=18, fontweight='bold', labelpad=15)
    
    # Add a global colorbar indicating AI Importance
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Significant Attribution Score', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Save the filtered output
    plt.subplots_adjust(right=0.88, hspace=0.3)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

# --- 4. Main Execution Pipeline ---
def main():
    print("⏳ Initializing Absolute Threshold (0.002) 4-Lead XAI Pipeline...")
    os.makedirs(TMP_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Math Engine: {device}")

    # Load Model
    model = CancerDetector1DCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    ig = IntegratedGradients(model)

    # Grab ONE specific cancer patient file to map
    df_map = pd.read_csv('/home/conrad.brabec/ecg_file_map.csv')
    target_record = df_map.iloc[0] 
    raw_path = str(target_record['record_path'])
    
    print(f"📂 Extracting ECG Record: {raw_path}")

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extract(raw_path + '.dat', TMP_DIR)
        z.extract(raw_path + '.hea', TMP_DIR)
        
        file_path_base = os.path.join(TMP_DIR, raw_path)
        
        # Read the full 12-lead signal
        record = wfdb.rdrecord(file_path_base) 
        signal = np.nan_to_num(record.p_signal) 
        
        # Run Forward Pass (Requires 12 Leads)
        sample_ecg = torch.tensor(signal, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
        sample_ecg.requires_grad_()
        
        with torch.no_grad():
            pred_prob = model(sample_ecg).item() * 100
            
        print(f"🔬 AI Cancer Prediction Probability: {pred_prob:.2f}%")
        print("🧠 Calculating Integrated Gradients...")
        
        # Calculate Attributions
        attributions = ig.attribute(sample_ecg, target=0)
        
        # Detach to standard arrays
        ecg_np = sample_ecg.squeeze().cpu().detach().numpy()
        attr_np = attributions.squeeze().cpu().detach().numpy()
        
        print("🎨 Rendering Filtered 4-Lead High-Resolution Plot...")
        plot_4lead_heatmap(ecg_np, attr_np, pred_prob, OUTPUT_IMAGE)
        
        # Clean up
        os.remove(file_path_base + '.dat')
        os.remove(file_path_base + '.hea')

    print(f"✅ Success! Image saved as: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
