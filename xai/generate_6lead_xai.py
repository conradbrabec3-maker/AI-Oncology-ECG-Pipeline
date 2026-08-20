import os
import re
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
TMP_DIR = '/tmp/xai_highres'
MODEL_PATH = '/home/conrad.brabec/detector_v4/ecg_cancer_detector.pt'
OUTPUT_IMAGE = 'high_res_6lead_xai.png'

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

# --- 3. High-Resolution 6-Lead Plotting Function ---
def plot_6lead_heatmap(ecg_signal, attributions, probability, save_path):
    # SLICE DATA: Keep only the first 6 leads (Limb Leads)
    ecg_6 = ecg_signal[:6, :]
    attr_6 = attributions[:6, :]
    
    lead_names = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF']
    time_axis = np.arange(5000) / 500.0  # Assumes 500Hz sampling rate (10 seconds)
    
    # Define a clean, publication-ready colormap (Red/Blue Diverging)
    # Red = Pushes model toward "Cancer", Blue = Pushes model toward "Healthy"
    cmap = plt.get_cmap('RdBu_r')
    vmax = np.max(np.abs(attr_6))  # Center the colormap dynamically on zero
    vmin = -vmax

    # Create a massive figure to prevent the "squish" effect
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(18, 16), dpi=600, sharex=True)
    fig.suptitle(f"1D-CNN Autonomic Signature Attributions (Limb Leads)\nAI Cancer Probability: {probability:.2f}%", 
                 fontsize=22, fontweight='bold', y=0.92)

    for i in range(6):
        ax = axes[i]
        
        # Plot the baseline ECG waveform in a faint grey so the AI colors pop
        ax.plot(time_axis, ecg_6[i], color='black', alpha=0.3, linewidth=1.5, zorder=1)
        
        # Overlay the XAI attributions as a rich, color-mapped scatter
        sc = ax.scatter(time_axis, ecg_6[i], c=attr_6[i], cmap=cmap, vmin=vmin, vmax=vmax, 
                        s=15, zorder=2, edgecolor='none')
        
        # Professional formatting
        ax.set_ylabel(lead_names[i], fontsize=16, fontweight='bold', labelpad=15)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Global X-Axis formatting
    axes[-1].set_xlabel("Time (Seconds)", fontsize=16, fontweight='bold', labelpad=15)
    
    # Add a global colorbar indicating AI Importance
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Integrated Gradients Attribution Score', fontsize=14, fontweight='bold')
    
    # Save the masterpiece
    plt.subplots_adjust(right=0.9) # Leave room for colorbar
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

# --- 4. Main Execution Pipeline ---
def main():
    print("⏳ Initializing High-Res 6-Lead XAI Pipeline...")
    os.makedirs(TMP_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Math Engine: {device}")

    # Load Model
    model = CancerDetector1DCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    ig = IntegratedGradients(model)

    # 1. Grab ONE specific cancer patient file to map (using your file map)
    df_map = pd.read_csv('/home/conrad.brabec/ecg_file_map.csv')
    
    # We will just pick the very first valid row for the visual demonstration
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
        
        # 2. Run Forward Pass (Requires 12 Leads)
        sample_ecg = torch.tensor(signal, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
        sample_ecg.requires_grad_()
        
        with torch.no_grad():
            pred_prob = model(sample_ecg).item() * 100
            
        print(f"🔬 AI Cancer Prediction Probability: {pred_prob:.2f}%")
        print("🧠 Calculating Integrated Gradients...")
        
        # Calculate Attributions (Requires 12 Leads)
        attributions = ig.attribute(sample_ecg, target=0)
        
        # Detach to standard arrays
        ecg_np = sample_ecg.squeeze().cpu().detach().numpy()
        attr_np = attributions.squeeze().cpu().detach().numpy()
        
        print("🎨 Rendering 6-Lead High-Resolution Plot...")
        # 3. Call the plotting function (Which slices to 6 leads internally)
        plot_6lead_heatmap(ecg_np, attr_np, pred_prob, OUTPUT_IMAGE)
        
        # Clean up
        os.remove(file_path_base + '.dat')
        os.remove(file_path_base + '.hea')

    print(f"✅ Success! Image saved as: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
