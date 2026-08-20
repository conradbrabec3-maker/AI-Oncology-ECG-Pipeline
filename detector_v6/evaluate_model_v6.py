import os
import torch
import pandas as pd
import numpy as np
import zipfile
import wfdb
import shutil
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from train_model_v6 import CancerDetector1DCNN, MasterECGDataset, BATCH_SIZE

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerDetector1DCNN()
    
    # Load V6 Weights safely
    state_dict = torch.load('/home/conrad.brabec/detector_v6/ecg_cancer_detector_v6.pt', map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.to(device).eval()

    # Load ONLY the 'test' split
    test_loader = DataLoader(MasterECGDataset('test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for sig, lbl in tqdm(test_loader, desc="Evaluating V6 Test Cohort"):
            preds = (model(sig.to(device)) >= 0.5).float()
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(lbl.view(-1).cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\n🏥 V6 AI DIAGNOSTIC REPORT (VERIFIED TEST SET ONLY)")
    print(f"True Negatives (Healthy): {cm[0][0]}  |  False Positives: {cm[0][1]}")
    print(f"False Negatives (Missed): {cm[1][0]}  |  True Positives (Caught Cancer): {cm[1][1]}\n")
    print(classification_report(all_labels, all_preds, target_names=['Healthy (0.0)', 'Cancer (1.0)']))
