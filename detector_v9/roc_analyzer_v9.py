import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from tqdm import tqdm

# Import the V9 architecture
from train_hybrid_v9_signal import MasterECGDataset, HybridCancerDetectorV9, MODEL_SAVE_PATH, BATCH_SIZE

def run_roc_analysis_v9():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Initializing V9 ROC Analyzer on: {device}")

    # Load the untouched test cohort (Augmentation is off)
    test_dataset = MasterECGDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = HybridCancerDetectorV9().to(device)
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval() 

    all_probs = []
    all_labels = []

    print("🚀 Scanning pristine cohort to calculate V9 probability distributions...")
    with torch.no_grad(): 
        for sig, hrv, lbl in tqdm(test_loader, desc="Generating Probabilities"):
            sig, hrv = sig.to(device), hrv.to(device)
            raw_logits = model(sig, hrv)
            probabilities = torch.sigmoid(raw_logits)
            
            all_probs.extend(probabilities.cpu().numpy().flatten())
            all_labels.extend(lbl.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate ROC and Precision-Recall data
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)

    print("\n============================================================")
    print("📈 V9 CLINICAL THRESHOLD OPTIMIZATION REPORT")
    print("============================================================")

    # 1. The Standard ML Threshold (0.50)
    standard_preds = (all_probs >= 0.50).astype(int)
    std_f1 = f1_score(all_labels, standard_preds)
    print(f"Standard (50%) Cutoff -> F1-Score: {std_f1:.4f}")

    # 2. Finding the optimal threshold for F1-Score
    fscores = (2 * precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(fscores)
    best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
    print(f"Optimal Mathematical Cutoff -> {best_threshold * 100:.1f}% -> Max F1-Score: {fscores[best_f1_idx]:.4f}")

    # 3. High Sensitivity (Oncology) Thresholds
    print("\n--- High Sensitivity Clinical Thresholds ---")
    target_recalls = [0.90, 0.95, 0.98]
    for target in target_recalls:
        idx = np.where(tpr >= target)[0]
        if len(idx) > 0:
            target_idx = idx[0]
            thresh = roc_thresholds[target_idx]
            current_fpr = fpr[target_idx]
            print(f"To catch {target*100:.0f}% of Cancer (Recall): Set Threshold to {thresh*100:.1f}% (False Positive Rate: {current_fpr*100:.1f}%)")

    print("============================================================\n")

if __name__ == "__main__":
    run_roc_analysis_v9()
