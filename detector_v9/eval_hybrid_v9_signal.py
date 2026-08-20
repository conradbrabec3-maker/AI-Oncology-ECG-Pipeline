import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Import the V9 architecture with SE Attention and the Dataset
from train_hybrid_v9_signal import MasterECGDataset, HybridCancerDetectorV9, MODEL_SAVE_PATH, BATCH_SIZE

def evaluate_v9():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Initializing V9 Attention-Driven Evaluation on: {device}")

    # 1. Load the unseen Test Split (Augmentation is automatically bypassed)
    print("⏳ Loading pristine, un-augmented TEST cohort...")
    test_dataset = MasterECGDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Initialize Model and Handle DataParallel Keys
    model = HybridCancerDetectorV9().to(device)
    
    # Strip the "module." prefix added by nn.DataParallel during training
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.eval() # Lock the weights and disable Dropout/BatchNorm updating

    all_preds = []
    all_labels = []

    print("🚀 Commencing AI Diagnostic Scan...")
    with torch.no_grad(): # Disable gradients for pure inference speed
        for sig, hrv, lbl in tqdm(test_loader, desc="Evaluating V9 Test Cohort"):
            sig, hrv = sig.to(device), hrv.to(device)
            
            raw_logits = model(sig, hrv)
            probabilities = torch.sigmoid(raw_logits)
            
            # Standard ML 50% threshold for the baseline report
            predictions = (probabilities >= 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(lbl.numpy())

    # 3. Calculate and Format Results
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print("\n============================================================")
    print("🏥 V9 AI DIAGNOSTIC REPORT (SE-ATTENTION HYBRID)")
    print("============================================================")
    print(f"True Negatives (Healthy): {tn}  |  False Positives: {fp}")
    print(f"False Negatives (Missed): {fn}  |  True Positives (Caught Cancer): {tp}")
    print("------------------------------------------------------------")
    print(classification_report(all_labels, all_preds, target_names=['Healthy (0.0)', 'Cancer (1.0)'], digits=4))
    print("============================================================\n")

if __name__ == "__main__":
    evaluate_v9()
