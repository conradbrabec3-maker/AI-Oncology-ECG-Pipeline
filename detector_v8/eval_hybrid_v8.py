import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Import the V8 architecture and dataset from your training script
from train_hybrid_v8 import MasterECGDataset, HybridCancerDetectorV8, MODEL_SAVE_PATH, BATCH_SIZE

def evaluate_v8():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Initializing V8 Feature-Squeezed Evaluation on: {device}")

    # 1. Load the unseen Test Split
    print("⏳ Loading unseen TEST cohort...")
    test_dataset = MasterECGDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Initialize Model and Handle DataParallel Keys
    model = HybridCancerDetectorV8().to(device)
    
    # Strip the "module." prefix added by nn.DataParallel during training
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.eval() # Lock the weights for pure inference

    all_preds = []
    all_labels = []

    print("🚀 Commencing AI Diagnostic Scan...")
    with torch.no_grad(): # Disable gradients for speed
        for sig, hrv, lbl in tqdm(test_loader, desc="Evaluating V8 Test Cohort"):
            sig, hrv = sig.to(device), hrv.to(device)
            
            # The model outputs raw logits (BCEWithLogitsLoss architecture)
            raw_logits = model(sig, hrv)
            
            # Manually apply Sigmoid to convert to a 0.0 - 1.0 probability
            probabilities = torch.sigmoid(raw_logits)
            
            # Use standard 50% threshold for the baseline report
            predictions = (probabilities >= 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(lbl.numpy())

    # 3. Calculate and Format Results
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print("\n============================================================")
    print("🏥 V8 AI DIAGNOSTIC REPORT (FEATURE-SQUEEZED MULTIMODAL)")
    print("============================================================")
    print(f"True Negatives (Healthy): {tn}  |  False Positives: {fp}")
    print(f"False Negatives (Missed): {fn}  |  True Positives (Caught Cancer): {tp}")
    print("------------------------------------------------------------")
    print(classification_report(all_labels, all_preds, target_names=['Healthy (0.0)', 'Cancer (1.0)'], digits=4))
    print("============================================================\n")

if __name__ == "__main__":
    evaluate_v8()
