# AI Oncology ECG Pipeline
<div align="center">

**A Deep Learning Framework for Cancer Detection from Electrocardiogram (ECG) Signals using MIMIC-IV Data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)]()

</div>

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [Dataset & Data Pipeline](#-dataset--data-pipeline)
- [Model Architecture Evolution](#-model-architecture-evolution)
  - [Version 1: Baseline CNN](#version-1-baseline-cnn)
  - [Version 2-3: Early Iterations](#version-23-early-iterations)
  - [Version 4: CNN with Image Generation](#version-4-cnn-with-image-generation)
  - [Version 5: Batch Normalization & Dropout](#version-5-batch-normalization--dropout)
  - [Version 6: Optimized Training Pipeline](#version-6-optimized-training-pipeline)
  - [Version 7: Hybrid Multimodal Approach](#version-7-hybrid-multimodal-approach)
  - [Version 8-9: Advanced Iterations](#version-89-advanced-iterations)
- [Feature Engineering Toolkit](#-feature-engineering-toolkit)
- [Data Degradation & Robustness Testing](#-data-degradation--robustness-testing)
- [Performance Metrics](#-performance-metrics)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
- [Project Iterations & Development History](#-project-iterations--development-history)

---

##  Project Overview

**AI Oncology ECG Pipeline** is a comprehensive deep learning framework designed to detect cancer from electrocardiogram (ECG) signals using data from the MIMIC-IV database. The project demonstrates an iterative development approach, evolving through multiple detector versions with increasingly sophisticated architectures and multimodal features.

### Key Features
- **Multi-lead ECG Processing**: Handles 12-lead ECG recordings from MIMIC-IV
- **Deep Learning Architectures**: 1D-CNN models with batch normalization, dropout, and hybrid multimodal approaches
- **Hybrid Feature Fusion**: Combines raw signal processing with heart rate variability (HRV) features
- **Data Degradation Simulation**: Tests model robustness against real-world noise (baseline wander, EMG artifacts)
- **Comprehensive Evaluation**: Includes confusion matrices, classification reports, and XAI visualizations

### Research Focus
This project investigates whether ECG signal characteristics can predict oncology-related conditions through deep learning analysis of:
- Raw multichannel ECG waveforms
- Heart Rate Variability (HRV) metrics (RMSSD, SDNN, mean RR interval)
- Signal morphological patterns across multiple leads

---

##  Dataset & Data Pipeline

### MIMIC-IV ECG Database
The pipeline processes ECG records from the **MIMIC-IV Electrocardiogram Matched Subset**, which contains diagnostic-grade ECG recordings with associated patient metadata.

#### Data Files
```
mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip
├── *.hea (Header files with record metadata)
└── *.dat (Binary signal data in WFDB format)
```

### Database Setup (`SQL/build_db.py`)
The project includes utilities for:
- Ingesting MIMIC-IV diagnostic codes (`d_icd_diagnoses.csv.gz`)
- Mapping diagnoses to patient records (`diagnoses_icd.csv.gz`)
- Creating SQLite database for quick lookups

```python
# Example workflow:
db_path = "/home/conrad.brabec/mimic_data.db"
ingest_file(d_icd_diagnoses_path, "table_name_1", engine)
ingest_file(diagnoses_icd_path, "table_name_2", engine)
```

### Dataset Balancing (`SQL/build_ml_dataset.py`)
The ML dataset is carefully balanced with a strict 50/50 split between:
- **Cancer cohort**: Patients with oncology-related diagnoses (label=1)
- **Healthy control cohort**: Patients without cancer (label=0)

**Train/Validation/Test Split**: 70% / 15% / 15% (stratified to maintain class balance)

---

##  Model Architecture Evolution

The project evolved through multiple detector versions, each representing a significant iteration in design philosophy and technical capability.

### Version 1: Baseline CNN (`run_predictions.py`)
**First implementation of ECG cancer detection using PyTorch.**

```python
class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),  # Reduce temporal resolution
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 312, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(128, 1),
            nn.Sigmoid()      # Binary classification output
        )
```

**Key Innovations:**
- First end-to-end CNN architecture for ECG processing
- Basic two-stage convolutional feature extraction
- Batch size: 64 with 4 parallel workers

---

### Version 2-3: Early Iterations (`detector_v2/`, `detector_v3/`)
**Refinement of baseline model with improved evaluation scripts.**

`detector_v3/train_ecg_model.py` (5,110 bytes) introduced:
- Enhanced dataset loading from ML index CSV
- Proper train/val/test split integration
- Standardized evaluation pipeline with confusion matrix reporting

**Notable files:**
- `ecg_test_v3_43523519.out` - Test set predictions
- `evaluate_v3.py` - Comprehensive evaluation with sklearn metrics

---

### Version 4: CNN with Image Generation (`detector_v4/`)
**Experimented with CM (Confusion Matrix) image generation.**

```python
# Key addition in v4:
class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # ... same architecture as v1/v5 ...
        )

def generate_cm_image():
    """Generate confusion matrix visualization"""
```

**Key files:**
- `generate_cm_image.py` - Creates visualizations of model performance
- `xai_lead_II_analysis.png` - Explainable AI visualization for Lead II analysis (2MB)

---

### Version 5: Batch Normalization & Dropout (`detector_v5/`)
**Introduction of BatchNorm and refined architecture.**

```python
class CancerDetector1DCNN(nn.Module):
    def __init__(self):
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),    # NEW: Stabilizes training
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),    # NEW: Better feature learning
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 312, 128),
            nn.BatchNorm1d(128),   # NEW: Prevents overfitting
            nn.ReLU(),
            nn.Dropout(0.5),       # KEEP: Regularization
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
```

**Configuration:**
- Batch size increased to **128** (improved gradient estimation)
- **Z-Score Normalization**: `(signal - mean) / std` for each lead
- **Dynamic ZIP extraction**: Lazy-loading to support multiprocessing
- **NUM_WORKERS=4** parallel processing

---

### Version 6: Optimized Training Pipeline (`detector_v6/`)
**Streamlined training with unified dataset handling.**

```python
# Unified MasterECGDataset class handles all splits
class MasterECGDataset(Dataset):
    def __init__(self, split_target='train'):
        # Load from single balanced index file
        df_ml = pd.read_csv(ML_INDEX_CSV)
        df_split = df_ml[df_ml['split'] == split_target]
        
        # Dynamic ZIP extraction on-demand
        # Z-score normalization
        # Return: (normalized_signal, label_tensor)
```

**Training Script:**
```python
# detector_v6/train_model_v6.py
model = nn.DataParallel(CancerDetector1DCNN()) if multi_gpu else CancerDetector1DCNN()
criterion, optimizer = nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # EPOCHS = 10
    for sig, lbl in tqdm(train_loader):
        optimizer.zero_grad()
        loss = criterion(model(sig), lbl)
        loss.backward()
        optimizer.step()
```

**Performance:**
- Trained on balanced dataset with exact 50/50 split
- Saved weights: `ecg_cancer_detector_v6.pt`
- Evaluation script outputs confusion matrix and classification report

---

### Version 7: Hybrid Multimodal Approach (`detector_v7/`)
**Major architectural shift to incorporate HRV features.**

```python
class HybridCancerDetector(nn.Module):
    def __init__(self):
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten()
        )
        self.fc = nn.Sequential(
            # CONCATENATE CNN output with HRV features!
            nn.Linear((64*312) + 3, 128),  # 3 HRV features added
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(128, 1)              # NO SIGMOID - raw logits
        )
    
    def forward(self, x_sig, x_hrv):     # TWO INPUTS!
        cnn_out = self.conv(x_sig)
        combined = torch.cat((cnn_out, x_hrv), dim=1)
        return self.fc(combined)
```

**HRV Feature Extraction:**
```python
# From MasterECGDataset.__getitem__()
lead_II = raw_signal[:, 1]
peaks, _ = scipy.signal.find_peaks(lead_II, distance=150, prominence=0.3)

if len(peaks) > 1:
    rr_intervals = np.diff(peaks)
    rr_ms = rr_intervals * 2.0
    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_ms))))
else:
    mean_rr, sdnn, rmssd = 0.0, 0.0, 0.0

hrv_features = np.array([mean_rr/1000.0, sdnn/100.0, rmssd/100.0], 
                        dtype=torch.float32)
```

**Evaluation Results:**
```
============================================================
 V7 AI DIAGNOSTIC REPORT (HYBRID MULTIMODAL)
============================================================
True Negatives (Healthy):  2391 | False Positives:  745
False Negatives (Missed):   1133 | True Positives: 2003

              precision    recall  f1-score   support
 Healthy (0.0)     0.6785    0.7624    0.7180      3136
 Cancer (1.0)     0.7289    0.6387    0.6808      3136

 accuracy                         0.7006      6272
```

---

### Version 8-9: Advanced Iterations (`detector_v8/`, `detector_v9/`)
**Further refinements and experiments.**

Directory structure indicates ongoing development with evaluation scripts, training utilities, and experiment logs.

---

##  Feature Engineering Toolkit

Located in `toolkits/FeatureDB/`, this toolkit provides traditional signal processing features to complement deep learning:

### Available Feature Modules
- **[ECGAvgBeat](toolkits/FeatureDB/tutorials/ECGAvgBeat.md)** - Average beat morphology analysis
- **[FrequencyDomain](toolkits/FeatureDB/tutorials/FrequencyDomain.md)** - FFT-based spectral features
- **[HeartRateVariability](toolkits/FeatureDB/tutorials/HeartRateVariability.md)** - HRV metrics (RMSSD, SDNN, etc.)
- **[MorphologicVariability](toolkits/FeatureDB/tutorials/MorphologicVariability.md)** - Beat-to-beat variability
- **[QRSAxis](toolkits/FeatureDB/tutorials/QRSAxis.md)** - QRS axis analysis
- **[QTDispersion](toolkits/FeatureDB/tutorials/QTDispersion.md)** - QT interval variability

### Citation
```bibtex
@inproceedings{hong2017encase,
  author    = {Shenda Hong and Meng Wu and Yuxi Zhou and Qingyun Wang and Junyuan Shang and Hongyan Li and Junqing Xie},
  title     = {{ENCASE:} an ENsemble ClASsifiEr for {ECG} Classification Using Expert Features and Deep Neural Networks},
  booktitle = {CinC},
  year      = {2017}
}

@article{hong2019combining,
    doi = {10.1088/1361-6579/ab15a2},
    url = {https://doi.org/10.1088%2F1361-6579%2Fab15a2},
    year = 2019,
    publisher = {{IOP} Publishing},
    volume = {40},
    number = {5},
    pages = {054009},
    author = {Shenda Hong and Yuxi Zhou and Meng Wu and Junyuan Shang and Qingyun Wang and Hongyan Li and Junqing Xie},
    title = {Combining deep neural networks and engineered features for cardiac arrhythmia detection from {ECG} recordings},
    journal = {Physiological Measurement}
}
```

---

##  Data Degradation & Robustness Testing (`SmallModel/`)

To test model robustness against real-world noise, the project includes an ECG degradation pipeline:

### Noise Types Simulated
- **Baseline Wander**: Low-frequency drift (0.3 Hz) simulating loose electrode strap movement
- **EMG Artifacts**: High-frequency muscle noise with burst patterns
- Combined degradation for realistic wearable device testing

### Usage
```bash
# Process all records
python ecg_degrade_mimic.py

# Process specific slice (SLURM array job)
python ecg_degrade_mimic.py --start 0 --end 200000

# Override paths via environment variables
export ECG_ZIP_PATH=/path/to/your/mimic-iv-ecg.zip
export ECG_OUT_DIR=/output/directory
python ecg_degrade_mimic.py
```

### Noise Parameters
```python
WANDER_AMPLITUDE = 0.15   # ~15% of signal std → loose-strap drift
WANDER_FREQ_HZ   = 0.3    # Respiration rate band (0.15-0.4 Hz)
EMG_AMPLITUDE    = 0.10   # ~10% of signal std → muscle noise
EMG_BURST_PROB   = 0.02   # Per-sample probability of burst
```

---

## 📊 Performance Metrics

### Version 7 (Hybrid Multimodal) Results
- **Accuracy**: 70.06%
- **Precision (Cancer)**: 72.89%
- **Recall (Cancer)**: 63.87%
- **F1-Score (Cancer)**: 68.08%

### Confusion Matrix
|                 | Predicted Healthy | Predicted Cancer |
|-----------------|-------------------|------------------|
| **Actual Healthy** | 2,391 (TN)       | 745 (FP)         |
| **Actual Cancer**  | 1,133 (FN)       | 2,003 (TP)       |

### Key Observations from Iterations
- **V1-V3**: Baseline CNN architecture establishes feasibility
- **V4**: Image generation experiments for visualization
- **V5**: BatchNorm and larger batch size improve stability
- **V6**: Optimized training pipeline with unified dataset handling
- **V7**: HRV feature fusion provides multimodal advantage
- **V8-V9**: Further architectural refinements

---

## 🔧 Installation & Setup

### Requirements
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scipy wfdb tqdm scikit-learn sqlalchemy
```

### Environment Variables
```bash
# Point to your MIMIC-IV ECG zip file
export ECG_ZIP_PATH=/path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip

# Output directory for degraded data
export ECG_OUT_DIR=/path/to/ecg_degraded_output
```

### Database Setup
```bash
cd SQL
python build_db.py           # Ingest diagnostic codes
python build_ml_dataset.py   # Create balanced train/val/test splits
```

---

##  Usage Examples

### Running Predictions
```bash
python run_predictions.py
# Output: /home/conrad.brabec/ai_control_predictions.csv
```

### Training Model (Version 5)
```bash
cd detector_v5
source /home/conrad.brabec/software_env/bin/activate
sbatch submit_eval_v5.sh    # Or train with appropriate script
```

### Evaluating Version 6
```bash
python detector_v6/evaluate_model_v6.py
# Outputs confusion matrix and classification report
```

### Running Degradation Test
```bash
python SmallModel/ecg_degrade_mimic.py --slice 0
# Outputs noisy ECG CSV files for robustness testing
```

---

##  Project Directory Structure

```
AI-Oncology-ECG-Pipeline/
├── detector_v2/          # Early iterations
├── detector_v3/          # Train + evaluate scripts
│   ├── train_ecg_model.py
│   ├── evaluate_v3.py
│   └── ecg_test_v3_*.out
├── detector_v4/          # Image generation experiments
│   ├── generate_cm_image.py
│   ├── xai_lead_II_analysis.png
│   └── ...
├── detector_v5/          # BatchNorm + Dropout
│   ├── train_ecg_model.py
│   ├── evaluate_v5.py
│   └── submit_eval_v5.sh
├── detector_v6/          # Optimized training
│   ├── train_model_v6.py
│   ├── evaluate_model_v6.py
│   └── submit_train_v6.sh
├── detector_v7/          # Hybrid multimodal (HRV features)
│   ├── eval_hybrid_v7.py
│   └── eval_v7_*.out
├── detector_v8/          # Advanced iterations
├── detector_v9/          # Latest experiments
├── SmallModel/           # Data degradation pipeline
│   ├── ecg_degrade_mimic.py
│   ├── ecg_degrade_mimic.sh
│   └── ecg_degrade_*.out
├── SQL/                  # Database utilities
│   ├── build_db.py       # Ingest diagnostic codes
│   ├── build_ml_dataset.py  # Balanced dataset creation
│   ├── sql_cohort_*.out
│   └── submit_*.sh      # SLURM scripts
├── toolkits/FeatureDB/   # Traditional feature engineering
│   ├── README.md         # Toolkit documentation
│   ├── tutorials/        # Individual feature modules
│   └── tools/            # Feature extraction utilities
├── run_predictions.py    # Main inference script
└── ... (other files)
```

---

##  Project Iterations & Development History

### Iteration 1: Initial Concept (`run_predictions.py`)
- **Goal**: Establish baseline ECG cancer detection capability
- **Innovation**: First CNN architecture for multilead ECG classification
- **Challenge**: Basic design without regularization
- **Solution**: Added dropout (0.5) and simplified architecture

### Iteration 2-3: Training Infrastructure (`detector_v2/v3`)
- **Goal**: Create standardized training/evaluation pipeline
- **Innovation**: Dataset loading from CSV index with split handling
- **Challenge**: Inconsistent data paths between train/test
- **Solution**: Unified `MasterECGDataset` class with lazy ZIP extraction

### Iteration 4: Visualization Experiments (`detector_v4`)
- **Goal**: Develop explainability and visualization tools
- **Innovation**: Confusion matrix image generation, Lead II XAI
- **Challenge**: Model predictions were hard to interpret
- **Solution**: Added `generate_cm_image.py` and visualizations

### Iteration 5: Regularization & Stability (`detector_v5`)
- **Goal**: Improve training stability and generalization
- **Innovation**: Batch Normalization layers, larger batch size (128)
- **Challenge**: Model overfitting on training data
- **Solution**: `BatchNorm1d` after each conv layer + dropout

### Iteration 6: Pipeline Optimization (`detector_v6`)
- **Goal**: Streamline training workflow
- **Innovation**: Unified dataset class, simplified training loop
- **Challenge**: Redundant code between train/eval scripts
- **Solution**: Shared `MasterECGDataset` and model architecture

### Iteration 7: Multimodal Fusion (`detector_v7`)
- **Goal**: Leverage HRV features alongside raw signals
- **Innovation**: Hybrid CNN+HRV architecture with feature concatenation
- **Challenge**: Raw signal alone insufficient for accurate classification
- **Solution**: Combined CNN output (CNN features) with HRV features

### Iteration 8-9: Advanced Refinements
- **Goal**: Further architectural improvements
- **Innovation**: Continued exploration of model variants
- **Status**: Active development area

---

## 🔬 Scientific Contributions

1. **Multilead ECG Processing**: Demonstrated ability to process all 12 leads simultaneously
2. **Hybrid Feature Learning**: Showed benefit of combining raw signals with engineered HRV features
3. **Robustness Testing**: Developed data degradation pipeline for real-world applicability assessment
4. **Iterative Development**: Documented progression through 9+ detector versions
5. **Explainable AI**: Generated XAI visualizations (Lead II analysis images)

---

##  Disclaimer & Limitations

- This is a research project using synthetic/derived labels from MIMIC-IV diagnostic codes
- **NOT** intended for clinical diagnosis or medical decision-making
- Models are trained on balanced, synthetic cohorts—real-world performance may differ
- Always validate with domain experts before clinical application

---

##  License

MIT License - see LICENSE file for details.

---

##  References

1. **MIMIC-IV Database**: https://physionet.org/content/mimiciv/
2. **WFDB Library**: https://wfdb.readthedocs.io/
3. **ENCASE Paper**: Hong et al., "ENCASE: an Ensemble ClASsifiEr for ECG Classification" (CinC 2017)
4. **Feature Combination Paper**: Hong et al., "Combining deep neural networks and engineered features for cardiac arrhythmia detection" (Physiological Measurement, 2019)

---

<div align="center">

**For questions or collaboration, please contact the project author.**

</div>
