import torch
import torch.nn as nn

class SE_Block(nn.Module):
    """Squeeze-and-Excitation block to mute ICU noise and amplify biological signals."""
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ECG_Oncology_V10(nn.Module):
    def __init__(self):
        super(ECG_Oncology_V10, self).__init__()
        
        # Branch 1: 1D-CNN Structural Extractor
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), # Shock Absorber
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            SE_Block(channels=64), # Dynamic Noise Filtering
            
            nn.AdaptiveAvgPool1d(1), # Global Average Pooling to prevent massive flattening tensors
            nn.Flatten()
        )
        
        # CNN output will be size 64 (due to 64 channels being Global Average Pooled)
        # HRV output is size 3 (Mean NN, SDNN, RMSSD)
        # Demographic output is size 2 (Age, Gender)
        # Total concatenated size = 64 + 3 + 2 = 69
        
        # Branch 2: Multimodal Fusion & Classification Head
        self.fusion_head = nn.Sequential(
            nn.Linear(69, 32), # Compressing the fused vector
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, signal, hrv, demographics):
        # 1. Scan the waveform
        cnn_features = self.cnn_extractor(signal) # Shape: (Batch, 64)
        
        # 2. Fuse the modalities mathematically (Concatenation)
        # Here we stack them side-by-side along dimension 1
        fused_vector = torch.cat((cnn_features, hrv, demographics), dim=1) # Shape: (Batch, 69)
        
        # 3. Final Prediction
        probability = self.fusion_head(fused_vector)
        return probability
