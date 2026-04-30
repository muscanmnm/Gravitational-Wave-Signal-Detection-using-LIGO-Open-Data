## Overview
A hybrid CNN-LSTM deep learning pipeline to detect 
gravitational wave signals from raw LIGO strain data, 
trained on the G2Net Kaggle Competition dataset.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 74% |
| ROC-AUC | 0.8025 |
| Signal Precision | 81% |
| Noise Recall | 86% |

## Dataset
- G2Net Gravitational Wave Detection (Kaggle)
- 50,000 samples from 3 LIGO/Virgo detectors
- Binary classification: GW signal (1) vs noise (0)
- Balanced dataset: 50% positive rate

## Preprocessing Pipeline
1. **Bandpass Filter** — Butterworth filter (20-500 Hz)
   - GW signals exist only in this frequency range
   - Removes low-frequency seismic and high-frequency 
     electronic noise
   
2. **Spectral Whitening** — Flatten noise floor
   - Divides signal by its Power Spectral Density (PSD)
   - Makes faint GW chirp visible to the model
   - Most critical step for GW detection

3. **Normalization** — Zero mean, unit variance
   - Applied per detector channel
   - Performed in float64 before float32 conversion
     to avoid underflow (raw strain ~1e-20 scale)

## Model Architecture
Input: (batch, 3, 4096)
↓
CNN Block:
Conv1d(3→32,  k=64, s=4) + BN + ReLU + MaxPool
Conv1d(32→64, k=16, s=2) + BN + ReLU + MaxPool
Conv1d(64→128, k=8, s=2) + BN + ReLU + MaxPool
↓
Output: (batch, 128, ~16)
↓
Reshape: (batch, ~16, 128)
↓
BiLSTM(128→256, 2 layers, dropout=0.3)
↓
Last timestep: (batch, 256)
↓
Classifier: Linear(256→64) → ReLU → Linear(64→1)
↓
Output: (batch,) — binary prediction
## Why CNN-LSTM?
- **CNN** extracts local wave patterns (chirps, oscillations)
  from the raw time series
- **LSTM** captures how these patterns evolve over time
- **Bidirectional** reads signal both forward and backward
- Together they model both local features AND 
  temporal dependencies

## Training
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Epochs: 30 (best model saved at epoch 18)
- Hardware: Tesla T4 GPU

## Tech Stack
Python | PyTorch | SciPy | NumPy | Scikit-learn | Kaggle
"""
print(readme)
