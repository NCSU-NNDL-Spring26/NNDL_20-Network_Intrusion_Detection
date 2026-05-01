# Multi-Class Network Intrusion Detection

Neural Networks course project using a hybrid deep learning model for network intrusion detection on the NSL-KDD dataset.

## Overview

A custom **C-LSTM (Hybrid CNN-LSTM)** architecture is trained and evaluated on a 23-class classification task (1 normal + 22 attack types).

Primary evaluation metric: **Macro F1** (preferred over Weighted F1 under heavy class imbalance).


## Dataset

**NSL-KDD** - an improved version of the KDD Cup 1999 dataset. The training file (`KDDTrain+.txt`) contains 41 raw features plus an attack-type label and difficulty score (the latter is dropped). Categorical features are one-hot encoded; all features are standardized via `StandardScaler`.

Classes include: `normal`, `neptune`, `satan`, `ipsweep`, `portsweep`, `smurf`, `back`, `teardrop`, `warezclient`, `pod`, `guess_passwd`, and 12 others. Several classes (e.g. `perl`, `spy`, `multihop`) have near-zero support in the test split - treat their per-class metrics as unreliable.

## Pipeline

1. **Load & encode** - label-encode targets, one-hot encode categorical features
2. **Split** - 72% train / 8% val / 20% test (stratified)
3. **Scale** - `StandardScaler` fit on train only
4. **SMOTE** - capped oversampling applied to **training set only** (minority classes capped at 10% of majority count)
5. **Train** - C-LSTM on processed data
6. **Evaluate** - classification report, confusion matrix heatmap, and F1 scores

## Model
 
The project explores a progression of hybrid CNN-LSTM architectures, each building on the last:
 
### C-LSTM (Hybrid CNN-LSTM) - Baseline
Custom architecture defined in `Hybrid_CNN_LSTM`:
- **1D CNN** (`Conv1d` -> `ReLU` -> `MaxPool1d`) extracts local feature patterns
- **LSTM** captures sequential dependencies across the feature map
- **Fully connected head** outputs 23-class logits
- Loss: `CrossEntropyLoss` with inverse-frequency class weights
- Optimizer: Adam
### C-BiLSTM (Bidirectional)
Replaces the unidirectional LSTM with a **BiLSTM**, allowing the model to attend to context in both directions across the CNN feature map.
 
### C-LSTM + Attention
Adds a **self-attention layer**  to let the model weight the most discriminative time steps before classification.
 
### C-LSTM + Focal Loss
Swaps `CrossEntropyLoss` for **Focal Loss**, which down-weights easy examples and focuses training on hard-to-classify minority attack types.

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
torch
seaborn
matplotlib
```

Install with:
```bash
pip install pandas numpy scikit-learn imbalanced-learn torch seaborn matplotlib
```

The notebook was developed in **Google Colab** (Python 3.12, CUDA 12.8). GPU acceleration is used automatically if available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

## Usage

1. Upload `KDDTrain+.txt` when prompted (Colab file upload cell), or adjust the path for local execution.
2. Run all cells in order.
3. Outputs: classification report, confusion matrix heatmap, and F1 scores.
