# Multi-Class Network Intrusion Detection — Team 20

Neural Networks course project comparing classical and deep learning approaches for network intrusion detection on the NSL-KDD dataset.

## Overview

Three models are trained and evaluated on a 23-class classification task (1 normal + 22 attack types):

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble baseline |
| C-LSTM (Hybrid CNN-LSTM) | Deep learning model (PyTorch) |

Primary evaluation metric: **Macro F1** (preferred over Weighted F1 under heavy class imbalance).


## Dataset

**NSL-KDD** — an improved version of the KDD Cup 1999 dataset. The training file (`KDDTrain+.txt`) contains 41 raw features plus an attack-type label and difficulty score (the latter is dropped). Categorical features are one-hot encoded; all features are standardized via `StandardScaler`.

Classes include: `normal`, `neptune`, `satan`, `ipsweep`, `portsweep`, `smurf`, `back`, `teardrop`, `warezclient`, `pod`, `guess_passwd`, and 12 others. Several classes (e.g. `perl`, `spy`, `multihop`) have near-zero support in the test split — treat their per-class metrics as unreliable.

## Pipeline

1. **Load & encode** — label-encode targets, one-hot encode categorical features
2. **Split** — 72% train / 8% val / 20% test (stratified)
3. **Scale** — `StandardScaler` fit on train only
4. **SMOTE** — capped oversampling applied to **training set only** (minority classes capped at 10% of majority count)
5. **Train** — all three models on the same processed data
6. **Evaluate** — classification report + confusion matrix per model; bar chart comparing Macro/Weighted F1 across all three

## Model Details

### C-LSTM (Hybrid CNN-LSTM)
Custom architecture defined in `Hybrid_CNN_LSTM`:
- **1D CNN** (`Conv1d` → `ReLU` → `MaxPool1d`) extracts local feature patterns
- **LSTM** captures sequential dependencies across the feature map
- **Fully connected head** outputs 23-class logits
- Loss: `CrossEntropyLoss` with inverse-frequency class weights
- Optimizer: Adam

### Random Forest
- `class_weight='balanced'`
- Trained on original scaled data (no SMOTE)

### Logistic Regression
- `class_weight='balanced'`, `max_iter=1000`
- Trained on original scaled data (no SMOTE)

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
3. Outputs: per-model classification reports, confusion matrix heatmaps, and a comparative F1 bar chart.
