# Classic Methods Violence Predict

Binary classification of violence against women in images using **Random Forest** and **SVM (RBF kernel)**.

Part of the Computer Vision seminar at **CEFET-MG — Campus V (Divinópolis)**, supervised by Prof. Thabatta Moreira.

---

## Pipeline

1. **Collection** — 24 images (12 violence, 12 non-violence), copyright-free, distinct contexts
2. **Anonymization** — Gaussian blur 5×5 via OpenCV
3. **Preprocessing** — Resize to 150×150, flatten, normalize (÷255)
4. **Data augmentation** — 50 variations per original (rotation 30°, zoom 20%, shift 10%, horizontal flip)
5. **Model training** — RF n_estimators tuning (1–100, step 5, 5 reps each) + SVM RBF
6. **Evaluation** — 5 runs per model with accuracy, precision, recall, F1 (macro avg) and confusion matrix

---

## Key Results

### ⚠ Data Leakage Discovery

Early runs split the test set **after** augmentation, causing data leakage (test images derived from the same originals as training). After correcting the pipeline (75/25 split **before** augmentation), results changed significantly:

| Scenario | RF | SVM |
|---|---|---|
| With leakage | 96.09% | 99.43% |
| **Corrected** | **51.13%** | **75.33%** |

### Confusion Matrices (Corrected Pipeline)

**Random Forest** (n_estimators=41, 300 test samples):

| | Pred: Non-violence | Pred: Violence |
|---|---|---|
| **Actual: Non-violence** | 110 | 40 |
| **Actual: Violence** | 105 | 45 |

**SVM RBF** (300 test samples):

| | Pred: Non-violence | Pred: Violence |
|---|---|---|
| **Actual: Non-violence** | 131 | 19 |
| **Actual: Violence** | 55 | 95 |

The corrected SVM reaches 75.33% accuracy but with **55 false negatives** — a critical issue for violence detection.

---

## Project Structure

```
classic-methods-violence-predict/
├── Classic Methods/
│   ├── projeto_rf/
│   │   └── rf_violencia.py       # Random Forest with n_estimators tuning
│   ├── projeto_svm/
│   │   └── svm_violencia.py       # SVM with RBF kernel
│   ├── main_classica_RF.ipynb    # RF notebook
│   └── main_classica_SVM.ipynb   # SVM notebook
├── .gitignore
├── README.md
└── requirements.txt
```

Each script generates output inside its `imagens/` subfolder:
- `metricas_*.txt` — per-run metrics
- `matriz_confusao_*.png` — confusion matrix plot
- `grafico_n_estimators*.png` — n_estimators vs accuracy (RF only)
- `predicao_*.png` — prediction on `violencia.webp`

---

## Dataset Setup

Place images in `Classic Methods/projeto_*/imagens/originais/`:

```
imagens/originais/
├── violencia/          # 12 images
└── nao_violencia/      # 12 images
```

Images must be JPEG, any size (resized automatically). Anonymization with Gaussian blur is applied during preprocessing.

---

## Requirements

```
tensorflow>=2.12.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
Pillow>=9.4.0
```

---

## How to Run

```bash
pip install -r requirements.txt
python "Classic Methods/projeto_rf/rf_violencia.py"
python "Classic Methods/projeto_svm/svm_violencia.py"
```

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Cortes, C. & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Moreira, T. (2026). Lecture notebooks — Computer Vision. CEFET-MG.

---

*Author: Igor Moreira Lopes — CEFET-MG Campus V (Divinópolis)*
