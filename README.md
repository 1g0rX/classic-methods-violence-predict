# Classic Methods Violence Predict

Detecção de violência em vídeos utilizando métodos clássicos de Machine Learning: **Random Forest** e **SVM**.

## Estrutura

```
classic-methods-violence-predict/
├── Classic Methods/
│   ├── projeto_rf/
│   │   └── rf_violencia.py
│   └── projeto_svm/
│       └── svm_violencia.py
└── requirements.txt
```

## Modelos

- **Random Forest** — `Classic Methods/projeto_rf/rf_violencia.py`
- **SVM** — `Classic Methods/projeto_svm/svm_violencia.py`

## Requisitos

```
tensorflow>=2.12.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
Pillow>=9.4.0
```

## Como usar

```bash
pip install -r requirements.txt
python "Classic Methods/projeto_rf/rf_violencia.py"
python "Classic Methods/projeto_svm/svm_violencia.py"
```
