# Classic Methods Violence Predict

Classificação binária de violência contra a mulher em imagens usando **Random Forest** e **SVM** (kernel RBF).

## Pipeline

1. **Divisão** — 9 imagens/classe para treino, restante para validação
2. **Pré-processamento** — Gaussian blur `(5,5)` via OpenCV
3. **Data augmentation** — 50 versões por imagem original (rotação `30°`, zoom `20%`, shift `10%`, flip horizontal)
4. **Carregamento** — Imagens redimensionadas para `150×150`, achatadas e normalizadas (`/255.0`)
5. **Split** — 80% treino / 20% teste
6. **Treino e avaliação** — 5 execuções com métricas: acurácia, precisão, recall, F1 e matriz de confusão

## Estrutura

```
classic-methods-violence-predict/
├── Classic Methods/
│   ├── projeto_rf/
│   │   └── rf_violencia.py   # Random Forest com tuning de n_estimators (1-100)
│   └── projeto_svm/
│       └── svm_violencia.py   # SVM com kernel RBF
└── requirements.txt
```

## Dados

Organizar imagens em `Classic Methods/projeto_rf/imagens/originais/` e `Classic Methods/projeto_svm/imagens/originais/` com subpastas:

```
imagens/originais/
├── violencia/
└── nao_violencia/
```

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

Cada script gera:
- `metricas_rf.txt` / `metricas_svm.txt` — relatório das 5 execuções
- `matriz_confusao_random_forest.png` / `matriz_confusao_svm.png`
- `grafico_n_estimators.png` e `grafico_n_estimators_media.png` (apenas RF)
- `predicao_rf.png` / `predicao_svm.png` — predição em imagem de exemplo
