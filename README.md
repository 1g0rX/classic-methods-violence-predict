# Classic Methods Violence Predict

Classificação binária de violência contra a mulher em imagens usando **Random Forest** e **SVM (kernel RBF)**.

Trabalho da disciplina Visão Computacional — **CEFET-MG Campus V (Divinópolis)**, sob orientação da Prof.ª Thabatta Moreira.

---

## 📋 Pipeline

1. **Coleta** — 24 imagens (12 violência, 12 não violência), livres de direitos autorais, contextos distintos
2. **Anonimização** — Filtro Gaussiano 5×5 via OpenCV
3. **Pré-processamento** — Redimensionar para 150×150, achatar (flatten), normalizar (÷255)
4. **Data augmentation** — 50 variações por original (rotação 30°, zoom 20%, deslocamento 10%, flip horizontal)
5. **Treinamento** — RF com tuning de n_estimators (1–100, passo 5, 5 repetições) + SVM RBF (5 execuções)
6. **Avaliação** — Acurácia, precisão, recall, F1 (média macro) e matriz de confusão

---

## 📦 Estrutura do Projeto

```
classic-methods-violence-predict/
├── Classic Methods/
│   ├── projeto_rf/
│   │   └── rf_violencia.py          # Random Forest com tuning
│   ├── projeto_svm/
│   │   └── svm_violencia.py          # SVM com kernel RBF
│   ├── main_classica_RF.ipynb       # Notebook do RF
│   └── main_classica_SVM.ipynb      # Notebook do SVM
├── .gitignore
├── README.md
└── requirements.txt
```

Cada script gera dentro de `imagens/`:
- `metricas_*.txt` — relatório das execuções
- `matriz_confusao_*.png` — matriz de confusão
- `grafico_n_estimators*.png` — número de árvores × acurácia (apenas RF)
- `predicao_*.png` — predição em `violencia.webp`

---

## 🖼️ Como configurar as imagens

O dataset **não está incluído** neste repositório. Você deve criar as pastas e adicionar manualmente as imagens:

```
Classic Methods/projeto_rf/imagens/originais/
├── violencia/          # 12 imagens com indícios de agressão
└── nao_violencia/      # 12 imagens de casais sem agressão
```

```
Classic Methods/projeto_svm/imagens/originais/
├── violencia/          # 12 imagens com indícios de agressão
└── nao_violencia/      # 12 imagens de casais sem agressão
```

**Requisitos das imagens:**
- Formato JPEG
- Livres de direitos autorais
- Contextos distintos entre si
- Serão anonimizadas automaticamente pelo filtro Gaussiano durante o pré-processamento

> ⚠️ As pastas `aumentadas/`, `treino/` e `validacao/` são geradas automaticamente na primeira execução.

---

## 🔬 Resultados

### ⚠ Descoberta: Vazamento de Dados (Data Leakage)

As primeiras execuções separavam o conjunto de teste **após** o data augmentation, o que causava vazamento: as imagens de teste eram variações das mesmas originais usadas no treino. Após corrigir o pipeline (divisão 75/25 **antes** do augmentation), os resultados mudaram drasticamente:

| Cenário | RF | SVM |
|---|---|---|
| Com vazamento | 96,09% | 99,43% |
| **Corrigido** | **51,13%** | **75,33%** |

### Matrizes de Confusão (Pipeline Corrigido)

**Random Forest** (n_estimators=41, 300 amostras de teste):

| | Pred: Não Violência | Pred: Violência |
|---|---|---|
| **Real: Não Violência** | 110 | 40 |
| **Real: Violência** | 105 | 45 |

**SVM RBF** (300 amostras de teste):

| | Pred: Não Violência | Pred: Violência |
|---|---|---|
| **Real: Não Violência** | 131 | 19 |
| **Real: Violência** | 55 | 95 |

O SVM corrigido atingiu 75,33% de acurácia, porém com **55 falsos negativos** — um número crítico para detecção de violência, pois significa deixar de identificar uma agressão.

---

## 📦 Requisitos

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

## 🚀 Como executar

```bash
pip install -r requirements.txt
python "Classic Methods/projeto_rf/rf_violencia.py"
python "Classic Methods/projeto_svm/svm_violencia.py"
```

---

## 📚 Referências

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Cortes, C. & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Moreira, T. (2026). Cadernos de aula — Visão Computacional. CEFET-MG.

---

*Autor: Igor Moreira Lopes — CEFET-MG Campus V (Divinópolis)*
