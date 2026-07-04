# -*- coding: utf-8 -*-
"""SVM - Classificação de Violência contra a Mulher (corrigido)"""

import tensorflow as tf
print(tf.__version__)

import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)

from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

import cv2

BASE_DIR = Path(__file__).parent / "imagens"
origem = BASE_DIR / "originais"
classes = ['violencia', 'nao_violencia']
img_size = (150, 150)

# =========== CARREGAMENTO ===========
X, y = [], []
for label, cls in enumerate(classes):
    pasta = origem / cls
    for fname in sorted(os.listdir(pasta)):
        img = load_img(pasta / fname, target_size=img_size)
        X.append(img_to_array(img).flatten() / 255.0)
        y.append(label)

X = np.array(X)
y = np.array(y)

# =========== DIVISÃO 75/25 (9 e 3 de cada) ===========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"Treino: {len(X_train)} ({sum(y_train==0)}/{sum(y_train==1)})")
print(f"Teste:  {len(X_test)} ({sum(y_test==0)}/{sum(y_test==1)})")

# =========== FUNÇÃO DE SUAVIZAÇÃO ===========
def suavizar(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# =========== DATA AUGMENTATION ===========
datagen = ImageDataGenerator(
    preprocessing_function=suavizar,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def gerar_aumentadas(imagens_flat, labels, n_por_img=50):
    todas_aug = []
    todas_labels = []
    for img_flat, lbl in zip(imagens_flat, labels):
        img_3d = img_flat.reshape((150, 150, 3))
        x = img_3d.reshape((1,) + img_3d.shape)
        for batch in datagen.flow(x, batch_size=1, save_format='jpeg'):
            img_aug = batch[0].flatten()
            todas_aug.append(img_aug)
            todas_labels.append(lbl)
            if len(todas_aug) % n_por_img == 0:
                break
    return np.array(todas_aug), np.array(todas_labels)

print("Gerando aumentadas de treino...")
X_train_aug, y_train_aug = gerar_aumentadas(X_train, y_train)
print(f"Treino aumentado: {len(X_train_aug)} imagens")

print("Gerando aumentadas de teste (separadas)...")
X_test_aug, y_test_aug = gerar_aumentadas(X_test, y_test)
print(f"Teste aumentado: {len(X_test_aug)} imagens")

# =========== TREINAMENTO E MÉTRICAS ===========
metricas_path = BASE_DIR / "metricas_svm.txt"

with open(metricas_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("RELATÓRIO DE MÉTRICAS - SVM (kernel RBF)\n")
    f.write("=" * 60 + "\n\n")

    for i in range(5):
        print(f"Treinamento {i+1}/5 - SVM RBF...")
        svm_model = SVC(kernel='rbf', random_state=42 + i, cache_size=2000)
        svm_model.fit(X_train_aug, y_train_aug)
        y_pred = svm_model.predict(X_test_aug)

        acc = accuracy_score(y_test_aug, y_pred)
        prec = precision_score(y_test_aug, y_pred, average='macro')
        rec = recall_score(y_test_aug, y_pred, average='macro')
        f1 = f1_score(y_test_aug, y_pred, average='macro')
        cm = confusion_matrix(y_test_aug, y_pred)

        f.write(f"--- Execução {i+1} ---\n")
        f.write(f"Acurácia: {acc:.4f}\n")
        f.write(f"Precisão (macro): {prec:.4f}\n")
        f.write(f"Recall (macro): {rec:.4f}\n")
        f.write(f"F1-Score (macro): {f1:.4f}\n")
        f.write("Matriz de Confusão:\n")
        f.write(np.array2string(cm) + "\n\n")

    f.write("=" * 60 + "\n")
    medias = [accuracy_score(y_test_aug,
               SVC(kernel='rbf', random_state=42 + j, cache_size=2000)
               .fit(X_train_aug, y_train_aug).predict(X_test_aug)) for j in range(5)]
    f.write(f"Acurácia média: {np.mean(medias):.4f}\n")
    f.write("=" * 60 + "\n")

print(f"Métricas salvas em: {metricas_path}")

# =========== MATRIZ DE CONFUSÃO ===========
svm_final = SVC(kernel='rbf', random_state=42, cache_size=2000)
svm_final.fit(X_train_aug, y_train_aug)
y_pred_final = svm_final.predict(X_test_aug)

cm = confusion_matrix(y_test_aug, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - SVM')
plt.tight_layout()
plt.savefig(BASE_DIR / "matriz_confusao_svm.png")
plt.close()
print("Matriz de confusão salva.")

# =========== PREDIÇÃO ===========
def mostrar_predicao_svm(modelo, caminho_imagem):
    img = load_img(caminho_imagem, target_size=img_size)
    img_array = img_to_array(img).flatten() / 255.0
    img_array = img_array.reshape(1, -1)
    pred = modelo.predict(img_array)
    classe_predita = classes[int(pred[0])]
    plt.figure(figsize=(4, 4))
    plt.imshow(load_img(caminho_imagem))
    plt.axis('off')
    plt.text(75, 50, classe_predita.upper(), fontsize=16, ha='center',
             color='red' if classe_predita == 'violencia' else 'green')
    plt.title('SVM - Classificação')
    plt.savefig(BASE_DIR / "predicao_svm.png")
    plt.close()
    print(f'Classe predita: {classe_predita}')

mostrar_predicao_svm(svm_final, '/home/igor/Downloads/violencia.webp')
