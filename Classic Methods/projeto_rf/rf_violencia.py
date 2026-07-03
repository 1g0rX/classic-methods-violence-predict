# -*- coding: utf-8 -*-
"""Random Forest - Classificação de Violência contra a Mulher"""

import tensorflow as tf
print(tf.__version__)

import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import cv2

# =========== DIRETÓRIOS ===========

BASE_DIR = Path(__file__).parent / "imagens"

origem = BASE_DIR / "originais"
treino = BASE_DIR / "treino"
validacao = BASE_DIR / "validacao"
aug_treino = BASE_DIR / "aumentadas"

# =========== CLASSES ===========

classes = ['violencia', 'nao_violencia']

# =========== DIVISÃO TREINO/VALIDAÇÃO ===========

for classe in classes:
    pasta_origem = origem / classe
    arquivos = sorted(os.listdir(pasta_origem))
    treino_classe = treino / classe
    validacao_classe = validacao / classe
    treino_classe.mkdir(parents=True, exist_ok=True)
    validacao_classe.mkdir(parents=True, exist_ok=True)
    for i, arquivo in enumerate(arquivos):
        src = pasta_origem / arquivo
        if i < 9:
            shutil.copy(src, treino_classe / arquivo)
        else:
            shutil.copy(src, validacao_classe / arquivo)

print("Divisão treino/validação concluída.")

# =========== FUNÇÃO DE SUAVIZAÇÃO ===========

def suavizar(image):
    suavizada = cv2.GaussianBlur(image, (5, 5), 0)
    return suavizada

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

if aug_treino.exists():
    shutil.rmtree(aug_treino)
aug_treino.mkdir(parents=True, exist_ok=True)

for classe in classes:
    input_path = treino / classe
    output_path = aug_treino / classe
    output_path.mkdir(parents=True, exist_ok=True)
    for img_file in os.listdir(input_path):
        img_path = input_path / img_file
        img = load_img(img_path, target_size=(150, 150))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_path,
                                  save_prefix=classe, save_format='jpeg'):
            i += 1
            if i >= 50:
                break

print("Data augmentation concluído (50 imagens aumentadas por imagem original).")

# =========== CARREGAMENTO DOS DADOS ===========

img_size = (150, 150)

X_train, y_train = [], []
for label, classe in enumerate(classes):
    pasta = aug_treino / classe
    for arquivo in os.listdir(pasta):
        caminho_img = pasta / arquivo
        try:
            img = load_img(caminho_img, target_size=img_size)
            img_array = img_to_array(img)
            img_array = img_array.flatten() / 255.0
            X_train.append(img_array)
            y_train.append(label)
        except:
            print(f"Erro ao carregar {caminho_img}")

print(f"Total de imagens aumentadas carregadas: {len(X_train)}")

# =========== SEPARAÇÃO 20% PARA TESTE ===========

indices = np.arange(len(X_train))
np.random.shuffle(indices)
n_teste = int(len(X_train) * 0.2)
teste_idx = indices[:n_teste]
treino_idx = indices[n_teste:]

X_train = np.array(X_train)
y_train = np.array(y_train)
X_treino, X_teste = X_train[treino_idx], X_train[teste_idx]
y_treino, y_teste = y_train[treino_idx], y_train[teste_idx]

print(f"Conjunto: {len(X_treino)} treino / {len(X_teste)} teste (20% das aumentadas).")

# =========== AJUSTE DO N_ESTIMATORS ===========

n_arvores = range(1, 100, 5)
train_acc = []
val_acc = []

for n in n_arvores:
    train_acc_rep = []
    val_acc_rep = []
    for _ in range(5):
        clf = RandomForestClassifier(n_estimators=n, random_state=42)
        clf.fit(X_treino, y_treino)
        train_acc_rep.append(clf.score(X_treino, y_treino))
        val_acc_rep.append(accuracy_score(y_teste, clf.predict(X_teste)))
    train_acc.append(np.mean(train_acc_rep))
    val_acc.append(np.mean(val_acc_rep))
    print(f"n_estimators={n:3d} | Acurácia treino: {train_acc[-1]:.4f} | Acurácia val: {val_acc[-1]:.4f}")

melhor_n = n_arvores[np.argmax(val_acc)]
print(f"\nMelhor n_estimators: {melhor_n} (acurácia de validação: {max(val_acc):.4f})")

plt.plot(n_arvores, train_acc, label='Acurácia Treinamento')
plt.plot(n_arvores, val_acc, label='Acurácia Validação')
plt.xlabel('Número de Árvores')
plt.ylabel('Acurácia')
plt.title('Desempenho do Random Forest por número de árvores')
plt.legend()
plt.grid(True)
plt.savefig(BASE_DIR / "grafico_n_estimators.png")
plt.close()

plt.plot(n_arvores, val_acc, 'g-s', linewidth=2)
plt.axvline(x=melhor_n, color='red', linestyle='--', label=f'Melhor n = {melhor_n}')
plt.xlabel('Número de Árvores')
plt.ylabel('Acurácia Média de Validação')
plt.title('Número de Árvores × Acurácia Média (5 execuções)')
plt.legend()
plt.grid(True)
plt.savefig(BASE_DIR / "grafico_n_estimators_media.png")
plt.close()

print("Gráficos salvos.")

# =========== TREINAMENTO E VALIDAÇÃO DO RF ===========

metricas_path = BASE_DIR / "metricas_rf.txt"

with open(metricas_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("RELATÓRIO DE MÉTRICAS - RANDOM FOREST\n")
    f.write(f"Melhor n_estimators: {melhor_n}\n")
    f.write("=" * 60 + "\n\n")

    for i in range(5):
        print(f"Treinamento {i+1}/5 com n_estimators={melhor_n}...")
        clf = RandomForestClassifier(n_estimators=melhor_n, random_state=42 + i)
        clf.fit(X_treino, y_treino)
        y_pred = clf.predict(X_teste)

        acc = accuracy_score(y_teste, y_pred)
        prec = precision_score(y_teste, y_pred, average='macro')
        rec = recall_score(y_teste, y_pred, average='macro')
        f1 = f1_score(y_teste, y_pred, average='macro')
        cm = confusion_matrix(y_teste, y_pred)

        f.write(f"--- Execução {i+1} ---\n")
        f.write(f"Acurácia: {acc:.4f}\n")
        f.write(f"Precisão (macro): {prec:.4f}\n")
        f.write(f"Recall (macro): {rec:.4f}\n")
        f.write(f"F1-Score (macro): {f1:.4f}\n")
        f.write("Matriz de Confusão:\n")
        f.write(np.array2string(cm) + "\n\n")

    f.write("=" * 60 + "\n")
    f.write(f"Acurácia média: {np.mean([accuracy_score(y_teste, RandomForestClassifier(n_estimators=melhor_n, random_state=42 + j).fit(X_treino, y_treino).predict(X_teste)) for j in range(5)]):.4f}\n")
    f.write("=" * 60 + "\n")

print(f"Métricas salvas em: {metricas_path}")

# =========== MATRIZ DE CONFUSÃO ===========

clf_final = RandomForestClassifier(n_estimators=melhor_n, random_state=42)
clf_final.fit(X_treino, y_treino)
y_pred_final = clf_final.predict(X_teste)

cm = confusion_matrix(y_teste, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - Random Forest')
plt.tight_layout()
plt.savefig(BASE_DIR / "matriz_confusao_random_forest.png")
plt.close()

print("Matriz de confusão salva.")

# =========== PREDIÇÃO ===========

def mostrar_predicao_rf(modelo, caminho_imagem):
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
    plt.title('Random Forest - Classificação')
    plt.savefig(BASE_DIR / "predicao_rf.png")
    plt.close()
    print(f'Classe predita: {classe_predita}')

mostrar_predicao_rf(clf_final, '/home/igor/Downloads/violencia.webp')

