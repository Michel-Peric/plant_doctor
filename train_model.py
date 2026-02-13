#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plant Doctor - Script d'entraînement du modèle (TensorFlow)
===========================================================

Ce script permet d'entraîner un modèle CNN personnalisé (ou EfficientNet)
sur le dataset PlantVillage pour la détection de maladies des plantes.

Fonctionnalités :
- Chargement automatique du dataset (train/validation)
- Augmentation de données (Data Augmentation)
- Entraînement en deux phases (apprentissage initial + fine-tuning)
- Sauvegarde automatique des meilleurs modèles et checkpoints
- Génération de graphiques de suivi (perte, précision)
- Support GPU automatique (avec fallback CPU)

Usage:
    python train_model.py

Sorties (dans le dossier `models/`) :
- `efficientnet_plant_disease.h5` : Modèle final format H5
- `efficientnet_plant_disease.keras` : Modèle final format Keras Natif
- `best_model_phase*.keras` : Meilleurs modèles par phase
- `training_history.png` : Courbes d'apprentissage
- `class_labels.json` : Mapping des classes (sauvegardé dans `data/`)

Auteur : Plant Doctor Team
"""

import os
import sys
import json
import time
from datetime import timedelta
from pathlib import Path

# Configuration de l'environnement TensorFlow
# Masquer les logs INFO et WARNING, garder les ERAOR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Désactiver DirectML (conflit de kernel avec TF 2.10 sur certaines configs Windows)
os.environ['TF_DIRECTML_DISABLE'] = '1'
# Ligne suivante à commenter si vous voulez utiliser le GPU sur une config compatible
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force l'utilisation du CPU par sécurité

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# Note : L'utilisation de ImageDataGenerator est préférée ici pour sa simplicité
# bien que dépréciée au profit de tf.data dans les versions récentes.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour générer des fichiers PNG sans fenêtre
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION ET CHEMINS
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Hyperparamètres
IMG_SIZE = 224      # Taille standard pour la plupart des CNN (ResNet, EfficientNet)
BATCH_SIZE = 32     # Taille du lot (réduire à 16 ou 8 si erreur OOM sur GPU)
EPOCHS_PHASE1 = 10  # Époques pour la phase d'apprentissage initial
EPOCHS_PHASE2 = 15  # Époques pour le fine-tuning
SEED = 42           # Grain pour la reproductibilité

# ============================================================
# COMPOSANTS UTILITAIRES
# ============================================================
class ProgressCallback(keras.callbacks.Callback):
    """Callback personnalisé pour afficher une barre de progression détaillée.
    
    Affiche :
    - La progression de l'époque
    - Les temps écoulés et estimés (ETA)
    - Les métriques (Loss, Accuracy)
    - Alertes de sur-apprentissage (Overfitting)
    """

    def __init__(self, total_epochs, phase_name):
        super().__init__()
        self.total_epochs = total_epochs
        self.phase_name = phase_name
        self.epoch_times = []
        # Historique manuel pour calculs internes
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_train_begin(self, logs=None):
        print(f"\n{'='*60}")
        print(f"  {self.phase_name} - {self.total_epochs} époques prévues")
        print(f"{'='*60}")
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\nÉpoque {epoch+1}/{self.total_epochs}", end=" ")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)

        # Stockage des métriques
        self.history['loss'].append(logs.get('loss', 0))
        self.history['accuracy'].append(logs.get('accuracy', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['val_accuracy'].append(logs.get('val_accuracy', 0))

        # Calcul du temps restant estimé (ETA)
        avg_time = np.mean(self.epoch_times)
        remaining = self.total_epochs - (epoch + 1)
        eta = timedelta(seconds=int(avg_time * remaining))

        # Barre de progression ASCII
        progress = (epoch + 1) / self.total_epochs
        bar_len = 20
        filled = int(bar_len * progress)
        bar = '#' * filled + '-' * (bar_len - filled)

        # Détection basique de sur-apprentissage (Overfitting)
        warning = ""
        if len(self.history['val_loss']) > 3:
            # Si la perte de validation diverge trop de la perte d'entraînement
            gap = np.mean(self.history['val_loss'][-3:]) - np.mean(self.history['loss'][-3:])
            if gap > 0.3:
                warning = " [!] ATTENTION: SUR-APPRENTISSAGE DÉTECTÉ"

        print(f"[{bar}] {progress*100:.0f}%")
        print(f"  Perte: {logs['loss']:.4f} / Val: {logs['val_loss']:.4f} | "
              f"Précision: {logs['accuracy']*100:.1f}% / Val: {logs['val_accuracy']*100:.1f}%")
        print(f"  Temps: {timedelta(seconds=int(epoch_time))} | Fin estimée dans: {eta}{warning}")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start
        print(f"\n{'='*60}")
        print(f"  {self.phase_name} terminé!")
        print(f"  Temps total: {timedelta(seconds=int(total_time))}")
        if self.history['val_accuracy']:
            print(f"  Meilleure précision (validation): {max(self.history['val_accuracy'])*100:.2f}%")
        print(f"{'='*60}")


def plot_training_history(history1, history2, save_path):
    """Génère et sauvegarde les courbes d'apprentissage combinées des deux phases.
    
    Args:
        history1: Historique de la phase 1.
        history2: Historique de la phase 2.
        save_path (Path): Chemin de sauvegarde de l'image.
    """
    # Extraction des dictionnaires d'historique qu'ils soient objets History ou dicts
    if hasattr(history1, 'history'):
        h1 = history1.history
    else:
        h1 = history1
    if hasattr(history2, 'history'):
        h2 = history2.history
    else:
        h2 = history2

    # Concaténation des deux phases
    acc = h1['accuracy'] + h2['accuracy']
    val_acc = h1['val_accuracy'] + h2['val_accuracy']
    loss = h1['loss'] + h2['loss']
    val_loss = h1['val_loss'] + h2['val_loss']

    epochs = range(1, len(acc) + 1)
    phase1_end = len(h1['accuracy'])

    # Création de la figure 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Graphique 1 : Fonction de perte (Loss)
    axes[0, 0].plot(epochs, loss, 'b-', label='Entraînement', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    axes[0, 0].axvline(x=phase1_end, color='green', linestyle='--', alpha=0.7, label='Fin Phase 1')
    axes[0, 0].set_title('Fonction de Perte (Loss)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Époque')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Graphique 2 : Précision (Accuracy)
    axes[0, 1].plot(epochs, [a*100 for a in acc], 'b-', label='Entraînement', linewidth=2)
    axes[0, 1].plot(epochs, [a*100 for a in val_acc], 'r-', label='Validation', linewidth=2)
    axes[0, 1].axvline(x=phase1_end, color='green', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Précision (Accuracy)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Époque')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Graphique 3 : Écart de généralisation (Overfitting Gap)
    gap = [t - v for t, v in zip(acc, val_acc)]
    axes[1, 0].plot(epochs, [g*100 for g in gap], 'purple', linewidth=2)
    axes[1, 0].axhline(y=5, color='orange', linestyle='--', alpha=0.7)
    axes[1, 0].axhline(y=10, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Écart Généralisation (Train - Val)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Époque')
    axes[1, 0].set_ylabel('Écart (%)')
    axes[1, 0].grid(True, alpha=0.3)

    # Graphique 4 : Meilleur modèle
    best_epoch = np.argmax(val_acc) + 1
    axes[1, 1].plot(epochs, [a*100 for a in val_acc], 'r-', linewidth=2)
    axes[1, 1].scatter([best_epoch], [val_acc[best_epoch-1]*100], color='gold', s=200,
                       marker='*', zorder=5, label=f'Best: {val_acc[best_epoch-1]*100:.1f}%')
    axes[1, 1].set_title('Meilleure Performance Validation', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nGraphiques sauvegardés : {save_path}")


def main():
    print("\n" + "="*60)
    print("  PLANT DOCTOR - ENTRAÎNEMENT DU MODÈLE")
    print("="*60)

    # Vérification TensorFlow et GPU
    print(f"\nVersion TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU détecté activé : {gpus}")
    else:
        print("Aucun GPU détecté ou désactivé - Entraînement sur CPU (sera plus lent)")

    # Vérification présence des données
    if not TRAIN_DIR.exists():
        print(f"\n[X] Dataset non trouvé : {TRAIN_DIR}")
        print("Veuillez télécharger le dataset PlantVillage et le placer dans data/dataset/")
        sys.exit(1)

    # Analyse des classes
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    num_classes = len(classes)
    print(f"\nClasses trouvées : {num_classes}")

    # Préparation des générateurs de données
    # ImageDataGenerator gère le chargement, le redimensionnement et l'augmentation à la volée
    print("\nChargement et préparation des données...")
    
    # Augmentation de données pour l'entraînement (robustesse)
    train_datagen = ImageDataGenerator(
        rotation_range=20,      # Rotation aléatoire
        width_shift_range=0.2,  # Décalage horizontal
        height_shift_range=0.2, # Décalage vertical
        shear_range=0.2,        # Cisaillement
        zoom_range=0.2,         # Zoom aléatoire
        horizontal_flip=True,   # Retournement horizontal
        fill_mode='nearest'     # Remplissage des pixels créés
    )

    # Pas d'augmentation pour la validation, juste le redimensionnement implicite
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED
    )

    val_generator = val_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=SEED
    )

    print(f"Images d'entraînement : {train_generator.samples}")
    print(f"Images de validation  : {val_generator.samples}")

    # Sauvegarde des labels de classe pour l'application
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    labels_path = PROJECT_ROOT / "data" / "class_labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(class_labels, f, indent=2)
    print(f"Labels sauvegardés dans : {labels_path}")

    # Construction du modèle CNN
    # Nous utilisons une architecture personnalisée ici.
    # Note : Pour de meilleures performances, envisagez un Transfer Learning (MobileNet, EfficientNet)
    # mais assurez-vous de la compatibilité technique (cf. DirectML sur Windows).
    print("\nConstruction du modèle CNN personnalisé...")

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Normalisation des pixels [0, 255] -> [0, 1]
        layers.Rescaling(1./255),

        # Bloc Convolutionnel 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Régularisation

        # Bloc Convolutionnel 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc Convolutionnel 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc Convolutionnel 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc Convolutionnel 5
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(), # Réduit la dimension spatiale
        layers.Dropout(0.5),

        # Classificateur final (Dense)
        layers.Dense(512),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax') # Sortie probabilités
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Paramètres totaux du modèle : {model.count_params():,}")

    # Callbacks pour la Phase 1
    progress1 = ProgressCallback(EPOCHS_PHASE1, "Phase 1 : Extraction de caractéristiques")
    callbacks1 = [
        progress1,
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'checkpoint_p1_epoch{epoch:02d}.keras'),
            save_best_only=False, verbose=0
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'best_model_phase1.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1
        )
    ]

    # Phase 1 : Entraînement initial
    print(f"\n>>> Démarrage Phase 1 ({EPOCHS_PHASE1} époques)...")
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_PHASE1,
        validation_data=val_generator,
        callbacks=callbacks1,
        verbose=0 # Géré par ProgressCallback
    )

    # Phase 2 : Fine-tuning avec learning rate réduit
    print("\n\nPréparation Phase 2 : Entraînement fin...")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5), # LR plus faible
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    progress2 = ProgressCallback(EPOCHS_PHASE2, "Phase 2 : Affinage (Fine-Tuning)")
    callbacks2 = [
        progress2,
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'checkpoint_p2_epoch{epoch:02d}.keras'),
            save_best_only=False, verbose=0
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'best_model_phase2.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1
        )
    ]

    print(f"\n>>> Démarrage Phase 2 ({EPOCHS_PHASE2} époques)...")
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS_PHASE2,
        validation_data=val_generator,
        callbacks=callbacks2,
        verbose=0
    )

    # Évaluation finale
    print("\n\n>>> Évaluation finale du modèle...")
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Perte finale (Loss) : {loss:.4f}")
    print(f"Précision finale (Accuracy) : {accuracy*100:.2f}%")

    # Sauvegarde des modèles finaux
    final_path = MODEL_DIR / 'efficientnet_plant_disease.h5'
    model.save(final_path)
    print(f"\n[OK] Modèle final sauvegardé : {final_path}")

    # Sauvegarde au format natif Keras (recommandé pour le futur)
    keras_path = MODEL_DIR / 'efficientnet_plant_disease.keras'
    model.save(keras_path)
    print(f"[OK] Modèle format Keras sauvegardé : {keras_path}")

    # Génération des graphiques
    plot_training_history(
        progress1.history,
        progress2.history,
        MODEL_DIR / 'training_history.png'
    )

    # Rapport de fin
    print("\n" + "="*60)
    print("  ENTRAÎNEMENT TERMINÉ !")
    print("="*60)
    print(f"\n  Précision finale : {accuracy*100:.2f}%")
    print(f"  Modèle sauvegardé : {final_path}")
    print(f"  Graphiques générés : {MODEL_DIR / 'training_history.png'}")
    print("\n  Vous pouvez maintenant lancer l'application Plant Doctor !")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
