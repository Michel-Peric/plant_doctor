#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plant Doctor - Script d'entraînement PyTorch avec support DirectML (GPU AMD/Intel/NVIDIA)
========================================================================================

Ce script permet d'entraîner un modèle ResNet50 sur le dataset PlantVillage en utilisant
PyTorch. Il est spécifiquement optimisé pour Windows avec le support de `torch-directml`
permettant l'accélération matérielle sur une large gamme de GPU (AMD Radeon, Intel, etc.),
là où CUDA est limité aux GPU NVIDIA.

Fonctionnalités :
- Détection automatique du périphérique (DirectML > CPU).
- Augmentation de données à la volée.
- Fine-tuning d'un ResNet50 pré-entraîné sur ImageNet.
- Suivi en temps réel de l'entraînement avec graphiques interactifs.
- Sauvegarde au format PyTorch (.pth) et export ONNX (.onnx).

Usage:
    python train_model_pytorch.py              # Entraînement complet (Phase 1 + Phase 2)
    python train_model_pytorch.py --resume     # Reprendre à la Phase 2 (charge best_model_phase1.pth)

Auteur : Plant Doctor Team
"""

import os
import sys
import json
import time
import argparse
from datetime import timedelta
from pathlib import Path

# Force unbuffered output pour que les logs s'affichent en temps réel dans la console
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# torch_directml permet l'accès à l'API DirectML de Windows pour l'accélération GPU
# Utile pour les cartes graphiques non-NVIDIA (AMD RX, Intel Arc, iGPU...)
import torch_directml
import matplotlib
# Utilisation du backend TkAgg pour afficher les graphiques en temps réel dans une fenêtre
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()  # Active le mode interactif de Matplotlib

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
IMG_SIZE = 224          # Taille d'entrée pour ResNet50
BATCH_SIZE = 32         # Taille du lot standard
BATCH_SIZE_PHASE2 = 8   # Taille réduite pour le fine-tuning (économise la VRAM)
EPOCHS_PHASE1 = 10      # Époques pour l'entraînement du classifieur seul
EPOCHS_PHASE2 = 15      # Époques pour le fine-tuning complet
LEARNING_RATE = 0.001   # Taux d'apprentissage initial
SEED = 42               # Grain pour la reproductibilité

# Fixer les grains aléatoires pour avoir des résultats reproductibles
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# CHARGEMENT DES DONNÉES (DATA LOADING)
# ============================================================
def get_data_loaders(batch_size=BATCH_SIZE):
    """Crée les DataLoaders avec les transformations d'images.

    Applique des augmentations de données (flip, rotation, affine, jitter) 
    sur le jeu d'entraînement pour améliorer la robustesse du modèle.

    Args:
        batch_size (int): Taille du lot d'images.

    Returns:
        tuple: (train_loader, val_loader, classe_names, class_to_idx, train_dataset, val_dataset)
    """

    # Transformations pour l'entraînement (Augmentation de données)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),        # Miroir horizontal aléatoire
        transforms.RandomRotation(20),            # Rotation légère
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)), # Translation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),     # Variation luminosité/contraste
        transforms.ToTensor(),                    # Conversion en Tenseur PyTorch
        # Normalisation avec les moyennes/écarts-types standards ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Transformations pour la validation (Juste redimensionnement et normalisation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Chargement des datasets depuis les dossiers
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VALID_DIR, transform=val_transform)

    # Création des DataLoaders
    # num_workers=0 est recommandé sur Windows pour éviter certains problèmes de multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Mélanger les données à chaque époque
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # Pas besoin de mélanger la validation
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader, train_dataset.classes, train_dataset.class_to_idx, train_dataset, val_dataset


# ============================================================
# MODÈLE (MODEL)
# ============================================================
def create_model(num_classes, device):
    """Crée un modèle basé sur ResNet50 avec un classifieur personnalisé.

    Utilise le Transfer Learning :
    1. Charge ResNet50 pré-entraîné sur ImageNet.
    2. Gèle tous les paramètres (poids) du réseau.
    3. Remplace la couche finale (fc) par un nouveau bloc adapté à notre nombre de classes.

    Args:
        num_classes (int): Nombre de catégories de maladies.
        device (torch.device): Périphérique d'exécution (CPU ou DirectML).

    Returns:
        nn.Module: Le modèle configuré et déplacé sur le périphérique cible.
    """

    # Chargement des poids pré-entraînés (V1 est la version standard)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Geler (Freeze) tous les paramètres pour ne pas détruire les features apprises
    for param in model.parameters():
        param.requires_grad = False

    # Remplacement du classifieur final (Fully Connected layer)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),                # Dropout pour réduire l'overfitting
        nn.Linear(num_features, 512),   # Couche dense intermédiaire
        nn.ReLU(),                      # Activation non-linéaire
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)     # Couche de sortie
    )

    return model.to(device)


# ============================================================
# ENTRAÎNEMENT (TRAINING LOOPS)
# ============================================================
class TrainingHistory:
    """Classe utilitaire pour suivre les métriques d'entraînement."""

    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.epoch_times = []
        self.best_val_acc = 0
        self.best_epoch = 0


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Exécute une époque d'entraînement complète.

    Args:
        model: Modèle PyTorch.
        loader: DataLoader d'entraînement.
        criterion: Fonction de perte (Loss function).
        optimizer: Optimiseur.
        device: Périphérique cible.
        epoch: Numéro de l'époque actuelle.
        total_epochs: Nombre total d'époques.

    Returns:
        tuple: (loss_moyen, accuracy_moyenne) sur l'époque.
    """
    model.train()  # Mettre le modèle en mode entraînement (active Dropout, BatchNorm...)
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(loader)
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Déplacer les données sur le GPU/CPU
        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Remise à zéro des gradients
        optimizer.zero_grad()

        # 2. Forward pass (Prédiction)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 3. Backward pass (Calcul des gradients)
        loss.backward()

        # 4. Mise à jour des poids
        optimizer.step()

        # Suivi des statistiques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Affichage de la progression tous les 20 batches
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            progress = (batch_idx + 1) / num_batches
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '#' * filled + '-' * (bar_len - filled)

            elapsed = time.time() - start_time
            # Estimation du temps restant (ETA)
            eta = elapsed / progress - elapsed if progress > 0 else 0

            print(f"  Époque {epoch}/{total_epochs} [{bar}] {progress*100:.0f}% | "
                  f"Loss: {running_loss/(batch_idx+1):.4f} | "
                  f"Acc: {100.*correct/total:.1f}% | "
                  f"ETA: {timedelta(seconds=int(eta))}", flush=True)

    print(flush=True)
    return running_loss / num_batches, correct / total


def validate(model, loader, criterion, device):
    """Évalue le modèle sur le jeu de validation.

    Args:
        model: Modèle à évaluer.
        loader: DataLoader de validation.
        criterion: Fonction de perte.
        device: Périphérique cible.

    Returns:
        tuple: (loss_moyen, accuracy_moyenne).
    """
    model.eval()  # Mode évaluation (désactive Dropout, fige BatchNorm)
    running_loss = 0.0
    correct = 0
    total = 0

    # Pas de calcul de gradient nécessaire pour la validation (économise mémoire/temps)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), correct / total


class LivePlotter:
    """Visualisation en temps réel de l'entraînement avec Matplotlib.
    
    Affiche une fenêtre avec 4 graphiques mis à jour à chaque époque :
    - Loss (Train vs Valid)
    - Accuracy (Train vs Valid)
    - Overfitting Gap (Différence Train/Valid)
    - Best Validation Accuracy
    """

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        # Création de la figure 2x2
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Plant Doctor - Entraînement en cours...', fontsize=16, fontweight='bold')

        # Initialisation des listes de données
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        # Configuration initiale des axes
        self.setup_axes()
        plt.tight_layout()
        plt.show(block=False)  # Affichage non bloquant
        plt.pause(0.1)         # Petit délai pour laisser l'interface s'afficher

    def setup_axes(self):
        """Configure les titres, labels et grilles des graphiques."""
        # Graphique 1 : Loss
        self.axes[0, 0].set_title('Fonction de Perte (Loss)', fontsize=14, fontweight='bold')
        self.axes[0, 0].set_xlabel('Époque')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)

        # Graphique 2 : Accuracy
        self.axes[0, 1].set_title('Précision (Accuracy)', fontsize=14, fontweight='bold')
        self.axes[0, 1].set_xlabel('Époque')
        self.axes[0, 1].set_ylabel('%')
        self.axes[0, 1].grid(True, alpha=0.3)

        # Graphique 3 : Overfitting Gap
        self.axes[1, 0].set_title('Écart Généralisation (Train - Val)', fontsize=14, fontweight='bold')
        self.axes[1, 0].set_xlabel('Époque')
        self.axes[1, 0].set_ylabel('Écart (%)')
        self.axes[1, 0].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Attention (5%)')
        self.axes[1, 0].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Critique (10%)')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)

        # Graphique 4 : Validation Accuracy
        self.axes[1, 1].set_title('Meilleure Performance Validation', fontsize=14, fontweight='bold')
        self.axes[1, 1].set_xlabel('Époque')
        self.axes[1, 1].set_ylabel('%')
        self.axes[1, 1].grid(True, alpha=0.3)

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, best_epoch, best_val_acc, phase2_start=None):
        """Met à jour les graphiques avec les données de la nouvelle époque."""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc * 100)
        self.val_acc.append(val_acc * 100)

        # Effacer et redessiner (plus simple que update artists pour Matplotlib standard)
        for ax in self.axes.flat:
            ax.clear()
        self.setup_axes()

        # Trace : Loss
        self.axes[0, 0].plot(self.epochs, self.train_loss, 'b-o', label='Entraînement', linewidth=2, markersize=4)
        self.axes[0, 0].plot(self.epochs, self.val_loss, 'r-o', label='Validation', linewidth=2, markersize=4)
        if phase2_start:
            self.axes[0, 0].axvline(x=phase2_start, color='green', linestyle='--', alpha=0.7, label='Phase 2')
        self.axes[0, 0].legend()
        self.axes[0, 0].set_title('Fonction de Perte (Loss)', fontsize=14, fontweight='bold')
        self.axes[0, 0].set_xlabel('Époque')
        self.axes[0, 0].grid(True, alpha=0.3)

        # Trace : Accuracy
        self.axes[0, 1].plot(self.epochs, self.train_acc, 'b-o', label='Entraînement', linewidth=2, markersize=4)
        self.axes[0, 1].plot(self.epochs, self.val_acc, 'r-o', label='Validation', linewidth=2, markersize=4)
        if phase2_start:
            self.axes[0, 1].axvline(x=phase2_start, color='green', linestyle='--', alpha=0.7)
        self.axes[0, 1].legend()
        self.axes[0, 1].set_title('Précision (Accuracy)', fontsize=14, fontweight='bold')
        self.axes[0, 1].set_xlabel('Époque')
        self.axes[0, 1].set_ylabel('%')
        self.axes[0, 1].grid(True, alpha=0.3)

        # Trace : Overfitting Gap
        gap = [t - v for t, v in zip(self.train_acc, self.val_acc)]
        self.axes[1, 0].plot(self.epochs, gap, 'purple', linewidth=2, marker='o', markersize=4)
        self.axes[1, 0].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Attention (5%)')
        self.axes[1, 0].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Critique (10%)')
        self.axes[1, 0].legend()
        self.axes[1, 0].set_title('Écart Généralisation (Train - Val)', fontsize=14, fontweight='bold')
        self.axes[1, 0].set_xlabel('Époque')
        self.axes[1, 0].set_ylabel('Écart (%)')
        self.axes[1, 0].grid(True, alpha=0.3)

        # Trace : Best Validation Accuracy
        self.axes[1, 1].plot(self.epochs, self.val_acc, 'r-o', linewidth=2, markersize=4)
        if best_epoch > 0 and best_epoch <= len(self.val_acc):
            self.axes[1, 1].scatter([best_epoch], [best_val_acc * 100],
                                   color='gold', s=200, marker='*', zorder=5,
                                   label=f'Best: {best_val_acc*100:.1f}%')
            self.axes[1, 1].legend()
        self.axes[1, 1].set_title('Meilleure Performance Validation', fontsize=14, fontweight='bold')
        self.axes[1, 1].set_xlabel('Époque')
        self.axes[1, 1].set_ylabel('%')
        self.axes[1, 1].grid(True, alpha=0.3)

        # Mise à jour du titre global
        self.fig.suptitle(f'Plant Doctor - Époque {epoch}/{self.total_epochs} | Val Acc: {val_acc*100:.1f}%',
                         fontsize=16, fontweight='bold')

        # Rafraîchissement de l'interface
        plt.tight_layout()
        plt.pause(0.1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, save_path):
        """Sauvegarde l'image finale des graphiques."""
        self.fig.suptitle('Plant Doctor - Entraînement terminé !', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=150)
        print(f"\nGraphiques sauvegardés : {save_path}")


def plot_history(history, save_path):
    """Génère les graphiques finaux (version statique) post-entraînement."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history.train_loss) + 1)

    # Loss
    axes[0, 0].plot(epochs, history.train_loss, 'b-', label='Entraînement', linewidth=2)
    axes[0, 0].plot(epochs, history.val_loss, 'r-', label='Validation', linewidth=2)
    axes[0, 0].axvline(x=EPOCHS_PHASE1, color='green', linestyle='--', alpha=0.7, label='Début Phase 2')
    axes[0, 0].set_title('Fonction de Perte (Loss)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Époque')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, [a*100 for a in history.train_acc], 'b-', label='Entraînement', linewidth=2)
    axes[0, 1].plot(epochs, [a*100 for a in history.val_acc], 'r-', label='Validation', linewidth=2)
    axes[0, 1].axvline(x=EPOCHS_PHASE1, color='green', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Précision (Accuracy)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Époque')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Overfitting gap
    gap = [t - v for t, v in zip(history.train_acc, history.val_acc)]
    axes[1, 0].plot(epochs, [g*100 for g in gap], 'purple', linewidth=2)
    axes[1, 0].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Attention (5%)')
    axes[1, 0].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Critique (10%)')
    axes[1, 0].set_title('Écart Généralisation (Train - Val)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Époque')
    axes[1, 0].set_ylabel('Écart (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Best model marker
    axes[1, 1].plot(epochs, [a*100 for a in history.val_acc], 'r-', linewidth=2)
    axes[1, 1].scatter([history.best_epoch], [history.best_val_acc*100],
                       color='gold', s=200, marker='*', zorder=5,
                       label=f'Best: {history.best_val_acc*100:.1f}%')
    axes[1, 1].set_title('Meilleure Performance Validation', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Époque')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nGraphiques sauvegardés : {save_path}")


def save_for_tensorflow(model, class_to_idx, save_path):
    """Sauvegarde le modèle en formats PyTorch et ONNX.

    L'export ONNX permet d'utiliser le modèle plus tard avec ONNX Runtime 
    ou de le convertir vers TensorFlow/TFLite si nécessaire.

    Args:
        model: Modèle PyTorch entraîné.
        class_to_idx: Dictionnaire de mapping des classes.
        save_path (Path): Chemin de base pour la sauvegarde (sans extension).
    """
    # 1. Sauvegarde native PyTorch (.pth)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
    }, save_path.with_suffix('.pth'))
    print(f"[OK] Modèle PyTorch sauvegardé : {save_path.with_suffix('.pth')}")

    # 2. Export vers ONNX (.onnx)
    model.eval()
    # Création d'une entrée factice pour tracer le graphe d'exécution
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Déplacement sur CPU pour l'export (plus stable)
    model_cpu = model.to('cpu')
    onnx_path = save_path.with_suffix('.onnx')

    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[OK] Modèle ONNX sauvegardé : {onnx_path}")


# ============================================================
# MAIN
# ============================================================
# ============================================================
# MAIN
# ============================================================
def main():
    # Définition des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Plant Doctor - Entraînement du modèle')
    parser.add_argument('--resume', action='store_true',
                       help='Reprendre à la Phase 2 en chargeant best_model_phase1.pth')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PLANT DOCTOR - ENTRAÎNEMENT PyTorch + DirectML")
    print("="*60)

    if args.resume:
        print("\n  [MODE REPRISE] Passage direct à la Phase 2")
        print("  Chargement du modèle : best_model_phase1.pth")

    # Configuration du périphérique DirectML (GPU)
    print("\nConfiguration du GPU...")
    try:
        device = torch_directml.device()
        print(f"[OK] DirectML device: {device}")
        print(f"[OK] GPU AMD/Intel détecté et prêt !")
    except Exception as e:
        print(f"[!] Erreur DirectML : {e}")
        print("[!] Fallback sur CPU...")
        device = torch.device('cpu')

    # Vérification du Dataset
    if not TRAIN_DIR.exists():
        print(f"\n[X] Dataset non trouvé : {TRAIN_DIR}")
        sys.exit(1)

    # Chargement des données
    print("\nChargement des données...")
    train_loader, val_loader, classes, class_to_idx, train_dataset, val_dataset = get_data_loaders()
    num_classes = len(classes)

    print(f"Dataset trouvé : {num_classes} classes")
    print(f"Images train : {len(train_dataset)}")
    print(f"Images valid : {len(val_dataset)}")

    # Sauvegarde des labels de classe pour l'application Flask
    class_labels = {str(v): k for k, v in class_to_idx.items()}
    labels_path = PROJECT_ROOT / "data" / "class_labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(class_labels, f, indent=2)
    print(f"Labels sauvegardés : {labels_path}")

    # Création du modèle
    print("\nConstruction du modèle ResNet50...")
    model = create_model(num_classes, device)
    print(f"Paramètres totaux : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Paramètres entraînables : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    history = TrainingHistory()

    # Initialisation du traceur temps réel
    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2
    live_plotter = LivePlotter(total_epochs)
    global_epoch = 0

    best_model_path_p1 = MODEL_DIR / 'best_model_phase1.pth'

    # ========================================
    # PHASE 1: Entraînement du classifieur seul (sauf si reprise)
    # ========================================
    if args.resume:
        # Vérification de l'existence du checkpoint Phase 1
        if not best_model_path_p1.exists():
            print(f"\n[X] ERREUR: Fichier {best_model_path_p1} non trouvé !")
            print("    Impossible de reprendre sans le modèle de Phase 1.")
            print("    Lancez d'abord un entraînement complet.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  PHASE 1 : IGNORÉE (mode reprise)")
        print(f"{'='*60}")
        print(f"\n  Chargement de : {best_model_path_p1}")
        model.load_state_dict(torch.load(best_model_path_p1, weights_only=False))

        # Initialisation rapide des métriques (validation)
        val_loss_p1, val_acc_p1 = validate(model, val_loader, criterion, device)
        print(f"  Modèle chargé - Val Acc: {val_acc_p1*100:.1f}%")

        history.best_val_acc = val_acc_p1
        history.best_epoch = EPOCHS_PHASE1
        global_epoch = EPOCHS_PHASE1
        phase2_start = EPOCHS_PHASE1

        print(f"\n  [OK] Prêt pour Phase 2 !")
    else:
        print(f"\n{'='*60}")
        print(f"  PHASE 1 : Extraction de Caractéristiques ({EPOCHS_PHASE1} époques)")
        print(f"{'='*60}")

        # Optimiseur pour la couche fully connected seulement
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
        # Scheduler pour réduire le LR si ça sagne
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.2)

        patience_counter = 0
        patience = 5

        for epoch in range(1, EPOCHS_PHASE1 + 1):
            global_epoch += 1
            epoch_start = time.time()

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, EPOCHS_PHASE1
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            epoch_time = time.time() - epoch_start
            history.epoch_times.append(epoch_time)
            history.train_loss.append(train_loss)
            history.train_acc.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)

            # Vérification du meilleur modèle
            if val_acc > history.best_val_acc:
                history.best_val_acc = val_acc
                history.best_epoch = global_epoch
                torch.save(model.state_dict(), best_model_path_p1)
                patience_counter = 0
                marker = " [*] NEW BEST"
            else:
                patience_counter += 1
                marker = ""

            # Détection Overfitting
            gap = train_acc - val_acc
            warning = " [!] ATTENTION OVERFITTING" if gap > 0.05 else ""

            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
                  f"Temps: {timedelta(seconds=int(epoch_time))}{marker}{warning}")

            # Mise à jour graphique
            live_plotter.update(global_epoch, train_loss, val_loss, train_acc, val_acc,
                               history.best_epoch, history.best_val_acc)

            scheduler.step(val_loss)

            # Early stopping
            if patience_counter >= patience:
                print(f"\n[!] Arrêt précoce déclenché (patience={patience})")
                break

            # Sauvegarde checkpoint régulier
            checkpoint_path = MODEL_DIR / f'checkpoint_p1_epoch{epoch:02d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc,
            }, checkpoint_path)

        # Recharger le meilleur modèle de la Phase 1 pour continuer
        model.load_state_dict(torch.load(best_model_path_p1, weights_only=False))
        phase2_start = global_epoch

    # ========================================
    # PHASE 2: Fine-tuning complet (tout le réseau)
    # ========================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2 : Fine-Tuning ({EPOCHS_PHASE2} époques)")
    print(f"  (Utilisation d'un batch réduit : {BATCH_SIZE_PHASE2} pour économiser la VRAM)")
    print(f"{'='*60}")

    # Nettoyage VRAM avant Phase 2 (important sur GPU limités)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Nouveaux DataLoaders avec batch size réduit
    train_loader_p2 = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PHASE2,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader_p2 = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_PHASE2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Dégeler tous les paramètres
    for param in model.parameters():
        param.requires_grad = True

    # Nouvel optimiseur avec LR très faible (pour ne pas casser les features)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 100)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.2)

    best_model_path = MODEL_DIR / 'best_model_phase2.pth'
    patience_counter = 0
    patience = 5

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        epoch_start = time.time()
        global_epoch = EPOCHS_PHASE1 + epoch

        train_loss, train_acc = train_epoch(
            model, train_loader_p2, criterion, optimizer, device, epoch, EPOCHS_PHASE2
        )
        val_loss, val_acc = validate(model, val_loader_p2, criterion, device)

        epoch_time = time.time() - epoch_start
        history.epoch_times.append(epoch_time)
        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        # Check meilleur modèle
        if val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            history.best_epoch = global_epoch
            patience_counter = 0
            marker = " [*] NEW BEST"
            
            # Sauvegarde avec nettoyage mémoire explicite pour DirectML
            import gc
            gc.collect()
            if hasattr(torch, 'dml'): # Méthode spécfique DirectML si dispo
                try: torch.dml.empty_cache() 
                except: pass
                
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            marker = ""

        gap = train_acc - val_acc
        warning = " [!] ATTENTION OVERFITTING" if gap > 0.05 else ""

        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
              f"Temps: {timedelta(seconds=int(epoch_time))}{marker}{warning}")

        # Mise à jour graphique
        live_plotter.update(global_epoch, train_loss, val_loss, train_acc, val_acc,
                           history.best_epoch, history.best_val_acc, phase2_start=phase2_start)

        scheduler.step(val_loss)

        if patience_counter >= patience:
            print(f"\n[!] Arrêt précoce déclenché (patience={patience})")
            break

    # Recharger le meilleur modèle absolu
    model.load_state_dict(torch.load(best_model_path, weights_only=False))

    # ========================================
    # ÉVALUATION FINALE
    # ========================================
    print("\n>>> Évaluation finale...")
    val_loss, val_acc = validate(model, val_loader_p2, criterion, device)
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_acc*100:.2f}%")

    # Sauvegarde finale (PyTorch + ONNX)
    final_path = MODEL_DIR / 'plant_disease_model'
    save_for_tensorflow(model, class_to_idx, final_path)

    # Sauvegarde image finale
    live_plotter.save(MODEL_DIR / 'training_history.png')

    # Rapport final
    print("\n" + "="*60)
    print("  ENTRAÎNEMENT TERMINÉ !")
    print("="*60)
    print(f"\n  Meilleure accuracy: {history.best_val_acc*100:.2f}% (époque {history.best_epoch})")
    print(f"  Accuracy finale: {val_acc*100:.2f}%")
    print(f"  Modèle PyTorch: {final_path.with_suffix('.pth')}")
    print(f"  Modèle ONNX: {final_path.with_suffix('.onnx')}")
    print(f"  Graphiques: {MODEL_DIR / 'training_history.png'}")
    print("\n  Utilisez le modèle ONNX avec ONNX Runtime pour l'inférence.")
    print("="*60 + "\n")

    # Laisser la fenêtre graphique ouverte
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
