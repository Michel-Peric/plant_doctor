"""
Script pour exporter le modèle PyTorch entraîné au format ONNX.
==============================================================

Ce script permet de convertir les checkpoints PyTorch (.pth) générés par `train_model_pytorch.py`
en fichiers ONNX (.onnx). Le format ONNX est universel et permet d'utiliser le modèle
dans des environnements variés (web, mobile, C++, etc.) via ONNX Runtime.

Fonctionnalités :
- Reconstruction de l'architecture ResNet50.
- Chargement des poids depuis les checkpoints (Phase 1, Phase 2 ou automatique).
- Export avec paramètres dynamiques (batch size variable).
- Vérification automatique de la validité du modèle ONNX exporté.
- Test d'inférence avec ONNX Runtime pour confirmer le bon fonctionnement.

Usage:
    python export_onnx.py                    # Exporte le meilleur modèle trouvé (Phase 2 > Phase 1)
    python export_onnx.py --model phase1     # Force l'export du modèle de la Phase 1
    python export_onnx.py --model phase2     # Force l'export du modèle de la Phase 2
    python export_onnx.py --output mon_model.onnx  # Spécifie le fichier de sortie
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

# ============================================================
# CONFIGURATION
# ============================================================
# Chemins par défaut
MODEL_DIR = Path(__file__).parent / 'models'
OUTPUT_PATH = MODEL_DIR / 'plant_doctor_resnet50.onnx'

# Configuration du modèle (doit correspondre à l'entraînement)
NUM_CLASSES = 38
IMG_SIZE = 224


def create_model(num_classes):
    """Recrée l'architecture du modèle ResNet50 utilisée lors de l'entraînement.
    
    Args:
        num_classes (int): Nombre de classes de sortie.

    Returns:
        nn.Module: Le modèle PyTorch instancié (sur CPU).
    """
    # On ne charge pas les poids ImageNet ici (weights=None) car on va charger nos propres poids
    model = models.resnet50(weights=None)

    # Remplacement du classifieur (Fully Connected layer)
    # Doit être IDENTIQUE à ce qui est défini dans train_model_pytorch.py
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


def export_to_onnx(model_path, output_path, num_classes=38):
    """Exporte un fichier checkpoint PyTorch (.pth) vers ONNX.

    Args:
        model_path (Path): Chemin vers le fichier .pth source.
        output_path (Path): Chemin de destination pour le fichier .onnx.
        num_classes (int): Nombre de classes du modèle.

    Returns:
        bool: True si l'export et la vérification ont réussi, False sinon.
    """
    print(f"Chargement du modèle : {model_path}")

    # 1. Instancier le modèle
    model = create_model(num_classes)

    # 2. Charger les poids du checkpoint
    # map_location='cpu' est important si on exporte sur une machine sans GPU
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Gérer les différentes structures de sauvegarde possibles
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Checkpoint standard (recommandé)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Info Checkpoint - Époque : {checkpoint.get('epoch', 'N/A')}")
            print(f"  Info Checkpoint - Val Accuracy : {checkpoint.get('val_accuracy', checkpoint.get('val_acc', 'N/A'))}")
        elif 'state_dict' in checkpoint:
            # Checkpoint format Lightning ou autre
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Dictionnaire de poids direct
            model.load_state_dict(checkpoint)
    else:
        # Poids sauvegardés directement
        model.load_state_dict(checkpoint)

    # 3. Mettre le modèle en mode évaluation
    # Indispensable pour désactiver le Dropout et fixer la BatchNorm
    model.eval()

    # 4. Créer un tenseur d'entrée factice (Dummy Input)
    # Nécessaire pour que PyTorch trace le graphe d'exécution
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # 5. Exporter en ONNX
    print(f"Export en cours vers : {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,         # Inclure les poids entraînés
        opset_version=12,           # Version stable et compatible
        do_constant_folding=True,   # Optimisation (pré-calcul des constantes)
        input_names=['input'],      # Nom du nœud d'entrée
        output_names=['output'],    # Nom du nœud de sortie
        dynamic_axes={              # Pour supporter des batch sizes variables
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"[OK] Fichier ONNX généré !")

    # 6. Vérifier la validité du fichier ONNX généré
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("[OK] Structure du modèle ONNX validée par onnx.checker")
    except ImportError:
        print("[!] Bibliothèque 'onnx' non installée, validation structurelle ignorée")
    except Exception as e:
        print(f"[!] ERREUR lors de la validation ONNX : {e}")
        return False

    # 7. Test d'inférence avec ONNX Runtime (si disponible)
    try:
        import onnxruntime as ort
        import numpy as np

        # Création d'une session d'inférence
        session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        # Inférence sur des données aléatoires
        test_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        outputs = session.run(None, {input_name: test_input})

        print(f"[OK] Test d'inférence ONNX Runtime réussi !")
        print(f"     Shape entrée : {test_input.shape}")
        print(f"     Shape sortie : {outputs[0].shape}")
        
    except ImportError:
        print("[!] 'onnxruntime' non installé, test d'inférence ignoré")
    except Exception as e:
        print(f"[!] ERREUR lors du test d'inférence : {e}")
        return False

    return True


def main():
    # Analyse des arguments
    parser = argparse.ArgumentParser(description='Export modèle PyTorch en ONNX')
    parser.add_argument('--model', choices=['phase1', 'phase2', 'best'], default='best',
                        help='Choose which model to export (default: best available)')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output path (default: models/plant_doctor_resnet50.onnx)')

    args = parser.parse_args()

    # Sélection du fichier source
    if args.model == 'phase1':
        model_path = MODEL_DIR / 'best_model_phase1.pth'
    elif args.model == 'phase2':
        model_path = MODEL_DIR / 'best_model_phase2.pth'
    else:
        # Mode "best" : cherche Phase 2, sinon Phase 1
        phase2_path = MODEL_DIR / 'best_model_phase2.pth'
        phase1_path = MODEL_DIR / 'best_model_phase1.pth'

        if phase2_path.exists():
            model_path = phase2_path
            print("Mode 'best' : Modèle Phase 2 sélectionné.")
        elif phase1_path.exists():
            model_path = phase1_path
            print("Mode 'best' : Modèle Phase 1 sélectionné (Phase 2 non trouvée).")
        else:
            print("ERREUR : Aucun modèle trouvé !")
            print(f"Dossier recherché : {MODEL_DIR}")
            print("Veuillez lancer l'entraînement avec train_model_pytorch.py d'abord.")
            return

    if not model_path.exists():
        print(f"ERREUR : Le fichier modèle n'existe pas : {model_path}")
        return

    # Définition du chemin de sortie
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    # Création du dossier de sortie si absent
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Exécution de l'export
    success = export_to_onnx(model_path, output_path, NUM_CLASSES)

    if success:
        print(f"\n{'='*60}")
        print(f"SUCCÈS : Modèle exporté vers {output_path}")
        print(f"Taille du fichier : {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"{'='*60}")
    else:
        print(f"\n[!] L'export a rencontré des problèmes.")


if __name__ == '__main__':
    main()
