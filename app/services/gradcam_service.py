"""Service de génération de heatmaps Grad-CAM avec PyTorch."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import io
from pathlib import Path

class GradCAMService:
    """Service pour générer des cartes de chaleur (heatmaps) Grad-CAM via PyTorch.
    
    Permet de visualiser les zones de l'image qui ont le plus contribué à la prédiction
    du modèle, aidant ainsi à l'explicabilité de l'IA.
    """
    
    def __init__(self, model_path=None, layer_name=None):
        """Initialise le service Grad-CAM.
        
        Args:
            model_path (str or Path): Chemin vers le fichier modèle .pth.
            layer_name (str): Non utilisé pour PyTorch (on cible layer4 de ResNet par défaut).
        """
        self.model = None
        # Defaut: chercher le modèle phase 2 s'il n'est pas fourni
        if not model_path:
             # Fallback intelligent vers le meilleur modèle connu
             base_dir = Path(__file__).resolve().parent.parent.parent
             self.model_path = base_dir / "models" / "best_model_phase2.pth"
        else:
            self.model_path = Path(model_path)
            
        self._model_loaded = False
        self.gradients = None
        self.activations = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Charge le modèle PyTorch et prépare l'architecture ResNet50.
        
        Returns:
            bool: True si le chargement a réussi, False sinon.
        """
        if self._model_loaded:
            return True
            
        if not self.model_path.exists():
            print(f"Erreur: Modèle Grad-CAM non trouvé à : {self.model_path}")
            # Essayer un chemin alternatif commun
            alt_path = self.model_path.parent / "plant_disease_model.pth"
            if alt_path.exists():
                self.model_path = alt_path
            else:
                return False
            
        try:
            print(f"Chargement du modèle PyTorch depuis : {self.model_path}")
            # Recréer l'architecture ResNet50 sans les poids pré-entrainés par défaut
            self.model = models.resnet50(weights=None)
            num_ftrs = self.model.fc.in_features
            
            # Reconstruction de la tête de classification (doit matcher l'entraînement)
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 38) # 38 classes de maladies
            )
            
            # Charger les poids sauvegardés
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Gérer si c'est un state_dict complet ou partiel
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval() # Mode évaluation (fige Dropout et Batchnorm)
            self._model_loaded = True
            print("Modèle Grad-CAM (PyTorch) chargé avec succès.")
            return True
            
        except Exception as e:
            print(f"Erreur chargement modèle Grad-CAM PyTorch : {e}")
            import traceback
            traceback.print_exc()
            return False

    def _tensor_hook_grad(self, grad):
        """Hook pour capturer les gradients lors de la backpropagation."""
        self.gradients = grad

    def _tensor_hook_act(self, module, input, output):
        """Hook pour capturer les activations lors de la forward pass."""
        self.activations = output

    def generate_heatmap(self, image_bytes, predicted_class_index=None):
        """Génère une heatmap Grad-CAM superposée à l'image originale.
        
        Args:
            image_bytes (bytes): Données binaires de l'image brute.
            predicted_class_index (int, optional): Index de la classe à visualiser. 
                                                   Si None, utilise la classe prédite (max).
                                                   
        Returns:
            bytes: Image PNG de la heatmap générée.
        """
        if not self._model_loaded:
            if not self.load_model():
                return None

        try:
            # 1. Préprocessing de l'image
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Transformation PyTorch standard (compatible ImageNet)
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(img).unsqueeze(0).to(self.device)
            
            # 2. Enregistrement des Hooks sur la dernière couche convolutionnelle
            # On cible 'layer4' ou 'layer2' selon le besoin de résolution vs sémantique
            target_layer = self.model.layer2
            
            # Hook pour les activations (forward pass)
            handle_act = target_layer.register_forward_hook(self._tensor_hook_act)
            
            # Hook pour les gradients (backward pass)
            handle_grad = target_layer.register_full_backward_hook(
                lambda module, grad_in, grad_out: self._tensor_hook_grad(grad_out[0])
            )

            # 3. Forward pass
            self.model.zero_grad()
            output = self.model(input_tensor)
            
            # 4. Sélection de la classe cible
            if predicted_class_index is None:
                predicted_class_index = torch.argmax(output, dim=1).item()
            else:
                # Validation de l'index
                if predicted_class_index < 0 or predicted_class_index >= output.shape[1]:
                    print(f"Index invalide {predicted_class_index}, utilisation du max.")
                    predicted_class_index = torch.argmax(output, dim=1).item()
            
            # 5. Backward pass pour calculer les gradients
            score = output[0, predicted_class_index]
            score.backward()
            
            # 6. Génération de la Heatmap
            gradients = self.gradients # [1, 2048, 7, 7] ou dimensions similaires
            activations = self.activations
            
            # Retirer les hooks pour éviter les fuites de mémoire
            handle_act.remove()
            handle_grad.remove()
            
            if gradients is None or activations is None:
                print("Erreur: Gradients ou activations manquants")
                return None
                
            # Global Average Pooling des gradients (GAP)
            # Calcule l'importance de chaque canal de feature map
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            # Pondération des activations par les gradients
            # On multiplie chaque carte d'activation par son importance (poids)
            activations = activations.squeeze(0)
            for i in range(activations.shape[0]):
                activations[i, :, :] *= pooled_gradients[i]
                
            # Somme sur les canaux pour obtenir la heatmap 2D
            heatmap = torch.sum(activations, dim=0).detach().cpu().numpy()
            
            # ReLU : on ne garde que les contributions positives
            heatmap = np.maximum(heatmap, 0)
            
            # Normalisation Min-Max entre 0 et 1
            max_val = np.max(heatmap)
            if max_val > 0:
                heatmap /= max_val
            else:
                print("Avertissement: Heatmap vide.")
                
            # 7. Post-traitement et colorisation
            # Redimensionner à la taille de l'image d'entrée (224x224)
            heatmap = cv2.resize(heatmap, (224, 224))
            
            # Conversion en format image colorée (JET colormap)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Sauvegarder en buffer mémoire
            pil_img = Image.fromarray(heatmap)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            return buf.getvalue()
            
        except Exception as e:
            print(f"Erreur lors de la génération Grad-CAM : {e}")
            import traceback
            traceback.print_exc()
            # Nettoyage de sécurité en cas de crash
            try:
                handle_act.remove()
                handle_grad.remove()
            except:
                pass
            return None

# Singleton global
_gradcam_service = None

def get_gradcam_service(app=None):
    """Factory pour obtenir l'instance unique du service Grad-CAM."""
    global _gradcam_service
    if _gradcam_service is None:
        _gradcam_service = GradCAMService()
    return _gradcam_service
