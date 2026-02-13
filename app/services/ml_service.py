"""Service de Machine Learning pour le diagnostic des maladies."""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io


class MLService:
    """Service pour charger le modèle et effectuer des prédictions.

    Supporte les modèles ONNX (recommandé pour la production) et TensorFlow (.h5, legacy).
    Gère le pré-traitement des images et l'inférence.
    """

    # Dimensions attendues par le modèle ResNet50
    IMG_SIZE = (224, 224)

    # Moyennes et écarts-types ImageNet pour la normalisation
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path=None, class_labels_path=None):
        """Initialise le service ML.

        Args:
            model_path (str or Path): Chemin vers le fichier modèle (.onnx ou .h5).
            class_labels_path (str or Path): Chemin vers le fichier JSON des labels de classes.
        """
        self.model = None
        self.session = None  # Session pour ONNX Runtime
        self.class_labels = None
        self.model_path = model_path
        self.class_labels_path = class_labels_path
        self._model_loaded = False
        self._model_type = None  # 'onnx' ou 'tensorflow'

    def load_model(self):
        """Charge le modèle depuis le disque en mémoire.

        Cette méthode détermine automatiquement le type de modèle (ONNX ou TensorFlow)
        en fonction de l'extension du fichier.

        Returns:
            bool: True si le chargement a réussi, False sinon.
        """
        if self._model_loaded:
            return True

        if not self.model_path:
            print("Erreur: Chemin du modèle non spécifié.")
            return False

        model_path = Path(self.model_path)

        if not model_path.exists():
            print(f"Erreur: Modèle non trouvé à l'emplacement : {self.model_path}")
            return False

        try:
            # Déterminer le type de modèle selon l'extension
            if model_path.suffix.lower() == '.onnx':
                return self._load_onnx_model(model_path)
            elif model_path.suffix.lower() in ['.h5', '.keras']:
                return self._load_tensorflow_model(model_path)
            else:
                print(f"Format de modèle non supporté : {model_path.suffix}")
                return False

        except Exception as e:
            print(f"Erreur critique lors du chargement du modèle : {e}")
            return False

    def _load_onnx_model(self, model_path):
        """Charge un modèle au format ONNX via onnxruntime.

        Args:
            model_path (Path): Chemin vers le fichier .onnx.

        Returns:
            bool: True si le chargement a réussi.
        """
        try:
            import onnxruntime as ort

            # Configuration des options de session pour la performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Utiliser CPU par défaut (plus compatible et suffisant pour l'inférence simple)
            providers = ['CPUExecutionProvider']

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )

            self._model_type = 'onnx'
            self._model_loaded = True
            print(f"Modèle ONNX chargé avec succès : {model_path}")
            return True

        except ImportError:
            print("Erreur: ONNX Runtime non installé. Installez avec: pip install onnxruntime")
            return False
        except Exception as e:
            print(f"Erreur lors du chargement du modèle ONNX : {e}")
            return False

    def _load_tensorflow_model(self, model_path):
        """Charge un modèle TensorFlow/Keras.

        Args:
            model_path (Path): Chemin vers le fichier .h5 ou .keras.

        Returns:
            bool: True si le chargement a réussi.
        """
        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(str(model_path))
            self._model_type = 'tensorflow'
            self._model_loaded = True
            print(f"Modèle TensorFlow chargé avec succès : {model_path}")
            return True

        except ImportError:
            print("Erreur: TensorFlow non installé. Installez avec: pip install tensorflow")
            return False
        except Exception as e:
            print(f"Erreur lors du chargement du modèle TensorFlow : {e}")
            return False

    def load_class_labels(self):
        """Charge les labels de classes depuis le fichier JSON configuré.

        Returns:
            bool: True si le chargement a réussi, False sinon.
        """
        if self.class_labels is not None:
            return True

        try:
            if not self.class_labels_path or not Path(self.class_labels_path).exists():
                print(f"Erreur: Fichier de labels non trouvé à : {self.class_labels_path}")
                return False

            with open(self.class_labels_path, 'r', encoding='utf-8') as f:
                self.class_labels = json.load(f)

            print(f"Labels chargés : {len(self.class_labels)} classes disponibles.")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement des labels : {e}")
            return False

    def preprocess_image(self, image_bytes):
        """Prépare une image brute pour l'inférence (resize, normalisation).

        Args:
            image_bytes (bytes): Données binaires de l'image.

        Returns:
            numpy.ndarray: Image préprocessée prête pour le modèle.
        """
        # Charger l'image depuis les octets
        img = Image.open(io.BytesIO(image_bytes))

        # Convertir en RGB si nécessaire (ex: PNG avec transparence)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Redimensionner à la taille d'entrée du modèle (224x224)
        img = img.resize(self.IMG_SIZE, Image.Resampling.LANCZOS)

        # Convertir en array numpy float32
        img_array = np.array(img, dtype=np.float32)

        if self._model_type == 'onnx':
            # Normalisation spécifique pour PyTorch/ONNX (format ImageNet)
            # 1. Convertir de [0, 255] à [0, 1]
            img_array = img_array / 255.0
            # 2. Normaliser avec moyenne et écart-type ImageNet
            img_array = (img_array - self.IMAGENET_MEAN) / self.IMAGENET_STD
            # 3. Changer l'ordre des dimensions: (H, W, C) -> (C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))
            # 4. Ajouter dimension batch: (C, H, W) -> (1, C, H, W)
            img_array = np.expand_dims(img_array, axis=0)
        else:
            # Pour TensorFlow, on garde généralement (1, H, W, C)
            img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image_bytes):
        """Effectue une prédiction sur une image donnée.

        Args:
            image_bytes (bytes): Données binaires de l'image.

        Returns:
            dict: Résultat de la prédiction contenant :
                - class_id (int): ID de la classe prédite.
                - class_name (str): Nom lisible de la classe.
                - confidence (float): Score de confiance (0-1).
                - all_predictions (list): Top 5 des prédictions avec scores.
                - error (str, optional): Message d'erreur si échec.
        """
        # Vérifier que le modèle est chargé
        if not self._model_loaded:
            if not self.load_model():
                return {
                    'error': 'Service ML non disponible (modèle non chargé)',
                    'class_id': None,
                    'class_name': None,
                    'confidence': 0.0
                }

        # Vérifier que les labels sont chargés
        if self.class_labels is None:
            if not self.load_class_labels():
                return {
                    'error': 'Service ML non disponible (labels non chargés)',
                    'class_id': None,
                    'class_name': None,
                    'confidence': 0.0
                }

        try:
            # Pré-traitement de l'image
            img_array = self.preprocess_image(image_bytes)

            # Exécuter l'inférence selon le type de modèle
            if self._model_type == 'onnx':
                probs = self._predict_onnx(img_array)
            else:
                probs = self._predict_tensorflow(img_array)

            # Identifier la classe avec la plus haute probabilité
            top_idx = np.argmax(probs)
            confidence = float(probs[top_idx])

            # Obtenir le nom de la classe
            class_name = self.class_labels.get(str(top_idx), f"Inconnu_{top_idx}")

            # Récupérer le Top 5 des prédictions
            top_5_indices = np.argsort(probs)[-5:][::-1]
            top_5 = [
                {
                    'class_id': int(idx),
                    'class_name': self.class_labels.get(str(idx), f"Inconnu_{idx}"),
                    'confidence': float(probs[idx])
                }
                for idx in top_5_indices
            ]

            return {
                'class_id': int(top_idx),
                'class_name': class_name,
                'confidence': confidence,
                'all_predictions': top_5
            }

        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return {
                'error': str(e),
                'class_id': None,
                'class_name': None,
                'confidence': 0.0
            }

    def _predict_onnx(self, img_array):
        """Effectue une prédiction bas niveau avec le modèle ONNX.

        Args:
            img_array (numpy.ndarray): Image préprocessée (1, C, H, W).

        Returns:
            numpy.ndarray: Probabilités pour chaque classe (1D array).
        """
        # Obtenir le nom du nœud d'entrée du modèle
        input_name = self.session.get_inputs()[0].name

        # Inférence
        outputs = self.session.run(None, {input_name: img_array})
        logits = outputs[0][0]

        # Appliquer Softmax pour obtenir des probabilités
        exp_logits = np.exp(logits - np.max(logits))  # Astuce pour stabilité numérique
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def _predict_tensorflow(self, img_array):
        """Effectue une prédiction bas niveau avec le modèle TensorFlow.

        Args:
            img_array (numpy.ndarray): Image préprocessée (1, H, W, C).

        Returns:
            numpy.ndarray: Probabilités pour chaque classe (1D array).
        """
        import tensorflow as tf

        # Appliquer le preprocessing spécifique à EfficientNet de Keras
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Inférence (verbose=0 pour silencieux)
        predictions = self.model.predict(img_array, verbose=0)
        probs = predictions[0]

        return probs

    def is_ready(self):
        """Vérifie si le service est prêt à effectuer des prédictions.

        Returns:
            bool: True si modèle et labels sont chargés.
        """
        return self._model_loaded and self.class_labels is not None


# Instance globale du service (pattern Singleton)
_ml_service = None


def get_ml_service(app=None):
    """Factory pour obtenir l'instance unique du service ML.

    Args:
        app (Flask, optional): Instance Flask pour accéder à la configuration lors de la première création.

    Returns:
        MLService: L'instance unique du service ML.
    """
    global _ml_service

    if _ml_service is None:
        if app is not None:
            _ml_service = MLService(
                model_path=app.config.get('MODEL_PATH'),
                class_labels_path=app.config.get('CLASS_LABELS_PATH')
            )
        else:
            _ml_service = MLService()

    return _ml_service


def reset_ml_service():
    """Réinitialise l'instance du service (utile pour les tests unitaires)."""
    global _ml_service
    _ml_service = None
