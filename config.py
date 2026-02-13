"""Configuration Flask pour Plant Doctor."""
import os
from pathlib import Path

# Répertoire racine du projet
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Configuration de base de l'application.
    
    Cette classe définit les paramètres par défaut communs à tous les environnements
    (développement, production, test).
    """
    # Clé secrète pour la sécurité des sessions Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'plant-doctor-dev-key-change-in-prod'

    # Paramètres d'upload de fichiers
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 Mo maximum
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

    # Paramètres du modèle
    # Supporte à la fois les modèles TensorFlow (.h5) et ONNX (.onnx)
    MODEL_PATH = BASE_DIR / 'models' / 'plant_doctor_resnet50.onnx'  # Modèle principal actuel
    MODEL_PATH_H5 = BASE_DIR / 'models' / 'efficientnet_plant_disease.h5'  # Modèle legacy (backup)
    CLASS_LABELS_PATH = BASE_DIR / 'data' / 'class_labels.json'
    
    # Paramètres Grad-CAM (visualisation de l'activation)
    # Pour EfficientNetB0, la dernière couche de convolution est souvent 'top_activation'
    GRADCAM_LAYER_NAME = 'top_activation'

    # Chemins vers les fichiers de données (base de connaissances maladies)
    DISEASE_INFO_PATH = BASE_DIR / 'data' / 'diseases_database.json'
    PREVENTION_TIPS_PATH = BASE_DIR / 'data' / 'diseases_database.json'  # Base de données unifiée
    
    # Paramètres de base de données
    DATABASE_PATH = BASE_DIR / 'plant_doctor.db'

    # Seuil de confiance minimal pour une prédiction valide
    CONFIDENCE_THRESHOLD = 0.60  # 60%


class DevelopmentConfig(Config):
    """Configuration spécifique pour l'environnement de développement."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Configuration spécifique pour l'environnement de production.
    
    Désactive le mode DEBUG et utilise des paramètres plus sécurisés si nécessaire.
    """
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Configuration spécifique pour les tests automatisés."""
    DEBUG = True
    TESTING = True


# Dictionnaire de mappage des configurations
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
