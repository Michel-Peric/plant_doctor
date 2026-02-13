"""Service pour la gestion des informations sur les maladies des plantes."""
import json
from pathlib import Path


class DiseaseService:
    """Service d'accès aux informations sur les maladies et conseils de prévention.
    
    Cette classe gère le chargement et la récupération des données statiques
    concernant les pathologies, les symptômes et les traitements.
    """

    def __init__(self, disease_info_path=None, prevention_tips_path=None):
        """Initialise le service des maladies.

        Args:
            disease_info_path (str or Path): Chemin vers diseases_database.json.
            prevention_tips_path (str or Path): Chemin vers la base unifiée (même fichier).
        """
        self.disease_info_path = disease_info_path
        self.prevention_tips_path = prevention_tips_path
        self._disease_info = None
        self._prevention_tips = None

    def _load_disease_info(self):
        """Charge les informations sur les maladies depuis le fichier JSON.
        
        Returns:
            bool: True si le chargement a réussi, False sinon.
        """
        if self._disease_info is not None:
            return True

        try:
            if not self.disease_info_path or not Path(self.disease_info_path).exists():
                print(f"Erreur: Fichier d'informations maladies non trouvé : {self.disease_info_path}")
                return False

            with open(self.disease_info_path, 'r', encoding='utf-8') as f:
                self._disease_info = json.load(f)

            print(f"Info maladies chargées : {len(self._disease_info)} entrées.")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement des infos maladies : {e}")
            return False

    def _load_prevention_tips(self):
        """Charge les conseils de prévention depuis la base de données unifiée.
        
        Cette méthode extrait et structure les données de prévention par type de plante.
        
        Returns:
            bool: True si le chargement a réussi.
        """
        if self._prevention_tips is not None:
            return True

        # Utiliser la même base de données que disease_info
        if not self._load_disease_info():
            return False

        # Extraire les conseils de prévention regroupés par plante
        self._prevention_tips = {}
        for class_name, info in self._disease_info.items():
            plant_en = info.get('plant_en', '').lower()
            if plant_en and plant_en not in self._prevention_tips:
                self._prevention_tips[plant_en] = {
                    'plant': info.get('plant'),
                    'plant_en': info.get('plant_en'),
                    'tips': info.get('prevention', []),
                    'healthy_tips': info.get('healthy_tips', [])
                }

        print(f"Conseils de prévention chargés pour {len(self._prevention_tips)} types de plantes.")
        return True

    def get_disease_info(self, class_name):
        """Récupère les informations complètes sur une maladie donnée.

        Args:
            class_name (str): Clé de la classe (ex: "Tomato___Late_blight").

        Returns:
            dict: Dictionnaire d'informations sur la maladie ou None si non trouvée.
        """
        self._load_disease_info()

        if self._disease_info is None:
            return None

        return self._disease_info.get(class_name)

    def get_prevention_tips(self, plant_id):
        """Récupère les conseils de prévention spécifiques pour une espèce de plante.

        Args:
            plant_id (str): Identifiant de la plante (ex: "tomato").

        Returns:
            dict: Dictionnaire de conseils ou None si non trouvé.
        """
        self._load_prevention_tips()

        if self._prevention_tips is None:
            return None

        return self._prevention_tips.get(plant_id.lower())

    def get_all_diseases_for_plant(self, plant_en):
        """Récupère la liste de toutes les maladies recensées pour une plante donnée.

        Args:
            plant_en (str): Nom anglais de la plante (ex: "Tomato").

        Returns:
            list: Liste de dictionnaires décrivant les maladies associées.
        """
        self._load_disease_info()

        if self._disease_info is None:
            return []

        diseases = []
        plant_lower = plant_en.lower()
        for class_name, info in self._disease_info.items():
            if info.get('plant_en', '').lower() == plant_lower:
                diseases.append({
                    'class_name': class_name,
                    'disease': info.get('disease'),
                    'disease_en': info.get('disease_en'),
                    'is_healthy': info.get('is_healthy', False),
                    'severity': info.get('severity')
                })

        return diseases

    def get_full_diagnosis(self, class_name, confidence, threshold=0.90):
        """Construit un objet diagnostic complet enrichi avec les informations médicales.

        Args:
            class_name (str): Nom de la classe prédite.
            confidence (float): Score de confiance de la prédiction (0-1).
            threshold (float): Seuil de confiance pour considérer le diagnostic comme fiable.

        Returns:
            dict: Diagnostic structuré contenant toutes les informations (symptômes, traitements...).
        """
        disease_info = self.get_disease_info(class_name)

        if disease_info is None:
            return {
                'error': f'Maladie inconnue: {class_name}',
                'disease_info': None,
                'is_confident': False,
                'confidence_percent': round(confidence * 100, 1)
            }

        # Déterminer le niveau de confiance
        is_confident = confidence >= threshold
        is_low_confidence = confidence < 0.5

        # Structuration de la réponse diagnostic
        diagnosis = {
            'disease_info': disease_info,
            'is_confident': is_confident,
            'is_low_confidence': is_low_confidence,
            'confidence_percent': round(confidence * 100, 1),
            'confidence_level': self._get_confidence_level(confidence),
            'is_healthy': disease_info.get('is_healthy', False),
            'severity': disease_info.get('severity', 'unknown'),
            'plant': disease_info.get('plant'),
            'disease_name': disease_info.get('disease'),
            'description': disease_info.get('description'),
            'symptoms': disease_info.get('symptoms', []),
            'causes': disease_info.get('causes', []),
            'treatments': disease_info.get('treatments', {}),
            'prevention': disease_info.get('prevention', []),
            'immediate_actions': disease_info.get('immediate_actions', [])
        }

        # Ajouter des conseils spécifiques pour plantes saines si applicable
        if disease_info.get('is_healthy', False):
            diagnosis['healthy_tips'] = disease_info.get('healthy_tips', [])

        # Ajouter un avertissement si la confiance est trop faible
        if is_low_confidence:
            diagnosis['warning'] = "La confiance du diagnostic est faible. Nous recommandons de reprendre la photo dans de meilleures conditions."

        return diagnosis

    def _get_confidence_level(self, confidence):
        """Convertit un score numérique en niveau de confiance textuel.

        Args:
            confidence (float): Score de confiance (0-1).

        Returns:
            str: Identifiant du niveau de confiance ('tres_eleve', 'eleve', etc.).
        """
        if confidence >= 0.95:
            return 'tres_eleve'
        elif confidence >= 0.90:
            return 'eleve'
        elif confidence >= 0.70:
            return 'moyen'
        elif confidence >= 0.50:
            return 'faible'
        else:
            return 'tres_faible'

    def get_treatment_priority(self, class_name, prefer_bio=True):
        """Ordonne les traitements disponibles selon la préférence (bio ou mixte).

        Args:
            class_name (str): Nom de la classe de maladie.
            prefer_bio (bool): Si True, met les traitements biologiques en premier.

        Returns:
            list: Liste ordonnée d'objets traitements avec type et priorité.
        """
        disease_info = self.get_disease_info(class_name)

        if disease_info is None:
            return []

        treatments = disease_info.get('treatments', {})
        bio = treatments.get('bio', [])
        chemical = treatments.get('chemical', [])

        if prefer_bio:
            # Priorité : Bio puis Chimique
            result = [{'name': t, 'type': 'bio', 'priority': i + 1} for i, t in enumerate(bio)]
            result.extend([{'name': t, 'type': 'chimique', 'priority': len(bio) + i + 1} for i, t in enumerate(chemical)])
        else:
            # Traitements mélangés sans préférence stricte
            result = [{'name': t, 'type': 'bio', 'priority': i + 1} for i, t in enumerate(bio)]
            result.extend([{'name': t, 'type': 'chimique', 'priority': i + 1} for i, t in enumerate(chemical)])

        return result


# Instance globale du service (pattern Singleton)
_disease_service = None


def get_disease_service(app=None):
    """Factory pour obtenir l'instance unique du service Disease.

    Args:
        app (Flask, optional): Instance Flask pour accéder à la configuration.

    Returns:
        DiseaseService: L'instance unique du service.
    """
    global _disease_service

    if _disease_service is None:
        if app is not None:
            _disease_service = DiseaseService(
                disease_info_path=app.config.get('DISEASE_INFO_PATH'),
                prevention_tips_path=app.config.get('PREVENTION_TIPS_PATH')
            )
        else:
            _disease_service = DiseaseService()

    return _disease_service


def reset_disease_service():
    """Réinitialise l'instance du service (utile pour les tests unitaires)."""
    global _disease_service
    _disease_service = None
