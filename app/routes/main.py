"""Définition des routes principales (Frontend) pour Plant Doctor."""
from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Page d'accueil de l'application.
    
    Affiche l'interface principale permettant l'upload de photos.
    """
    return render_template('index.html')


@main_bp.route('/result')
def result():
    """Page d'affichage du résultat de diagnostic.
    
    Cette route est généralement appelée après une soumission réussie
    sur la page d'accueil ou via une redirection JS.
    """
    return render_template('result.html')


@main_bp.route('/history')
def history():
    """Page d'historique des diagnostics.
    
    Récupère tous les diagnostics passés depuis la base de données
    et les affiche sous forme de liste chronologique.
    """
    from app.database import get_all_diagnoses
    diagnoses = get_all_diagnoses()
    return render_template('history.html', diagnoses=diagnoses)


@main_bp.route('/history/<int:diagnosis_id>')
def history_detail(diagnosis_id):
    """Page de détail d'un diagnostic archivé.
    
    Affiche la vue détaillée (comme la page de résultat) pour un
    diagnostic spécifique de l'historique.
    
    Args:
        diagnosis_id (int): ID du diagnostic à récupérer.
    """
    from app.database import get_diagnosis_by_id
    from flask import current_app, abort
    from app.services.disease_service import get_disease_service
    import json
    
    # Récupération du diagnostic brut en BDD
    diag = get_diagnosis_by_id(diagnosis_id)
    if not diag:
        abort(404)
        
    # Reconstruction de l'objet de données attendu par le template (structure identique au frontend JS)
    data = {
        'imageData': diag['image_path'],  # Chemin relatif vers l'image sauvegardée
        'prediction': {
            'class_name': diag['class_name'] if 'class_name' in diag.keys() else None,
            'confidence': diag['confidence'],
            'confidence_percent': round(diag['confidence'] * 100, 1)
        },
        'diagnosis': {
            'plant': diag['plant_name'],
            'disease_name': diag['disease_name'],
            'is_confident': True  # On suppose que les données historiques sont validées
        }
    }
    
    # Tentative de récupération des détails complets si le nom de classe est disponible
    if data['prediction']['class_name']:
        disease_service = get_disease_service(current_app)
        # Utilisation d'un seuil arbitraire (0.60) car on veut juste afficher les infos
        full_diagnosis = disease_service.get_full_diagnosis(
            data['prediction']['class_name'], 
            diag['confidence'], 
            0.60
        )
        data['diagnosis'] = full_diagnosis
    else:
        # Fallback pour les anciens enregistrements sans 'class_name' (rétrocompatibilité)
        data['diagnosis']['disease_info'] = {
            'plant': diag['plant_name'],
            'disease': diag['disease_name'],
            'description': diag['details'],
            'is_healthy': diag['is_healthy'],
            'symptoms': [],
            'treatment': {},
            'prevention': []
        }

    return render_template('result.html', diagnosis_data=data)
