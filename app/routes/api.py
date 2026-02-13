"""Définition des routes API pour l'application Plant Doctor."""
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.services.ml_service import get_ml_service
from app.services.disease_service import get_disease_service
from app.services.gradcam_service import get_gradcam_service
import base64
import os
import uuid
from app.database import save_diagnosis

api_bp = Blueprint('api', __name__, url_prefix='/api')


def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée.
    
    Args:
        filename (str): Nom du fichier à vérifier.
        
    Returns:
        bool: True si l'extension est valide, False sinon.
    """
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'jpg', 'jpeg', 'png'})
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@api_bp.route('/diagnose', methods=['POST'])
def diagnose():
    """Endpoint principal de diagnostic d'image.

    Accepte une image via une requête POST multipart/form-data.
    Effectue l'analyse, sauvegarde le résultat et retourne le diagnostic complet.

    Returns:
        JSON Response:
        - success (bool): État de la requête.
        - prediction (dict): Classe prédite et score de confiance.
        - diagnosis (dict): Informations détaillées sur la maladie (symptômes, traitements).
        - error (str, optional): Message d'erreur en cas d'échec.
    """
    # Vérifier la présence du champ 'image' dans la requête
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Aucune image fournie',
            'error_code': 'NO_IMAGE'
        }), 400

    file = request.files['image']

    # Vérifier que le fichier n'est pas vide
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Aucun fichier sélectionné',
            'error_code': 'NO_FILE'
        }), 400

    # Vérifier l'extension du fichier
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Format de fichier non supporté. Utilisez JPG ou PNG.',
            'error_code': 'INVALID_FORMAT'
        }), 400

    try:
        # Lecture des données binaires de l'image
        image_bytes = file.read()

        # Vérification de la taille (protection contre DoS)
        max_size = current_app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024)
        if len(image_bytes) > max_size:
            return jsonify({
                'success': False,
                'error': 'Image trop volumineuse (max 10 Mo)',
                'error_code': 'FILE_TOO_LARGE'
            }), 400

        # Récupération du service ML
        ml_service = get_ml_service(current_app)

        # Vérifier la disponibilité du modèle
        if not ml_service.load_model():
            return jsonify({
                'success': False,
                'error': 'Le modèle de diagnostic n\'est pas disponible. Veuillez réessayer plus tard.',
                'error_code': 'MODEL_UNAVAILABLE'
            }), 503

        # Vérifier le chargement des labels de classes
        if not ml_service.load_class_labels():
            return jsonify({
                'success': False,
                'error': 'Configuration du modèle incomplète.',
                'error_code': 'CONFIG_ERROR'
            }), 503

        # Exécution de la prédiction
        prediction = ml_service.predict(image_bytes)

        # Gestion des erreurs de prédiction
        if 'error' in prediction and prediction['class_id'] is None:
            return jsonify({
                'success': False,
                'error': prediction.get('error', 'Erreur lors de l\'analyse'),
                'error_code': 'PREDICTION_ERROR'
            }), 500

        # Enrichissement avec les informations médicales
        disease_service = get_disease_service(current_app)
        threshold = current_app.config.get('CONFIDENCE_THRESHOLD', 0.90)

        diagnosis = disease_service.get_full_diagnosis(
            prediction['class_name'],
            prediction['confidence'],
            threshold
        )
        
        # --- Sauvegarde dans l'historique ---
        try:
            # Génération d'un nom de fichier unique pour le stockage
            ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{ext}"
            
            # Création du dossier d'upload si nécessaire
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Sauvegarde physique de l'image
            save_path = os.path.join(upload_dir, unique_filename)
            # Réécriture des bytes car le pointeur de fichier est à la fin après read()
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
                
            # Chemin relatif pour l'accès web
            image_url = f"/static/uploads/{unique_filename}"
            
            # Extraction des données pour la DB
            is_healthy = diagnosis['disease_info'].get('is_healthy', False)
            plant_name = diagnosis['disease_info'].get('plant') or diagnosis['plant']
            disease_name = diagnosis['disease_info'].get('disease') or diagnosis['disease_name']
            
            # Sauvegarde en base de données
            save_diagnosis(
                filename=file.filename,
                plant_name=plant_name,
                disease_name=disease_name,
                confidence=prediction['confidence'],
                is_healthy=is_healthy,
                image_path=image_url,
                details=diagnosis['disease_info'].get('description', ''),
                class_name=prediction['class_name']
            )
            
        except Exception as e:
            # L'échec de la sauvegarde ne doit pas bloquer la réponse principale
            current_app.logger.error(f"Erreur lors de la sauvegarde de l'historique : {e}")
        
        # Construction de la réponse finale
        response = {
            'success': True,
            'prediction': {
                'class_id': prediction['class_id'],
                'class_name': prediction['class_name'],
                'confidence': prediction['confidence'],
                'confidence_percent': round(prediction['confidence'] * 100, 1),
                'top_predictions': prediction.get('all_predictions', [])
            },
            'diagnosis': diagnosis
        }

        return jsonify(response), 200

    except Exception as e:
        current_app.logger.error(f"Erreur inattendue lors du diagnostic : {e}")
        return jsonify({
            'success': False,
            'error': 'Une erreur inattendue s\'est produite',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@api_bp.route('/explain', methods=['POST'])
def explain():
    """Endpoint de génération d'explication visuelle (Grad-CAM).
    
    Génère une carte de chaleur montrant les zones de l'image ayant influencé la décision.
    
    Returns:
        JSON Response:
        - success (bool)
        - heatmap (str): Image encodée en base64 (data URI scheme).
        - error (str, optional)
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Aucune image fournie'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
        
    try:
        image_bytes = file.read()
        
        # Récupération du service ML pour mapper le nom de classe à son index
        ml_service = get_ml_service(current_app)
        class_name = request.form.get('class_name')
        class_index = None

        if class_name:
            if ml_service.load_class_labels():
                # Recherche inversée de l'index à partir du nom
                for idx_str, name in ml_service.class_labels.items():
                    if name == class_name:
                        class_index = int(idx_str)
                        break
                
                if class_index is None:
                    current_app.logger.warning(f"Classe non trouvée pour Grad-CAM: {class_name}")

        # Génération de la heatmap
        gradcam_service = get_gradcam_service(current_app)
        heatmap_bytes = gradcam_service.generate_heatmap(image_bytes, predicted_class_index=class_index)
        
        if heatmap_bytes is None:
             return jsonify({'success': False, 'error': 'Échec de la génération de la heatmap'}), 500
             
        # Encodage en base64 pour l'envoi au frontend
        heatmap_b64 = base64.b64encode(heatmap_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'heatmap': f"data:image/png;base64,{heatmap_b64}"
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Erreur lors de l'explication Grad-CAM : {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health():
    """Endpoint de vérification de santé du service (Health Check).
    
    Utile pour le monitoring et les load balancers.
    """
    ml_service = get_ml_service(current_app)
    
    return jsonify({
        'status': 'ok',
        'model_loaded': ml_service.is_ready(),
        'version': '1.0.0'
    }), 200


@api_bp.route('/disease/<class_name>', methods=['GET'])
def get_disease_info(class_name):
    """Récupère les informations statiques sur une maladie spécifique.

    Args:
        class_name (str): Nom technique de la classe (ex: Tomato___Late_blight).

    Returns:
        JSON Response.
    """
    disease_service = get_disease_service(current_app)
    info = disease_service.get_disease_info(class_name)

    if info is None:
        return jsonify({
            'success': False,
            'error': f'Maladie non trouvée: {class_name}'
        }), 404

    return jsonify({
        'success': True,
        'disease_info': info
    }), 200


@api_bp.route('/prevention/<plant_id>', methods=['GET'])
def get_prevention_tips(plant_id):
    """Récupère les conseils de prévention généraux pour une plante.

    Args:
        plant_id (str): Identifiant de la plante (ex: tomato).

    Returns:
        JSON Response.
    """
    disease_service = get_disease_service(current_app)
    tips = disease_service.get_prevention_tips(plant_id)

    if tips is None:
        return jsonify({
            'success': False,
            'error': f'Plante non trouvée: {plant_id}'
        }), 404

    return jsonify({
        'success': True,
        'plant': tips
    }), 200


@api_bp.route('/history', methods=['DELETE'])
def clear_history():
    """Efface tout l'historique des diagnostics de la base de données.
    
    Attention : Action irréversible.
    """
    try:
        from app.database import delete_all_diagnoses
        if delete_all_diagnoses():
            return jsonify({'success': True, 'message': 'Historique effacé avec succès'}), 200
        else:
            return jsonify({'success': False, 'error': 'Erreur lors de la suppression'}), 500
    except Exception as e:
        current_app.logger.error(f"Erreur lors de la suppression de l'historique : {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
