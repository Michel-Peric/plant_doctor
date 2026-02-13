
import sqlite3
from flask import current_app, g
import os
from datetime import datetime

def get_db():
    """Récupère la connexion à la base de données configurée.
    
    La connexion est unique pour chaque requête et sera réutilisée si appelée à nouveau.
    Stocke la connexion dans l'objet global 'g' de Flask.
    
    Returns:
        sqlite3.Connection: Objet connexion à la base de données.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE_PATH'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        # Permet d'accéder aux colonnes par leur nom
        g.db.row_factory = sqlite3.Row

    return g.db

def close_db(e=None):
    """Ferme la connexion à la base de données si elle existe.
    
    Cette fonction est appelée automatiquement à la fin de chaque requête.
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db(app):
    """Initialise la base de données : crée les tables si elles n'existent pas.
    
    Args:
        app (Flask): L'instance de l'application Flask.
    """
    with app.app_context():
        db_path = app.config['DATABASE_PATH']
        
        # Connexion explicite pour la création de table
        conn = sqlite3.connect(
            db_path,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        c = conn.cursor()
        
        # Création de la table des diagnostics
        c.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                plant_name TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                class_name TEXT,
                confidence REAL NOT NULL,
                is_healthy BOOLEAN NOT NULL,
                image_path TEXT,
                details TEXT
            )
        ''')
        
        # Tentative de migration : ajout de la colonne class_name si elle n'existe pas
        try:
            c.execute('ALTER TABLE diagnoses ADD COLUMN class_name TEXT')
        except sqlite3.OperationalError:
            # La colonne existe probablement déjà
            pass
            
        conn.commit()
        conn.close()
        print(f"Base de données initialisée à : {db_path}")

def save_diagnosis(filename, plant_name, disease_name, confidence, is_healthy, image_path=None, details=None, class_name=None):
    """Sauvegarde un enregistrement de diagnostic en base de données.
    
    Args:
        filename (str): Nom du fichier image.
        plant_name (str): Nom de la plante.
        disease_name (str): Nom de la maladie détectée.
        confidence (float): Score de confiance de la prédiction.
        is_healthy (bool): État de santé de la plante.
        image_path (str, optional): Chemin relatif vers l'image.
        details (str, optional): Détails supplémentaires au format JSON ou texte.
        class_name (str, optional): Nom de la classe brute du modèle.
        
    Returns:
        bool: True si succès, False sinon.
    """
    try:
        db = get_db()
        db.execute(
            'INSERT INTO diagnoses (filename, plant_name, disease_name, confidence, is_healthy, image_path, details, class_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (filename, plant_name, disease_name, confidence, is_healthy, image_path, details, class_name)
        )
        db.commit()
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du diagnostic : {e}")
        return False

def get_all_diagnoses():
    """Récupère tous les diagnostics triés par date décroissante.
    
    Returns:
        list: Liste des diagnostics (sqlite3.Row objects).
    """
    try:
        db = get_db()
        cur = db.execute('SELECT * FROM diagnoses ORDER BY timestamp DESC')
        diagnoses = cur.fetchall()
        return diagnoses
    except Exception as e:
        print(f"Erreur lors de la récupération des diagnostics : {e}")
        return []

def get_diagnosis_by_id(diagnosis_id):
    """Récupère un diagnostic unique par son ID.
    
    Args:
        diagnosis_id (int): L'identifiant unique du diagnostic.
        
    Returns:
        sqlite3.Row: L'enregistrement du diagnostic ou None si non trouvé.
    """
    try:
        db = get_db()
        cur = db.execute('SELECT * FROM diagnoses WHERE id = ?', (diagnosis_id,))
        return cur.fetchone()
    except Exception as e:
        print(f"Erreur lors de la récupération du diagnostic {diagnosis_id} : {e}")
        return None

def delete_all_diagnoses():
    """Supprime tous les enregistrements de diagnostics.
    
    Attention : Cette action est irréversible.
    
    Returns:
        bool: True si succès, False sinon.
    """
    try:
        db = get_db()
        db.execute('DELETE FROM diagnoses')
        db.commit()
        return True
    except Exception as e:
        print(f"Erreur lors de la suppression des diagnostics : {e}")
        return False
