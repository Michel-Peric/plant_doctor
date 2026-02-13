"""Factory de l'application Flask pour Plant Doctor."""
from flask import Flask
from config import config


def create_app(config_name='default'):
    """Crée et configure l'instance de l'application Flask.
    
    Args:
        config_name (str): Nom de la configuration à utiliser ('development', 'production', 'testing').
    
    Returns:
        Flask: L'instance de l'application configurée.
    """
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Enregistrement des Blueprints (modules de routes)
    from app.routes.main import main_bp
    from app.routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    # Initialisation de la base de données
    from app.database import init_db, close_db
    init_db(app)
    
    # S'assure que la connexion DB est fermée à la fin de la requête
    app.teardown_appcontext(close_db)

    return app
