"""Point d'entree pour Plant Doctor."""
import os
from app import create_app

app = create_app(os.environ.get('FLASK_CONFIG') or 'default')

if __name__ == '__main__':
    app.run(debug=True)
