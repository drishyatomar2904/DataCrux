from flask import Flask
from flask_bootstrap import Bootstrap5
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Get secret key from environment
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret-key')
    
    # Create absolute path for uploads folder
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    # Create upload folder if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    from . import routes
    app.register_blueprint(routes.bp)
    
    # Initialize Bootstrap
    Bootstrap5(app)
    
    return app