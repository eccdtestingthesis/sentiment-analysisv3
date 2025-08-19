"""
Application factory for creating Flask app instances
"""
from flask import Flask
from flask_cors import CORS

from app.config import get_config
from app.api.routes import api_bp
from app.middleware.error_handler import ErrorHandler
from app.utils.logger import setup_logger


def create_app(config_name=None):
    """
    Application factory function
    
    Args:
        config_name: Configuration environment name
    
    Returns:
        Flask application instance
    """
    # Create Flask app
    app = Flask(__name__, template_folder='../templates')
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup logging
    setup_logger('sentiment_analysis', config.LOG_LEVEL, config.LOG_FORMAT)
    
    # Initialize CORS
    CORS(app)
    
    # Initialize error handling
    error_handler = ErrorHandler(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    return app
