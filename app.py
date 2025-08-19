"""
Advanced Sentiment Analysis Flask Application
Main entry point using application factory pattern
"""
import os
from app.factory import create_app
from app.config import get_config

# Create application instance
app = create_app()

if __name__ == '__main__':
    config = get_config()
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
