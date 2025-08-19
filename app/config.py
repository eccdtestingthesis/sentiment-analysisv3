"""
Configuration management for the sentiment analysis application
"""
import os
from typing import Dict, Any


class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Application settings
    HOST = os.environ.get('FLASK_HOST') or '0.0.0.0'
    PORT = int(os.environ.get('FLASK_PORT') or 5000)
    
    # Model settings
    TRANSFORMER_MODEL_PRIMARY = "siebert/sentiment-roberta-large-english"
    TRANSFORMER_MODEL_FALLBACK = "siebert/sentiment-roberta-large-english"
    
    # Text processing settings
    MAX_TEXT_LENGTH = 5000
    BATCH_SIZE_LIMIT = 100
    
    # Sentiment thresholds
    POSITIVE_THRESHOLD = 0.1
    NEGATIVE_THRESHOLD = -0.1
    VADER_POSITIVE_THRESHOLD = 0.05
    VADER_NEGATIVE_THRESHOLD = -0.05
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'primary_model': cls.TRANSFORMER_MODEL_PRIMARY,
            'fallback_model': cls.TRANSFORMER_MODEL_FALLBACK,
            'positive_threshold': cls.POSITIVE_THRESHOLD,
            'negative_threshold': cls.NEGATIVE_THRESHOLD,
            'vader_positive_threshold': cls.VADER_POSITIVE_THRESHOLD,
            'vader_negative_threshold': cls.VADER_NEGATIVE_THRESHOLD
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Override with environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY') or Config.SECRET_KEY


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config_map.get(config_name, DevelopmentConfig)
