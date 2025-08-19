"""
API routes for sentiment analysis application
"""
from flask import Blueprint, request, jsonify, render_template
from typing import Dict, Any

from app.services.sentiment_analyzer import EnsembleAnalyzer
from app.services.visualization import VisualizationService
from app.utils.logger import get_app_logger
from app.config import get_config

logger = get_app_logger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
config = get_config()
analyzer = EnsembleAnalyzer(config)
viz_service = VisualizationService()


@api_bp.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@api_bp.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment for a single text"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        result = analyzer.analyze_single(text)
        return jsonify(result)
    
    except ValueError as e:
        logger.warning(f"Validation error in analyze_sentiment: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@api_bp.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze sentiment for multiple texts"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        texts = data.get('texts', [])
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "No texts provided or invalid format"}), 400
        
        if not texts:
            return jsonify({"error": "Empty text list provided"}), 400
        
        result = analyzer.analyze_batch(texts)
        return jsonify(result)
    
    except ValueError as e:
        logger.warning(f"Validation error in batch_analyze: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@api_bp.route('/visualize', methods=['POST'])
def create_visualization():
    """Create visualization for sentiment analysis results"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        results = data.get('results', [])
        if not results:
            return jsonify({"error": "No results provided"}), 400
        
        viz_data = viz_service.create_visualization_data(results)
        return jsonify(viz_data)
    
    except ValueError as e:
        logger.warning(f"Validation error in create_visualization: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        model_status = analyzer.get_model_status()
        return jsonify({
            "status": "healthy",
            "models_loaded": model_status,
            "config": {
                "max_text_length": config.MAX_TEXT_LENGTH,
                "batch_size_limit": config.BATCH_SIZE_LIMIT
            }
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@api_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({"error": "Method not allowed"}), 405


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500
