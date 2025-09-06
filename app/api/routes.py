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


@api_bp.route('/analyze_cdc', methods=['POST'])
def analyze_cdc():
    """Analyze sentiment for CDC remarks across different areas"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate input format
        required_areas = [
            'area1_remarks', 'area2_remarks', 'area3_remarks',
            'area4_remarks', 'area5_remarks', 'area6_remarks',
            'area7_remarks'
        ]
        
        for area in required_areas:
            if area not in data:
                return jsonify({"error": f"Missing {area} in request"}), 400
            if not isinstance(data[area], list):
                return jsonify({"error": f"{area} must be a list"}), 400

        # Analyze each area
        results = {
            "cdc_id": 1,  # You might want to make this dynamic
        }
        
        total_positive = 0
        total_negative = 0
        total_neutral = 0
        total_confidence = 0
        total_texts = 0

        for area in required_areas:
            remarks = data[area]
            sentiments = []
            
            for remark in remarks:
                if remark.strip():
                    analysis = analyzer.analyze_single(remark)
                    sentiments.append(analysis['final_sentiment'].capitalize())
                    
                    # Update counters
                    if analysis['final_sentiment'] == 'positive':
                        total_positive += 1
                    elif analysis['final_sentiment'] == 'negative':
                        total_negative += 1
                    else:
                        total_neutral += 1
                    
                    total_confidence += analysis['confidence']
                    total_texts += 1

            # Store results
            results[area] = remarks
            results[f"{area[:-8]}_sentimental"] = sentiments  # Convert area1_remarks to area1_sentimental

        # Calculate summary
        avg_confidence = total_confidence / total_texts if total_texts > 0 else 0
        
        # Determine average sentiment
        if total_positive > total_negative and total_positive > total_neutral:
            avg_sentiment = "positive"
        elif total_negative > total_positive and total_negative > total_neutral:
            avg_sentiment = "negative"
        else:
            avg_sentiment = "neutral"

        response = {
            "results": [results],
            "summary": {
                "average_sentiment": avg_sentiment,
                "average_confidence": avg_confidence,
                "positive_count": total_positive,
                "negative_count": total_negative,
                "neutral_count": total_neutral,
                "total_texts": total_texts
            }
        }

        return jsonify(response)

    except ValueError as e:
        logger.warning(f"Validation error in analyze_cdc: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"CDC analysis error: {e}")
        return jsonify({"error": "Internal server error"}), 500


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
