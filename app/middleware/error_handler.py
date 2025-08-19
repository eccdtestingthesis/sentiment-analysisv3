"""
Error handling middleware for the Flask application
"""
from flask import jsonify, request
from werkzeug.exceptions import HTTPException
import traceback

from app.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class ErrorHandler:
    """Centralized error handling for the application"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize error handlers with Flask app"""
        app.register_error_handler(400, self.handle_bad_request)
        app.register_error_handler(404, self.handle_not_found)
        app.register_error_handler(405, self.handle_method_not_allowed)
        app.register_error_handler(500, self.handle_internal_error)
        app.register_error_handler(Exception, self.handle_generic_exception)
    
    def handle_bad_request(self, error):
        """Handle 400 Bad Request errors"""
        logger.warning(f"Bad request: {request.url} - {error}")
        return jsonify({
            "error": "Bad request",
            "message": str(error.description) if hasattr(error, 'description') else "Invalid request"
        }), 400
    
    def handle_not_found(self, error):
        """Handle 404 Not Found errors"""
        logger.warning(f"Not found: {request.url}")
        return jsonify({
            "error": "Not found",
            "message": "The requested resource was not found"
        }), 404
    
    def handle_method_not_allowed(self, error):
        """Handle 405 Method Not Allowed errors"""
        logger.warning(f"Method not allowed: {request.method} {request.url}")
        return jsonify({
            "error": "Method not allowed",
            "message": f"The {request.method} method is not allowed for this endpoint"
        }), 405
    
    def handle_internal_error(self, error):
        """Handle 500 Internal Server Error"""
        logger.error(f"Internal server error: {request.url} - {error}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
    
    def handle_generic_exception(self, error):
        """Handle any unhandled exceptions"""
        if isinstance(error, HTTPException):
            return error
        
        logger.error(f"Unhandled exception: {request.url} - {error}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
