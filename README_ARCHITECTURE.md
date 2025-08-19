# Advanced Sentiment Analysis - Architecture Documentation

## Project Structure

The application has been refactored into a modular, scalable architecture following Flask best practices:

```
sentiment-analysis/
├── app/                           # Main application package
│   ├── __init__.py               # Package initialization
│   ├── factory.py                # Application factory
│   ├── config.py                 # Configuration management
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   └── routes.py             # API endpoints
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py # Core sentiment analysis
│   │   └── visualization.py      # Data visualization
│   ├── middleware/               # Middleware components
│   │   ├── __init__.py
│   │   └── error_handler.py      # Centralized error handling
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── logger.py             # Logging configuration
├── templates/                    # HTML templates
│   └── index.html
├── app.py                        # Application entry point
├── requirements.txt              # Dependencies
├── test_api.py                   # API tests
└── README.md                     # Main documentation
```

## Architecture Principles

### 1. **Separation of Concerns**
- **API Layer** (`app/api/`): Handles HTTP requests/responses
- **Service Layer** (`app/services/`): Contains business logic
- **Middleware** (`app/middleware/`): Cross-cutting concerns
- **Utils** (`app/utils/`): Shared utilities

### 2. **Dependency Injection**
- Configuration is injected into services
- Services are initialized with their dependencies
- Loose coupling between components

### 3. **Single Responsibility Principle**
- Each class has one clear responsibility
- `TextPreprocessor`: Text cleaning
- `TextBlobAnalyzer`: TextBlob-specific analysis
- `VaderAnalyzer`: VADER-specific analysis
- `TransformerAnalyzer`: Transformer-specific analysis
- `EnsembleAnalyzer`: Combines multiple analyzers

### 4. **Configuration Management**
- Environment-based configuration
- Centralized settings in `config.py`
- Support for development, production, and testing environments

### 5. **Error Handling**
- Centralized error handling middleware
- Consistent error response format
- Proper logging of errors

### 6. **Logging**
- Structured logging throughout the application
- Configurable log levels
- Consistent log formatting

## Key Components

### Configuration (`app/config.py`)
```python
# Environment-based configuration
config = get_config('production')  # or 'development', 'testing'

# Access configuration
max_length = config.MAX_TEXT_LENGTH
model_config = config.get_model_config()
```

### Application Factory (`app/factory.py`)
```python
# Create application with specific configuration
app = create_app('production')
```

### Service Layer (`app/services/`)
- **Modular analyzers**: Each model has its own analyzer class
- **Ensemble approach**: Combines multiple models intelligently
- **Error resilience**: Graceful fallbacks when models fail

### API Layer (`app/api/routes.py`)
- **Blueprint-based routing**: Organized API endpoints
- **Input validation**: Proper request validation
- **Error responses**: Consistent error handling

## Usage Examples

### Running the Application
```python
from app.factory import create_app

# Development
app = create_app('development')
app.run(debug=True)

# Production
app = create_app('production')
# Use with gunicorn: gunicorn -w 4 app:app
```

### Using Services Directly
```python
from app.services.sentiment_analyzer import EnsembleAnalyzer
from app.config import get_config

config = get_config()
analyzer = EnsembleAnalyzer(config)

# Single analysis
result = analyzer.analyze_single("I love this product!")

# Batch analysis
results = analyzer.analyze_batch(["Great!", "Terrible!", "Okay"])
```

### Custom Configuration
```python
import os
os.environ['FLASK_ENV'] = 'production'
os.environ['SECRET_KEY'] = 'your-secret-key'
os.environ['LOG_LEVEL'] = 'WARNING'

app = create_app()  # Will use production config
```

## Benefits of This Architecture

### 1. **Maintainability**
- Clear separation of concerns
- Easy to locate and modify specific functionality
- Consistent code organization

### 2. **Testability**
- Services can be tested independently
- Mock dependencies easily
- Clear interfaces between components

### 3. **Scalability**
- Easy to add new sentiment analysis models
- Modular components can be scaled independently
- Configuration-driven behavior

### 4. **Flexibility**
- Environment-specific configurations
- Easy to swap implementations
- Plugin-like architecture for analyzers

### 5. **Production Ready**
- Proper error handling and logging
- Security considerations
- Performance optimizations

## Development Workflow

### Adding a New Analyzer
1. Create analyzer class in `app/services/sentiment_analyzer.py`
2. Implement the required interface
3. Add to `EnsembleAnalyzer`
4. Update configuration if needed
5. Add tests

### Adding New API Endpoints
1. Add route to `app/api/routes.py`
2. Implement business logic in services
3. Add error handling
4. Update tests

### Environment Configuration
1. Update `app/config.py` with new settings
2. Set environment variables as needed
3. Update documentation

## Testing

The modular architecture makes testing straightforward:

```python
# Test individual components
def test_textblob_analyzer():
    config = get_config('testing')
    analyzer = TextBlobAnalyzer(config)
    result = analyzer.analyze("Great product!")
    assert result['sentiment'] == 'positive'

# Test services
def test_ensemble_analyzer():
    config = get_config('testing')
    analyzer = EnsembleAnalyzer(config)
    result = analyzer.analyze_single("I love this!")
    assert 'final_sentiment' in result
```

## Deployment

### Development
```bash
python app.py
```

### Production
```bash
export FLASK_ENV=production
export SECRET_KEY=your-production-secret
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

This architecture provides a solid foundation for a production-ready sentiment analysis application that's maintainable, testable, and scalable.
