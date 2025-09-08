# Advanced Sentiment Analysis with Flask

A sophisticated sentiment analysis web application that combines multiple AI models for accurate sentiment detection. The application uses TextBlob, VADER, and transformer models to provide comprehensive sentiment analysis with confidence scores and visualizations.

## Features

- **Multi-Model Analysis**: Combines TextBlob, VADER, and transformer models for ensemble predictions
- **Single & Batch Processing**: Analyze individual texts or multiple texts at once
- **Interactive Web Interface**: Modern, responsive UI with real-time results
- **Data Visualizations**: Pie charts and histograms for batch analysis results
- **REST API**: Full API endpoints for programmatic access
- **Text Preprocessing**: Automatic cleaning and preprocessing of input text
- **Confidence Scoring**: Provides confidence levels for all predictions

## Models Used

1. **TextBlob**: Rule-based sentiment analysis with polarity and subjectivity scores
2. **VADER**: Lexicon and rule-based sentiment analysis optimized for social media text
3. **Transformers**: Deep learning models (RoBERTa/DistilBERT) for state-of-the-art accuracy

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatically handled on first run):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

### Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Web Interface

1. **Single Text Analysis**:
   - Enter text in the input field
   - Click "Analyze Sentiment"
   - View detailed results with model breakdown

2. **Batch Analysis**:
   - Switch to "Batch Analysis" mode
   - Enter multiple texts (one per line)
   - Click "Batch Analyze"
   - View summary statistics and visualizations

### API Endpoints

#### Analyze Single Text
```bash
POST /analyze
Content-Type: application/json

{
    "text": "I love this product! It's amazing."
}
```

**Response**:
```json
{
    "original_text": "I love this product! It's amazing.",
    "preprocessed_text": "I love this product! It's amazing.",
    "final_sentiment": "positive",
    "confidence": 0.85,
    "sentiment_scores": {
        "positive": 2.1,
        "negative": 0.0,
        "neutral": 0.2
    },
    "individual_results": {
        "textblob": {
            "sentiment": "positive",
            "polarity": 0.625,
            "subjectivity": 0.9,
            "confidence": 0.625
        },
        "vader": {
            "sentiment": "positive",
            "compound": 0.6249,
            "positive": 0.661,
            "negative": 0.0,
            "neutral": 0.339,
            "confidence": 0.6249
        },
        "transformer": {
            "sentiment": "positive",
            "confidence": 0.9998,
            "all_scores": [...]
        }
    }
}
```

#### Batch Analysis
```bash
POST /batch_analyze
Content-Type: application/json

{
    "texts": [
        "I love this!",
        "This is terrible.",
        "It's okay, I guess."
    ]
}
```

#### Create Visualizations
```bash
POST /visualize
Content-Type: application/json

{
    "results": [/* array of analysis results */]
}
```

#### Health Check
```bash
GET /health
```

### CDC Analysis Endpoint

```bash
POST /analyze_cdc
Content-Type: application/json

{
    "area1_remarks": ["string", "string"],
    "area2_remarks": ["string", "string"],
    "area3_remarks": ["string", "string"],
    "area4_remarks": ["string", "string"],
    "area5_remarks": ["string", "string"],
    "area6_remarks": ["string", "string"],
    "area7_remarks": ["string", "string"]
}
```

**Response**:
```json
{
    "results": [{
        "cdc_id": 1,
        "area1_remarks": ["string", "string"],
        "area1_sentimental": ["Positive", "Positive"],
        "area2_remarks": ["string", "string"],
        "area2_sentimental": ["Positive", "Positive"],
        "area3_remarks": ["string", "string"],
        "area3_sentimental": ["Positive", "Positive"],
        "area4_remarks": ["string", "string"],
        "area4_sentimental": ["Positive", "Positive"],
        "area5_remarks": ["string", "string"],
        "area5_sentimental": ["Positive", "Positive"],
        "area6_remarks": ["string", "string"],
        "area6_sentimental": ["Positive", "Positive"],
        "area7_remarks": ["string", "string"],
        "area7_sentimental": ["Positive", "Positive"]
    }],
    "summary": {
        "average_sentiment": "positive",
        "average_confidence": 0.85,
        "positive_count": 7,
        "negative_count": 0,
        "neutral_count": 0,
        "total_texts": 7
    }
}
```

## Technical Details

### Text Preprocessing
- URL removal
- User mention and hashtag removal
- Whitespace normalization

### Ensemble Method
The application combines predictions from multiple models using weighted averaging based on confidence scores. The final sentiment is determined by the model with the highest confidence.

### Model Loading
- TextBlob and VADER load instantly
- Transformer models are loaded asynchronously
- Fallback mechanisms ensure the app works even if transformer models fail to load

## Performance Considerations

- **Memory Usage**: Transformer models require significant memory (~500MB-1GB)
- **Processing Time**: Single text analysis: <1s, Batch processing: varies by size
- **Concurrent Users**: Flask development server supports limited concurrency

## Production Deployment

For production use:

1. **Use a production WSGI server**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Environment Variables**:
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

3. **Resource Requirements**:
   - RAM: 2GB+ recommended
   - CPU: 2+ cores for good performance
   - Storage: 1GB+ for models

## Troubleshooting

### Common Issues

1. **Transformer model fails to load**:
   - Check internet connection for model download
   - Ensure sufficient memory is available
   - App will fallback to TextBlob + VADER only

2. **NLTK data missing**:
   - Run `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

3. **Port already in use**:
   - Change port in `app.py`: `app.run(port=5001)`

### Performance Optimization

- Use GPU for transformer models (requires PyTorch GPU support)
- Implement caching for repeated analyses
- Use async processing for batch operations

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Create an issue with detailed error information
