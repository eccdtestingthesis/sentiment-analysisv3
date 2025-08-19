"""
Advanced sentiment analysis service using multiple models
"""
import re
import nltk
from typing import Dict, Any, Optional, List
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import numpy as np

from app.utils.logger import get_app_logger
from app.config import Config

logger = get_app_logger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    # Domain-specific keyword mappings for compliance/requirements context
    POSITIVE_KEYWORDS = {
        'complete': 2.0, 'completed': 2.0, 'fully': 1.5, 'excellent': 2.0,
        'all': 1.2, 'followed': 1.5, 'updated': 3.0, 'met': 1.5,
        'safety': 1.2, 'measures': 1.0, 'documentation': 1.0,
        'requirements': 0.5, 'standard': 0.5
    }
    
    NEGATIVE_KEYWORDS = {
        'incomplete': -2.0, 'missing': -2.0, "didn't": -1.8, 'not': -1.5,
        'failed': -2.0, 'violation': -2.0, 'breach': -2.0, 'lacking': -1.8,
        'insufficient': -1.8, 'partial': -1.2, 'requirements': 0.0
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    @staticmethod
    def validate_text(text: str, max_length: int = 5000) -> bool:
        """Validate text input"""
        if not text or not text.strip():
            return False
        if len(text) > max_length:
            return False
        return True
    
    @classmethod
    def calculate_domain_score(cls, text: str) -> float:
        """Calculate domain-specific sentiment score"""
        text_lower = text.lower()
        words = text_lower.split()
        
        score = 0.0
        word_count = 0
        
        for word in words:
            if word in cls.POSITIVE_KEYWORDS:
                score += cls.POSITIVE_KEYWORDS[word]
                word_count += 1
            elif word in cls.NEGATIVE_KEYWORDS:
                score += cls.NEGATIVE_KEYWORDS[word]
                word_count += 1
        
        # Normalize by word count to avoid bias toward longer texts
        if word_count > 0:
            score = score / len(words)  # Normalize by total words
        
        return score


class TextBlobAnalyzer:
    """TextBlob sentiment analysis"""
    
    def __init__(self, config: Config):
        self.positive_threshold = config.POSITIVE_THRESHOLD
        self.negative_threshold = config.NEGATIVE_THRESHOLD
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob with domain enhancement"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Get domain-specific score
        domain_score = TextPreprocessor.calculate_domain_score(text)
        
        # Combine TextBlob polarity with domain score
        enhanced_polarity = polarity + (domain_score * 0.7)  # Weight domain score
        
        # Convert enhanced polarity to sentiment label with adjusted thresholds
        if enhanced_polarity > 0.05:  # Lower threshold for positive
            sentiment = "positive"
        elif enhanced_polarity < -0.05:  # Lower threshold for negative
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "polarity": enhanced_polarity,
            "original_polarity": polarity,
            "domain_score": domain_score,
            "subjectivity": subjectivity,
            "confidence": abs(enhanced_polarity)
        }


class VaderAnalyzer:
    """VADER sentiment analysis"""
    
    def __init__(self, config: Config):
        self.analyzer = SentimentIntensityAnalyzer()
        self.positive_threshold = config.VADER_POSITIVE_THRESHOLD
        self.negative_threshold = config.VADER_NEGATIVE_THRESHOLD
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER with domain enhancement"""
        scores = self.analyzer.polarity_scores(text)
        
        # Get domain-specific score
        domain_score = TextPreprocessor.calculate_domain_score(text)
        
        # Enhance compound score with domain knowledge
        compound = scores['compound']
        enhanced_compound = compound + (domain_score * 0.8)  # Higher weight for VADER
        
        # Determine sentiment based on enhanced compound score with lower thresholds
        if enhanced_compound >= 0.03:  # Lower threshold for positive
            sentiment = "positive"
        elif enhanced_compound <= -0.03:  # Lower threshold for negative
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "compound": enhanced_compound,
            "original_compound": compound,
            "domain_score": domain_score,
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu'],
            "confidence": abs(enhanced_compound)
        }


class TransformerAnalyzer:
    """Transformer-based sentiment analysis"""
    
    def __init__(self, config: Config):
        self.analyzer = None
        self.config = config
        self._load_model()
    
    def _load_model(self):
        """Load transformer model with fallback"""
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model=self.config.TRANSFORMER_MODEL_PRIMARY,
                return_all_scores=True
            )
            logger.info(f"Primary transformer model loaded: {self.config.TRANSFORMER_MODEL_PRIMARY}")
        except Exception as e:
            logger.warning(f"Could not load primary transformer model: {e}")
            try:
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.config.TRANSFORMER_MODEL_FALLBACK,
                    return_all_scores=True
                )
                logger.info(f"Fallback transformer model loaded: {self.config.TRANSFORMER_MODEL_FALLBACK}")
            except Exception as e2:
                logger.error(f"Could not load any transformer model: {e2}")
                self.analyzer = None
    
    def analyze(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment using transformer model"""
        if not self.analyzer:
            return None
        
        try:
            results = self.analyzer(text)
            
            # Handle different model outputs
            if isinstance(results[0], list):
                scores = results[0]
            else:
                scores = results
            
            # Find the highest scoring sentiment
            best_result = max(scores, key=lambda x: x['score'])
            
            # Normalize label names
            label = best_result['label'].lower()
            if 'pos' in label or label == 'label_2':
                sentiment = "positive"
            elif 'neg' in label or label == 'label_0':
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "confidence": best_result['score'],
                "all_scores": scores
            }
        except Exception as e:
            logger.error(f"Transformer analysis failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if transformer model is available"""
        return self.analyzer is not None


class EnsembleAnalyzer:
    """Ensemble sentiment analyzer combining multiple models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.textblob_analyzer = TextBlobAnalyzer(config)
        self.vader_analyzer = VaderAnalyzer(config)
        self.transformer_analyzer = TransformerAnalyzer(config)
        
        # Download NLTK data if needed
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords')
    
    def analyze_single(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment for a single text"""
        if not self.preprocessor.validate_text(text, self.config.MAX_TEXT_LENGTH):
            raise ValueError("Invalid text input")
        
        preprocessed_text = self.preprocessor.clean_text(text)
        
        # Get results from all models
        textblob_result = self.textblob_analyzer.analyze(preprocessed_text)
        vader_result = self.vader_analyzer.analyze(preprocessed_text)
        transformer_result = self.transformer_analyzer.analyze(preprocessed_text)
        
        # Calculate ensemble sentiment
        sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
        confidence_sum = 0
        model_count = 0
        
        # TextBlob contribution
        sentiment_scores[textblob_result["sentiment"]] += textblob_result["confidence"]
        confidence_sum += textblob_result["confidence"]
        model_count += 1
        
        # VADER contribution
        sentiment_scores[vader_result["sentiment"]] += vader_result["confidence"]
        confidence_sum += vader_result["confidence"]
        model_count += 1
        
        # Transformer contribution (if available)
        if transformer_result:
            sentiment_scores[transformer_result["sentiment"]] += transformer_result["confidence"]
            confidence_sum += transformer_result["confidence"]
            model_count += 1
        
        # Determine final sentiment
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        avg_confidence = confidence_sum / model_count if model_count > 0 else 0
        
        return {
            "original_text": text,
            "preprocessed_text": preprocessed_text,
            "final_sentiment": final_sentiment,
            "confidence": avg_confidence,
            "sentiment_scores": sentiment_scores,
            "individual_results": {
                "textblob": textblob_result,
                "vader": vader_result,
                "transformer": transformer_result
            }
        }
    
    def analyze_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment for multiple texts"""
        if len(texts) > self.config.BATCH_SIZE_LIMIT:
            raise ValueError(f"Batch size exceeds limit of {self.config.BATCH_SIZE_LIMIT}")
        
        results = []
        for i, text in enumerate(texts):
            if text.strip():
                try:
                    result = self.analyze_single(text)
                    result['index'] = i
                    results.append(result)
                except ValueError as e:
                    logger.warning(f"Skipping invalid text at index {i}: {e}")
        
        # Calculate summary statistics
        if not results:
            raise ValueError("No valid texts to analyze")
        
        sentiments = [r['final_sentiment'] for r in results]
        summary = {
            "total_texts": len(results),
            "positive_count": sentiments.count('positive'),
            "negative_count": sentiments.count('negative'),
            "neutral_count": sentiments.count('neutral'),
            "average_confidence": np.mean([r['confidence'] for r in results])
        }
        
        return {
            "results": results,
            "summary": summary
        }
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {
            "textblob": True,
            "vader": True,
            "transformer": self.transformer_analyzer.is_available()
        }
