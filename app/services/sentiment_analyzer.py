"""
Advanced sentiment analysis service using multiple transformer models
"""
import re
import nltk
from typing import Dict, Any, Optional, List
from transformers import pipeline
import numpy as np

from app.utils.logger import get_app_logger
from app.config import Config

logger = get_app_logger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities"""
    
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
    


class RobertaAnalyzer:
    """RoBERTa-based sentiment analysis"""
    
    def __init__(self, config: Config):
        self.analyzer = None
        self.config = config
        self._load_model()
    
    def _load_model(self):
        """Load RoBERTa model with fallback"""
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("RoBERTa sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load RoBERTa model: {e}")
            try:
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    return_all_scores=True
                )
                logger.info("Fallback BERT sentiment model loaded")
            except Exception as e2:
                logger.error(f"Could not load any RoBERTa/BERT model: {e2}")
                self.analyzer = None
    
    def analyze(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment using RoBERTa"""
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
            
            # Normalize label names for RoBERTa Twitter model
            label = best_result['label'].lower()
            if 'positive' in label or label == 'label_2':
                sentiment = "positive"
            elif 'negative' in label or label == 'label_0':
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "confidence": best_result['score'],
                "all_scores": scores
            }
        except Exception as e:
            logger.error(f"RoBERTa analysis failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if RoBERTa model is available"""
        return self.analyzer is not None


class DistilBertAnalyzer:
    """DistilBERT-based sentiment analysis"""
    
    def __init__(self, config: Config):
        self.analyzer = None
        self.config = config
        self._load_model()
    
    def _load_model(self):
        """Load DistilBERT model with fallback"""
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            logger.info("DistilBERT sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load DistilBERT model: {e}")
            try:
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                logger.info("Fallback emotion DistilRoBERTa model loaded")
            except Exception as e2:
                logger.error(f"Could not load any DistilBERT model: {e2}")
                self.analyzer = None
    
    def analyze(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment using DistilBERT"""
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
            if 'positive' in label or label == 'label_1':
                sentiment = "positive"
            elif 'negative' in label or label == 'label_0':
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "confidence": best_result['score'],
                "all_scores": scores
            }
        except Exception as e:
            logger.error(f"DistilBERT analysis failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if DistilBERT model is available"""
        return self.analyzer is not None


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
    """Ensemble sentiment analyzer combining multiple transformer models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.roberta_analyzer = RobertaAnalyzer(config)
        self.distilbert_analyzer = DistilBertAnalyzer(config)
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
        """Analyze sentiment for a single text using transformer ensemble"""
        if not self.preprocessor.validate_text(text, self.config.MAX_TEXT_LENGTH):
            raise ValueError("Invalid text input")
        
        preprocessed_text = self.preprocessor.clean_text(text)
        
        # Get results from all transformer models
        roberta_result = self.roberta_analyzer.analyze(preprocessed_text)
        distilbert_result = self.distilbert_analyzer.analyze(preprocessed_text)
        transformer_result = self.transformer_analyzer.analyze(preprocessed_text)
        
        # Calculate ensemble sentiment using weighted averaging
        available_results = []
        if roberta_result:
            available_results.append(roberta_result)
        if distilbert_result:
            available_results.append(distilbert_result)
        if transformer_result:
            available_results.append(transformer_result)
        
        # Ensure at least one model worked
        if not available_results:
            raise RuntimeError("No transformer models are available for analysis")
        
        # Convert sentiment labels to numerical scores for averaging
        def sentiment_to_score(sentiment, confidence):
            """Convert sentiment label to numerical score weighted by confidence"""
            sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
            return sentiment_map[sentiment] * confidence
        
        # Calculate weighted average sentiment score
        total_weighted_score = 0
        total_confidence = 0
        
        for result in available_results:
            weighted_score = sentiment_to_score(result["sentiment"], result["confidence"])
            total_weighted_score += weighted_score
            total_confidence += result["confidence"]
        
        # Calculate final averaged sentiment score
        avg_sentiment_score = total_weighted_score / total_confidence if total_confidence > 0 else 0
        avg_confidence = total_confidence / len(available_results)
        
        # Convert averaged score back to sentiment label
        if avg_sentiment_score > 0.1:
            final_sentiment = "positive"
        elif avg_sentiment_score < -0.1:
            final_sentiment = "negative"
        else:
            final_sentiment = "neutral"
        
        # Calculate individual sentiment scores for display
        sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
        for result in available_results:
            sentiment_scores[result["sentiment"]] += result["confidence"]
        
        return {
            "original_text": text,
            "preprocessed_text": preprocessed_text,
            "final_sentiment": final_sentiment,
            "confidence": avg_confidence,
            "averaged_sentiment_score": avg_sentiment_score,
            "models_used": len(available_results),
            "sentiment_scores": sentiment_scores,
            "individual_results": {
                "roberta": roberta_result,
                "distilbert": distilbert_result,
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
        """Get status of all transformer models"""
        return {
            "roberta": self.roberta_analyzer.is_available(),
            "distilbert": self.distilbert_analyzer.is_available(),
            "transformer": self.transformer_analyzer.is_available()
        }
