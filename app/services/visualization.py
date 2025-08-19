"""
Visualization service for sentiment analysis results
"""
import json
from typing import Dict, List, Any
import plotly.graph_objs as go
import plotly.utils

from app.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class VisualizationService:
    """Service for creating sentiment analysis visualizations"""
    
    def __init__(self):
        self.colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6',
            'primary': '#3498db'
        }
    
    def create_sentiment_pie_chart(self, results: List[Dict[str, Any]]) -> str:
        """Create a pie chart showing sentiment distribution"""
        try:
            sentiments = [r['final_sentiment'] for r in results]
            
            sentiment_counts = {
                'Positive': sentiments.count('positive'),
                'Negative': sentiments.count('negative'),
                'Neutral': sentiments.count('neutral')
            }
            
            # Filter out zero counts
            filtered_counts = {k: v for k, v in sentiment_counts.items() if v > 0}
            
            if not filtered_counts:
                raise ValueError("No sentiment data to visualize")
            
            colors = [
                self.colors['positive'] if label == 'Positive' else
                self.colors['negative'] if label == 'Negative' else
                self.colors['neutral']
                for label in filtered_counts.keys()
            ]
            
            pie_chart = go.Figure(data=[go.Pie(
                labels=list(filtered_counts.keys()),
                values=list(filtered_counts.values()),
                hole=0.3,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            pie_chart.update_layout(
                title="Sentiment Distribution",
                showlegend=True,
                font=dict(size=12),
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            return json.dumps(pie_chart, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Failed to create pie chart: {e}")
            raise
    
    def create_confidence_histogram(self, results: List[Dict[str, Any]]) -> str:
        """Create a histogram showing confidence score distribution"""
        try:
            confidences = [r['confidence'] for r in results]
            
            if not confidences:
                raise ValueError("No confidence data to visualize")
            
            hist_chart = go.Figure(data=[go.Histogram(
                x=confidences,
                nbinsx=20,
                marker_color=self.colors['primary'],
                opacity=0.7,
                name='Confidence Scores'
            )])
            
            hist_chart.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency",
                bargap=0.1,
                font=dict(size=12),
                margin=dict(t=50, b=50, l=50, r=20)
            )
            
            return json.dumps(hist_chart, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Failed to create histogram: {e}")
            raise
    
    def create_model_comparison_chart(self, results: List[Dict[str, Any]]) -> str:
        """Create a chart comparing model predictions"""
        try:
            models = ['textblob', 'vader', 'transformer']
            model_sentiments = {model: {'positive': 0, 'negative': 0, 'neutral': 0} for model in models}
            
            for result in results:
                individual_results = result.get('individual_results', {})
                for model in models:
                    model_result = individual_results.get(model)
                    if model_result:
                        sentiment = model_result['sentiment']
                        model_sentiments[model][sentiment] += 1
            
            # Create grouped bar chart
            sentiments = ['positive', 'negative', 'neutral']
            colors = [self.colors['positive'], self.colors['negative'], self.colors['neutral']]
            
            fig = go.Figure()
            
            for i, sentiment in enumerate(sentiments):
                fig.add_trace(go.Bar(
                    name=sentiment.capitalize(),
                    x=models,
                    y=[model_sentiments[model][sentiment] for model in models],
                    marker_color=colors[i]
                ))
            
            fig.update_layout(
                title="Model Prediction Comparison",
                xaxis_title="Models",
                yaxis_title="Number of Predictions",
                barmode='group',
                font=dict(size=12),
                margin=dict(t=50, b=50, l=50, r=20)
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Failed to create model comparison chart: {e}")
            raise
    
    def create_visualization_data(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create all visualization data"""
        if not results:
            raise ValueError("No results provided for visualization")
        
        try:
            return {
                "pie_chart": self.create_sentiment_pie_chart(results),
                "histogram": self.create_confidence_histogram(results),
                "model_comparison": self.create_model_comparison_chart(results)
            }
        except Exception as e:
            logger.error(f"Failed to create visualization data: {e}")
            raise
