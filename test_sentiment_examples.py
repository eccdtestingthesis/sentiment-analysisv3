#!/usr/bin/env python3
"""
Test script for specific sentiment analysis examples
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.sentiment_analyzer import EnsembleAnalyzer
from app.config import get_config


def test_specific_sentiment_examples():
    """Test the improved model with specific examples"""
    
    # Initialize analyzer
    config = get_config('development')
    analyzer = EnsembleAnalyzer(config)
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS TEST - IMPROVED MODEL")
    print("=" * 60)
    
    # 3 Positive examples
    positive_examples = [
        "Complete Requirements, The Child Health Record is updated every year",
        "All safety measures are followed",
        "Requirements fully met with excellent documentation"
    ]
    
    # 3 Negative examples  
    negative_examples = [
        "Incomplete Requirements, didn't follow on the standard",
        "Incomplete requirements",
        "Missing documentation"
    ]
    
    print("\n🟢 POSITIVE EXAMPLES:")
    print("-" * 40)
    for i, text in enumerate(positive_examples, 1):
        result = analyzer.analyze_single(text)
        sentiment = result['final_sentiment']
        confidence = result['confidence']
        
        # Color coding for output
        color = "✅" if sentiment == "positive" else "❌" if sentiment == "negative" else "⚪"
        
        print(f"{i}. {text}")
        print(f"   Result: {color} {sentiment.upper()} (confidence: {confidence:.3f})")
        
        # Show individual model results
        individual = result['individual_results']
        print(f"   TextBlob: {individual['textblob']['sentiment']} ({individual['textblob']['confidence']:.3f})")
        print(f"   VADER: {individual['vader']['sentiment']} ({individual['vader']['confidence']:.3f})")
        if individual['transformer']:
            print(f"   Transformer: {individual['transformer']['sentiment']} ({individual['transformer']['confidence']:.3f})")
        print()
    
    print("\n🔴 NEGATIVE EXAMPLES:")
    print("-" * 40)
    for i, text in enumerate(negative_examples, 1):
        result = analyzer.analyze_single(text)
        sentiment = result['final_sentiment']
        confidence = result['confidence']
        
        # Color coding for output
        color = "✅" if sentiment == "negative" else "❌" if sentiment == "positive" else "⚪"
        
        print(f"{i}. {text}")
        print(f"   Result: {color} {sentiment.upper()} (confidence: {confidence:.3f})")
        
        # Show individual model results
        individual = result['individual_results']
        print(f"   TextBlob: {individual['textblob']['sentiment']} ({individual['textblob']['confidence']:.3f})")
        print(f"   VADER: {individual['vader']['sentiment']} ({individual['vader']['confidence']:.3f})")
        if individual['transformer']:
            print(f"   Transformer: {individual['transformer']['sentiment']} ({individual['transformer']['confidence']:.3f})")
        print()
    
    # Calculate accuracy
    all_examples = [(text, "positive") for text in positive_examples] + [(text, "negative") for text in negative_examples]
    correct = 0
    total = len(all_examples)
    
    for text, expected in all_examples:
        result = analyzer.analyze_single(text)
        if result['final_sentiment'] == expected:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"\n📊 ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 60)


def test_domain_scoring():
    """Test the domain-specific scoring mechanism"""
    from app.services.sentiment_analyzer import TextPreprocessor
    
    print("\n🔍 DOMAIN SCORING TEST:")
    print("-" * 40)
    
    test_texts = [
        "Complete Requirements, The Child Health Record is updated every year",
        "Incomplete Requirements, didn't follow on the standard",
        "All safety measures are followed",
        "Missing documentation"
    ]
    
    for text in test_texts:
        domain_score = TextPreprocessor.calculate_domain_score(text)
        print(f"Text: {text}")
        print(f"Domain Score: {domain_score:.4f}")
        print()


if __name__ == "__main__":
    test_specific_sentiment_examples()
    test_domain_scoring()
