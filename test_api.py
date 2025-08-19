#!/usr/bin/env python3
"""
Test script for the Advanced Sentiment Analysis API
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure the Flask app is running.")
        return False

def test_single_analysis():
    """Test single text analysis"""
    print("\nTesting single text analysis...")
    
    test_cases = [
        "I absolutely love this product! It's fantastic and works perfectly.",
        "This is the worst thing I've ever bought. Completely useless.",
        "It's okay, nothing special but not terrible either.",
        "The weather is nice today 🌞",
        "Check out this amazing deal at https://example.com #sale @everyone"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {text[:50]}...")
        
        payload = {"text": text}
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            sentiment = data['final_sentiment']
            confidence = data['confidence']
            print(f"✓ Sentiment: {sentiment} (confidence: {confidence:.2f})")
            
            # Show model breakdown
            models = data['individual_results']
            print(f"  TextBlob: {models['textblob']['sentiment']}")
            print(f"  VADER: {models['vader']['sentiment']}")
            if models['transformer']:
                print(f"  Transformer: {models['transformer']['sentiment']}")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")

def test_batch_analysis():
    """Test batch text analysis"""
    print("\nTesting batch analysis...")
    
    texts = [
        "I love this!",
        "This is terrible.",
        "It's okay.",
        "Amazing product!",
        "Could be better.",
        "Worst experience ever.",
        "Pretty good overall.",
        "Not what I expected.",
        "Excellent service!",
        "Mediocre at best."
    ]
    
    payload = {"texts": texts}
    response = requests.post(f"{BASE_URL}/batch_analyze", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        summary = data['summary']
        
        print(f"✓ Batch analysis completed:")
        print(f"  Total texts: {summary['total_texts']}")
        print(f"  Positive: {summary['positive_count']}")
        print(f"  Negative: {summary['negative_count']}")
        print(f"  Neutral: {summary['neutral_count']}")
        print(f"  Average confidence: {summary['average_confidence']:.2f}")
        
        # Test visualization endpoint
        viz_response = requests.post(f"{BASE_URL}/visualize", json={"results": data['results']})
        if viz_response.status_code == 200:
            print("✓ Visualization data generated successfully")
        else:
            print(f"✗ Visualization failed: {viz_response.status_code}")
    else:
        print(f"✗ Batch analysis failed: {response.status_code} - {response.text}")

def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")
    
    # Test empty text
    response = requests.post(f"{BASE_URL}/analyze", json={"text": ""})
    if response.status_code == 400:
        print("✓ Empty text properly rejected")
    else:
        print(f"✗ Empty text handling failed: {response.status_code}")
    
    # Test missing text field
    response = requests.post(f"{BASE_URL}/analyze", json={})
    if response.status_code == 400:
        print("✓ Missing text field properly rejected")
    else:
        print(f"✗ Missing text field handling failed: {response.status_code}")
    
    # Test invalid batch data
    response = requests.post(f"{BASE_URL}/batch_analyze", json={"texts": "not a list"})
    if response.status_code == 400:
        print("✓ Invalid batch data properly rejected")
    else:
        print(f"✗ Invalid batch data handling failed: {response.status_code}")

def main():
    """Run all tests"""
    print("Advanced Sentiment Analysis API Test Suite")
    print("=" * 50)
    
    # Check if server is running
    if not test_health_check():
        print("\nPlease start the Flask application first:")
        print("python app.py")
        return
    
    # Run tests
    test_single_analysis()
    test_batch_analysis()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo start the web interface, visit: http://localhost:5000")

if __name__ == "__main__":
    main()
