import sys
import os
import json
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.v1.predict import predict_news
from app.models.prediction_schema import PredictionRequest

def test_integration():
    texts = [
        # Fake / Sensational
        "Shocking news! The local bank was robbed by a massive alien spaceship! Absolutely unbelievable and unverified. You won't believe what happens next!",
        # Genuine / Neutral
        "Apple announced a new product on Tuesday. Tim Cook presented it at the California headquarters. Read more at https://apple.com.",
        # Short / Neutral
        "This is a regular statement."
    ]
    
    for i, text in enumerate(texts):
        print(f"\n{'='*50}\n--- Integration Test {i + 1} ---")
        request = PredictionRequest(text=text)
        
        try:
            response = predict_news(request)
            
            print(f"Prediction: {response['label']} ({response['confidence']}%)")
            print("\nExplanation:")
            pprint(response["explanation"])
            
            print("\nReason Text:")
            print(response["explanation"]["reason"])
            
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_integration()
