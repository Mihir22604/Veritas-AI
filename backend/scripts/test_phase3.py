import sys
import os
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.pattern_detector import PatternDetector

def test_phase3():
    texts = [
        "You won't believe what happens next! A shocking truth has been revealed. The government is hiding aliens.",
        "Apple announced a new product on Tuesday. Tim Cook presented it at the California headquarters. Read more at https://apple.com.",
        "I went to the store. The store was closed. A man told me to go away."
    ]
    
    detector = PatternDetector()
    
    for i, text in enumerate(texts):
        print(f"\n--- Testing Text {i + 1} ---")
        print(f"Text: {text}")
        result = detector.detect(text)
        pprint(result)

if __name__ == "__main__":
    test_phase3()
