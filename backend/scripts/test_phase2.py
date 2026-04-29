import sys
import os
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ml_service import classifier
from app.services.text_analyzer import TextAnalyzer

def test_phase2():
    text = "Shocking news!!! The local bank was robbed by a massive alien spaceship. Absolutely unbelievable and unverified. This is a very normal sentence that should score low. ALL CAPS EXCLAMATION!!!"
    
    print("--- Running Classifier ---")
    res = classifier.predict(text)
    meta = res["_metadata"]
    
    analyzer = TextAnalyzer()
    
    print("\n--- Running Sentence Analyzer ---")
    scored_sentences = analyzer.analyze(text, meta)
    
    for rank, s in enumerate(scored_sentences):
        print(f"\nRank {rank + 1}: {s['sentence']}")
        print(f"  Raw Score: {s['raw_score']:.4f}")
        print(f"  Normalized Score: {s['normalized_score']:.4f}")
        print(f"  Peak Word Score: {s['peak_word_score']:.4f}")
        print(f"  Length: {s['length']}")
        print(f"  Original Position: {s['position']}")

if __name__ == "__main__":
    test_phase2()
