import sys
import os
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ml_service import classifier
from app.services.explainer import WordImportanceExtractor

def test_phase1():
    text = "Shocking news! The local bank was robbed by a massive alien spaceship! Absolutely unbelievable and unverified."
    
    print("--- Running Classifier ---")
    res = classifier.predict(text)
    label = res["label"]
    meta = res["_metadata"]
    
    print(f"Prediction: {label}")
    
    vector = meta["vector"]
    coef = meta["coef"]
    feature_names = meta["feature_names"]
    
    indices = vector.indices
    data = vector.data
    
    print("\n--- Non-zero Indices & Contributions ---")
    contributions = []
    for i, idx in enumerate(indices):
        word = feature_names[idx]
        val = data[i]
        c = coef[idx]
        contrib = val * c
        contributions.append((word, val, c, contrib))
        print(f"Index: {idx:5d} | Word: {word:15s} | TF-IDF: {val:.4f} | Coef: {c:.4f} | Contribution: {contrib:.4f}")
    
    print("\n--- Final Selected Keywords ---")
    keywords = WordImportanceExtractor.extract(label, meta)
    print(f"Label '{label}' -> Keywords: {keywords}")

if __name__ == "__main__":
    test_phase1()
