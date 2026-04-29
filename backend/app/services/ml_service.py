import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Resolve backend root safely
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"


class FakeNewsClassifier:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.vectorizer: Optional[Any] = None

    def _load(self):
        if self.model is None or self.vectorizer is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError("Trained model not found")
            if not VECTORIZER_PATH.exists():
                raise FileNotFoundError("Vectorizer not found")

            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self._feature_names = self.vectorizer.get_feature_names_out()

    def predict(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        self._load()

        text_tfidf = self.vectorizer.transform([text.strip()])
        label = self.model.predict(text_tfidf)[0]

        # Logistic Regression supports predict_proba → safe
        proba = self.model.predict_proba(text_tfidf)[0]
        confidence = float(np.max(proba))

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "_metadata": {
                "vector": text_tfidf,
                "coef": self.model.coef_[0],
                "feature_names": self._feature_names
            }
        }


# Singleton
classifier = FakeNewsClassifier()
