import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Dict, Any, Tuple

# Download necessary NLTK data safely
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except (LookupError, OSError):
    nltk.download('vader_lexicon', quiet=True)

class TextAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_lexicon = {"shocking", "unbelievable", "bombshell", "urgent", "secret", "furious", "terrified", "outrageous", "massive", "destroy"}

    def get_word_contributions_dict(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Convert sparse tf-idf and coef to a dictionary of word -> absolute contribution."""
        vector = metadata.get("vector")
        coef = metadata.get("coef")
        feature_names = metadata.get("feature_names")
        
        if vector is None or coef is None or feature_names is None:
            return {}
            
        contributions = {}
        indices = vector.indices
        data = vector.data
        
        for i, idx in enumerate(indices):
            word = feature_names[idx]
            val = data[i]
            c = coef[idx]
            contributions[word] = val * c
            
        return contributions

    def analyze(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # NLTK Sentence Tokenization
        sentences = nltk.sent_tokenize(text)
        
        # Limit processing for performance (max 50 sentences)
        sentences = sentences[:50]
        
        word_contributions = self.get_word_contributions_dict(metadata)
        
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            words = [w.lower() for w in re.findall(r'\w+', sentence)]
            
            # 1. Keyword Impact
            sentence_keyword_score = 0.0
            peak_word_score = 0.0
            for word in words:
                contrib = abs(word_contributions.get(word, 0.0))
                sentence_keyword_score += contrib
                if contrib > peak_word_score:
                    peak_word_score = contrib
                    
            # 2. Emotional Intensity
            lexicon_count = sum(1 for word in words if word in self.emotion_lexicon)
            lexicon_density = lexicon_count / max(len(words), 1)
            
            polarity = abs(self.sia.polarity_scores(sentence)['compound'])
            emotion_score = (lexicon_density * 0.4) + (polarity * 0.6)
            
            # 3. Structural Signals
            exclamations = len(re.findall(r'!', sentence))
            exclamation_score = 1.0 if exclamations > 2 else (exclamations * 0.33)
            
            chars = [c for c in sentence if c.isalpha()]
            upper_ratio = sum(1 for c in chars if c.isupper()) / max(len(chars), 1)
            structural_score = min(exclamation_score + upper_ratio, 1.0)
            
            # Raw total score
            raw_score = (sentence_keyword_score * 0.6) + (emotion_score * 0.2) + (structural_score * 0.2)
            
            scored_sentences.append({
                "sentence": sentence,
                "raw_score": raw_score,
                "peak_word_score": peak_word_score,
                "length": len(sentence),
                "position": i
            })
            
        if not scored_sentences:
            return []

        # Normalization (Zero-Division Safe)
        max_score = max(s["raw_score"] for s in scored_sentences)
        min_score = min(s["raw_score"] for s in scored_sentences)
        
        range_val = max_score - min_score
        
        for s in scored_sentences:
            if range_val < 1e-6:
                s["normalized_score"] = 1.0
            else:
                s["normalized_score"] = (s["raw_score"] - min_score) / range_val
                
        # Tie-breaking logic (Sort by Normalized Score, then Peak Word Score, then shortest length, then earliest position)
        scored_sentences.sort(key=lambda x: (
            x["normalized_score"],
            x["peak_word_score"],
            -x["length"], # Negative because we want shortest length first when sorting descending
            -x["position"] # Negative because we want earliest position first
        ), reverse=True)
        
        return scored_sentences
