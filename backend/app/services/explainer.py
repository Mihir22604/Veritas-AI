from typing import List, Dict, Any, Tuple
from app.services.text_analyzer import TextAnalyzer
from app.services.pattern_detector import PatternDetector

class WordImportanceExtractor:
    @staticmethod
    def extract(label: str, metadata: Dict[str, Any]) -> List[str]:
        vector = metadata.get("vector")
        coef = metadata.get("coef")
        feature_names = metadata.get("feature_names")

        if vector is None or coef is None or feature_names is None:
            return []

        # Extract non-zero indices from sparse CSR matrix
        indices = vector.indices
        data = vector.data

        word_contributions: List[Tuple[str, float]] = []

        for i, idx in enumerate(indices):
            word = feature_names[idx]
            tfidf_val = data[i]
            coefficient = coef[idx]
            contribution = tfidf_val * coefficient
            word_contributions.append((word, contribution))

        # Directional Filtering
        if label == "Fake":
            filtered = [(w, c) for w, c in word_contributions if c > 0]
            filtered.sort(key=lambda x: x[1], reverse=True)
        else: # Genuine
            filtered = [(w, c) for w, c in word_contributions if c < 0]
            filtered.sort(key=lambda x: x[1], reverse=False)

        if not filtered:
            word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            filtered = word_contributions

        # Deduplication
        seen = set()
        unique_words = []
        for word, contrib in filtered:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
            if len(unique_words) == 5:
                break

        return unique_words

class ReasonGenerator:
    @staticmethod
    def generate(label: str, confidence: float, keywords: List[str], patterns: List[str], top_sentence: str) -> str:
        fragments = []
        
        # Low confidence disclaimer
        if confidence < 0.60:
            fragments.append("Warning: This analysis is based on low-confidence model signals. The indicators below may be subtle.")
            
        fragments.append(f"The model predicts this as {label}.")
        
        if patterns:
            pattern_str = ", ".join(patterns)
            fragments.append(f"The text shows signs of: {pattern_str}.")
            
        if keywords or top_sentence:
            evidence_fragments = []
            if keywords:
                keyword_str = ", ".join([f"'{k}'" for k in keywords])
                evidence_fragments.append(f"keywords like {keyword_str}")
            if top_sentence:
                evidence_fragments.append(f"the tone in sentence: '{top_sentence}'")
                
            combined_evidence = " and ".join(evidence_fragments)
            fragments.append(f"The reasoning is heavily influenced by {combined_evidence}.")
            
        if not patterns and not keywords:
            fragments.append("The classification is based on general linguistic structure rather than specific suspicious markers.")
            
        return " ".join(fragments)

class ExplanationService:
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.pattern_detector = PatternDetector()

    def generate(self, text: str, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        label = prediction_result["label"]
        confidence = prediction_result.get("confidence", 100.0) / 100.0 # Convert back to 0-1 scale for logic
        metadata = prediction_result.get("_metadata", {})
        
        # Phase 1: Keywords
        keywords = WordImportanceExtractor.extract(label, metadata)
        
        # Phase 2: Sentences
        scored_sentences = self.text_analyzer.analyze(text, metadata)
        highlighted_sentences = [s["sentence"] for s in scored_sentences[:2]] # Top 2
        top_sentence = highlighted_sentences[0] if highlighted_sentences else ""
        
        # Phase 3: Patterns
        pattern_result = self.pattern_detector.detect(text)
        patterns_detected = pattern_result["patterns_detected"]
        
        # Phase 4: Reason
        reason = ReasonGenerator.generate(label, confidence, keywords, patterns_detected, top_sentence)

        return {
            "keywords": keywords,
            "highlighted_sentences": highlighted_sentences,
            "patterns_detected": patterns_detected,
            "reason": reason
        }

explainer = ExplanationService()
