import re
import nltk
from typing import List, Dict, Any, Set

# Precompiled Regex Patterns
CLICKBAIT_REGEX = re.compile(r"(you won't believe|shocking truth|what happens next|doctors hate him|\d+ ways to|must see)", re.IGNORECASE)
URL_REGEX = re.compile(r"https?://|www\.")
DIGIT_REGEX = re.compile(r"\d")

class PatternDetector:
    def __init__(self):
        self.sensational_keywords = {"bombshell", "urgent", "breaking", "shocking", "mind-blowing", "secret"}

    def _extract_entities(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        
        # Track capitalized words in non-starting positions
        internal_caps: Set[str] = set()
        
        # Track sentence starters
        starters: Set[str] = set()
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            if not words:
                continue
                
            # Assume first token is a starter if it's alphabetic
            if words[0].isalpha():
                starters.add(words[0])
                
            for w in words[1:]:
                if w.isalpha() and w[0].isupper():
                    internal_caps.add(w)
                    
        entities = set()
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for i, w in enumerate(words):
                if not w.isalpha() or not w[0].isupper():
                    continue
                # It's a candidate if it's not the first word, OR if it's the first word but also appeared internally
                if i > 0 or w in internal_caps:
                    # Ignore common capitalized stopwords
                    if w.lower() not in {"i", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}:
                        entities.add(w)
                        
        return list(entities)

    def detect(self, text: str) -> Dict[str, Any]:
        patterns_detected = set()
        
        # 1. Clickbait Check
        if CLICKBAIT_REGEX.search(text):
            patterns_detected.add("Clickbait phrasing")
            
        # 2. Sensationalism Check
        words = set(re.findall(r'\w+', text.lower()))
        if self.sensational_keywords.intersection(words):
            patterns_detected.add("Sensationalism")
            
        # 3. Evidence Check
        has_digits = bool(DIGIT_REGEX.search(text))
        has_links = bool(URL_REGEX.search(text))
        
        entities = self._extract_entities(text)
        
        if not has_digits and not has_links and len(entities) < 2:
            patterns_detected.add("Lack of verifiable evidence (numbers or links)")
            
        return {
            "patterns_detected": list(patterns_detected),
            "evidence": {
                "has_digits": has_digits,
                "has_links": has_links,
                "entities_found": entities
            }
        }
