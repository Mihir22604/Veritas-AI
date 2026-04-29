from pydantic import BaseModel, Field
from typing import List

class ExplanationResponse(BaseModel):
    keywords: List[str] = Field(description="Top influencing words aligning with the prediction")
    highlighted_sentences: List[str] = Field(description="Top 1-2 most suspicious/influential sentences")
    patterns_detected: List[str] = Field(description="Linguistic or structural patterns detected")
    reason: str = Field(description="Human-readable explanation of the prediction")
