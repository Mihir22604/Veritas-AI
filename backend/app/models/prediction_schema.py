from pydantic import BaseModel, Field
from typing import Optional
from app.models.explain_schema import ExplanationResponse


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, description="News text to classify")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    explanation: Optional[ExplanationResponse] = None
