from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, description="News text to classify")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
