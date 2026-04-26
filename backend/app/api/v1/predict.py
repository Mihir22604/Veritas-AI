from fastapi import APIRouter, HTTPException
from app.models.prediction_schema import PredictionRequest, PredictionResponse
from app.services.ml_service import classifier

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=PredictionResponse)
def predict_news(request: PredictionRequest):
    try:
        return classifier.predict(request.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
