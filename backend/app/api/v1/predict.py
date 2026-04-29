from fastapi import APIRouter, HTTPException
from app.models.prediction_schema import PredictionRequest, PredictionResponse
from app.services.ml_service import classifier
from app.services.explainer import explainer

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=PredictionResponse)
def predict_news(request: PredictionRequest):
    try:
        # Predict using Feature 1
        result = classifier.predict(request.text)
        
        # Generate Explanation using Feature 2
        explanation = explainer.generate(request.text, result)
        
        # Clean up metadata before response
        if "_metadata" in result:
            del result["_metadata"]
            
        result["explanation"] = explanation
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
