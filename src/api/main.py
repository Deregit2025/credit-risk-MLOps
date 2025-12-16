# src/api/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import mlflow.sklearn
import pandas as pd
from pydantic import ValidationError
from src.api.pydantic_models import CustomerData, PredictionResponse
import os

# Initialize FastAPI app
app = FastAPI(
    title="BNPL Risk Prediction API",
    description="API to predict high-risk customers using GradientBoostingClassifier",
    version="1.0.0"
)

# Path to the local MLflow model artifacts
MODEL_PATH = os.path.join(
    "mlruns", "1", "models", "m-cacd6654018946898329870611bbc8e7", "artifacts"
)

# Load the model
try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model from local folder: {e}")

# Exception handler for Pydantic validation errors
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.get("/")
def root():
    return {"message": "BNPL Risk Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        # Convert validated request to DataFrame
        data = pd.DataFrame([customer.model_dump()])  # Pydantic v2 uses model_dump()

        # TODO: Apply same preprocessing as during training if needed

        # Generate prediction
        risk_prob = model.predict_proba(data)[:, 1][0]
        risk_label = int(risk_prob >= 0.5)

        return PredictionResponse(
            risk_label=risk_label,
            risk_probability=risk_prob
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
