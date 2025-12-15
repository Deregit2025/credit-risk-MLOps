# src/api/main.py
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from src.api.pydantic_models import CustomerData, PredictionResponse

# Initialize FastAPI app
app = FastAPI(
    title="BNPL Risk Prediction API",
    description="API to predict high-risk customers using GradientBoostingClassifier",
    version="1.0.0"
)

# Load best model from MLflow
MODEL_NAME = "bnpl_risk_best_model"
MODEL_VERSION = 1  # Use latest version if needed

try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def root():
    return {"message": "BNPL Risk Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([customer.dict()])

        # If any preprocessing is required, apply it here
        # Example: encoding categorical features, scaling numeric ones
        # TODO: Apply the same preprocessing used during training

        # Generate prediction
        risk_prob = model.predict_proba(data)[:, 1][0]
        risk_label = int(risk_prob >= 0.5)

        return PredictionResponse(
            risk_label=risk_label,
            risk_probability=risk_prob
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
