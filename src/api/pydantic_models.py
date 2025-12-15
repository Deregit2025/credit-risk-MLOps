# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: float
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    ProductId: str
    # Add any other features your model requires

class PredictionResponse(BaseModel):
    risk_label: int
    risk_probability: float
