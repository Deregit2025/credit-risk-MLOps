# src/api/pydantic_models.py
from pydantic import BaseModel, Field, field_validator
from typing import Annotated

class CustomerData(BaseModel):
    Amount: Annotated[float, Field(gt=0, description="Transaction amount must be positive")]
    Value: Annotated[float, Field(ge=0, description="Value must be zero or positive")]
    PricingStrategy: Annotated[float, Field(ge=0, description="Pricing strategy value must be non-negative")]
    ProductCategory: Annotated[str, Field(min_length=1, description="Product category cannot be empty")]
    ChannelId: Annotated[str, Field(min_length=1, description="Channel ID cannot be empty")]
    ProviderId: Annotated[str, Field(min_length=1, description="Provider ID cannot be empty")]
    ProductId: Annotated[str, Field(min_length=1, description="Product ID cannot be empty")]

    # Example of a custom validator for string fields
    @field_validator("ProductCategory", "ChannelId", "ProviderId", "ProductId")
    def non_empty_string(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v

class PredictionResponse(BaseModel):
    risk_label: Annotated[int, Field(ge=0, le=1, description="Risk label: 0 for low risk, 1 for high risk")]
    risk_probability: Annotated[float, Field(ge=0, le=1, description="Risk probability between 0 and 1")]
