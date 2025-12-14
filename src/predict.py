# src/predict.py

import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow
import mlflow.sklearn

# ------------------------------
# Config
# ------------------------------
DATA_FILE = "data/processed/modelling_data.csv"
OUTPUT_FILE = "data/processed/predictions.csv"
REQUIRED_COLUMNS = ["TransactionId", "CustomerId", "is_high_risk"]  # target + IDs
MODEL_NAME = "bnpl_risk_best_model"

# ------------------------------
# Load new data
# ------------------------------
if not os.path.exists(DATA_FILE):
    print(f"Error: Data file '{DATA_FILE}' not found.")
    sys.exit(1)

data = pd.read_csv(DATA_FILE)

missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
if missing_cols:
    print(f"Error: Missing required columns in data: {missing_cols}")
    sys.exit(1)

X_new = data.drop(columns=REQUIRED_COLUMNS)

# ------------------------------
# Encode categorical features
# ------------------------------
for col in X_new.select_dtypes(include=["object"]).columns:
    X_new[col] = LabelEncoder().fit_transform(X_new[col])

# Fill missing values
X_new.fillna(0, inplace=True)

# Scale numeric features
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

# ------------------------------
# Load best model from MLflow
# ------------------------------
try:
    model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    sys.exit(1)

# ------------------------------
# Make predictions
# ------------------------------
try:
    y_pred = model.predict(X_new)
    y_proba = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else y_pred
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

# ------------------------------
# Save predictions
# ------------------------------
results = data.copy()
results["predicted_risk"] = y_pred
results["predicted_risk_proba"] = y_proba

try:
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"Error saving predictions: {e}")
    sys.exit(1)

print(results[["TransactionId", "CustomerId", "predicted_risk", "predicted_risk_proba"]].head())
