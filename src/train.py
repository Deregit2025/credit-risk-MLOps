# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")
from mlflow.tracking import MlflowClient

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv("data/processed/modelling_data.csv")

# ------------------------------
# Features & target
# ------------------------------
target_col = "is_high_risk"
X = data.drop(columns=[target_col, "TransactionId", "CustomerId"])  # drop high-cardinality IDs
y = data[target_col]

# ------------------------------
# Encode remaining categorical features
# ------------------------------
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Fill missing values
X.fillna(0, inplace=True)

# ------------------------------
# Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Scale numeric features
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Models and hyperparameters
# ------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

param_distributions = {
    "LogisticRegression": {
        "C": np.logspace(-3, 3, 7),
        "penalty": ["l1", "l2"]
    },
    "DecisionTree": {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7]
    }
}

# ------------------------------
# MLflow experiment
# ------------------------------
mlflow.set_experiment("bnpl_risk_modeling")
client = MlflowClient()

best_model = None
best_score = 0
best_run_id = None  # <-- capture the run ID of the best model

for name, model in models.items():
    print(f"\nTraining model: {name}")
    with mlflow.start_run(run_name=name) as run:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=5,  # small number to reduce memory usage
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        
        # Predict & evaluate
        y_pred = best_est.predict(X_test)
        y_proba = best_est.predict_proba(X_test)[:, 1] if hasattr(best_est, "predict_proba") else y_pred

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        print(f"Metrics: {metrics}")

        # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_est, artifact_path="model")

        # Keep best model
        if metrics["roc_auc"] > best_score:
            best_model = best_est
            best_score = metrics["roc_auc"]
            best_run_id = run.info.run_id  # capture the run ID

# ------------------------------
# Register best model in MLflow Model Registry
# ------------------------------
# ------------------------------
# Register best model in MLflow Model Registry
# ------------------------------
if best_model is not None:
    model_name = "bnpl_risk_best_model"
    print(f"\nRegistering best model '{best_model.__class__.__name__}' in MLflow Model Registry...")

    # Try to create registered model (ignore if it already exists)
    try:
        client.create_registered_model(model_name)
        print(f"Registered new model '{model_name}'")
    except mlflow.exceptions.MlflowException:
        print(f"Model '{model_name}' already exists, using existing model")

    # Register the current run's model version
    model_uri = f"runs:/{best_run_id}/model"
    client.create_model_version(name=model_name, source=model_uri, run_id=best_run_id)
    print(f"Best model registered as '{model_name}' with version in MLflow.")

