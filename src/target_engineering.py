# src/target_engineering.py

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
RAW_DATA_PATH = PROJECT_ROOT / os.getenv("RAW_DATA_PATH")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR")
# ✅ Define the actual processed CSV file path
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "modelling_data.csv"

# -------------------------------
# Step 1: RFM Calculation
# -------------------------------
def calculate_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum"),
    ).reset_index()

    return rfm

# -------------------------------
# Step 2: Clustering
# -------------------------------
def cluster_customers(rfm: pd.DataFrame, n_clusters=3) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm

# -------------------------------
# Step 3: High-Risk Labeling
# -------------------------------
def assign_high_risk(rfm: pd.DataFrame) -> pd.DataFrame:
    cluster_profile = rfm.groupby("cluster")[["frequency", "monetary"]].mean()

    high_risk_cluster = cluster_profile.sum(axis=1).idxmin()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm

# -------------------------------
# Step 4: Pipeline Execution
# -------------------------------
def create_proxy_target():
    print("🔹 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["TransactionStartTime"])

    print("🔹 Calculating RFM metrics...")
    rfm = calculate_rfm(df)

    print("🔹 Clustering customers...")
    rfm = cluster_customers(rfm)

    print("🔹 Assigning high-risk labels...")
    rfm = assign_high_risk(rfm)

    print("🔹 Merging target into dataset...")
    final_df = df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    final_df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"✅ Modeling dataset saved to: {PROCESSED_DATA_DIR}")

    return final_df


# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    create_proxy_target()
