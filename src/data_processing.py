# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import category_encoders as ce  # WoE encoding
from sklearn.preprocessing import OneHotEncoder

# -------------------------------
# Step 0: Helper Transformers
# -------------------------------

class TransactionTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract hour, day, month, year from a datetime column.
    """
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['transaction_hour'] = df[self.datetime_col].dt.hour
        df['transaction_day'] = df[self.datetime_col].dt.day
        df['transaction_month'] = df[self.datetime_col].dt.month
        df['transaction_year'] = df[self.datetime_col].dt.year
        return df

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate transaction features per customer.
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_agg = X.groupby(self.customer_id_col).agg(
            total_amount=(self.amount_col, 'sum'),
            avg_amount=(self.amount_col, 'mean'),
            std_amount=(self.amount_col, 'std'),
            transaction_count=(self.amount_col, 'count')
        ).reset_index()
        return df_agg

# -------------------------------
# Step 1: Column lists
# -------------------------------
numerical_features = ['Amount', 'Value', 'PricingStrategy']
categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId', 'ProductId']

# -------------------------------
# Step 2: Pipelines for preprocessing
# -------------------------------

# Numerical pipeline: impute + scale
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute + WoE + OHE
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('woe', ce.WOEEncoder(cols=categorical_features, return_df=True)),
     ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])

# -------------------------------
# Step 3: Full pipeline
# -------------------------------
full_pipeline = Pipeline([
    ('time_features', TransactionTimeFeatures(datetime_col='TransactionStartTime')),
    ('preprocessor', preprocessor),
    ('aggregator', CustomerAggregator(customer_id_col='CustomerId', amount_col='Amount'))
])

# -------------------------------
# Step 4: Main processing function
# -------------------------------
def process_data(input_path: str, output_path: str = None):
    # Read raw CSV
    df = pd.read_csv(input_path, parse_dates=['TransactionStartTime'])

    # ---------------------------
    # Step 1: WoE encoding
    # ---------------------------
    woe_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'ProductId']
    target_col = 'FraudResult'
    woe_encoder = ce.WOEEncoder(cols=woe_cols)
    df[woe_cols] = woe_encoder.fit_transform(df[woe_cols], df[target_col])

    # ---------------------------
    # Step 2: Time Features
    # ---------------------------
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year

    # ---------------------------
    # Step 3: Aggregate Customer Features
    # ---------------------------
    df_agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        std_amount=('Amount', 'std'),
        transaction_count=('Amount', 'count')
    ).reset_index()

    # Merge aggregated features back with customer-level features
    df_final = df.merge(df_agg, on='CustomerId', how='left')

    # ---------------------------
    # Step 4: One-Hot Encoding categorical features
    # ---------------------------
    from sklearn.preprocessing import OneHotEncoder
    ohe_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'ProductId']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_array = ohe.fit_transform(df_final[ohe_cols])
    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(ohe_cols))
    df_final = pd.concat([df_final.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
    df_final.drop(columns=ohe_cols, inplace=True)

    # ---------------------------
    # Step 5: Handle Missing Values
    # ---------------------------
    df_final.fillna(0, inplace=True)

    # ---------------------------
    # Step 6: Standardize numerical features
    # ---------------------------
    from sklearn.preprocessing import StandardScaler
    num_cols = ['Amount', 'Value', 'PricingStrategy', 'total_amount', 'avg_amount', 'std_amount', 'transaction_count']
    scaler = StandardScaler()
    df_final[num_cols] = scaler.fit_transform(df_final[num_cols])

    # Save processed CSV
    if output_path:
        df_final.to_csv(output_path, index=False)

    print("Processing complete. Shape:", df_final.shape)
    return df_final


# -------------------------------
# Step 5: Train-test split
# -------------------------------
def split_data(df, target='FraudResult', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
if __name__ == "__main__":
    import os

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, "data", "raw", "data.csv")
    processed_data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")

    # Process data
    print("Processing raw data...")
    df_processed = process_data(input_path=raw_data_path, output_path=processed_data_path)
    
    print("Processed data shape:", df_processed.shape)
    print("Sample processed data:")
    print(df_processed.head())
