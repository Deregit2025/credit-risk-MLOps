import pandas as pd
from src.data_processing import (
    TransactionTimeFeatures,
    CustomerAggregator
)

# -------------------------------
# Test 1: TransactionTimeFeatures
# -------------------------------
def test_transaction_time_features_creates_columns():
    df = pd.DataFrame({
        "TransactionStartTime": pd.to_datetime([
            "2024-01-01 10:15:00",
            "2024-01-02 14:30:00"
        ])
    })

    transformer = TransactionTimeFeatures(
        datetime_col="TransactionStartTime"
    )
    transformed = transformer.transform(df)

    expected_columns = {
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year"
    }

    assert expected_columns.issubset(transformed.columns)


# -------------------------------
# Test 2: CustomerAggregator
# -------------------------------
def test_customer_aggregator_outputs_correct_shape_and_columns():
    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100, 200, 300]
    })

    aggregator = CustomerAggregator(
        customer_id_col="CustomerId",
        amount_col="Amount"
    )

    aggregated = aggregator.transform(df)

    expected_columns = {
        "CustomerId",
        "total_amount",
        "avg_amount",
        "std_amount",
        "transaction_count"
    }

    # Column check
    assert expected_columns.issubset(aggregated.columns)

    # One row per unique customer
    assert len(aggregated) == df["CustomerId"].nunique()
