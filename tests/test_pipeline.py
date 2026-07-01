import pandas as pd

from churn_prediction import build_preprocessor, clean_and_prepare, feature_engineering


def test_clean_and_prepare_encodes_churn():
    df = pd.DataFrame({
        "customerID": ["0001", "0002"],
        "Churn": ["Yes", "No"],
        "TotalCharges": ["100.0", "200.0"],
    })
    cleaned = clean_and_prepare(df)
    assert cleaned.loc[0, "Churn"] == 1
    assert cleaned.loc[1, "Churn"] == 0
    assert cleaned["TotalCharges"].dtype == float


def test_feature_engineering_adds_fields():
    df = pd.DataFrame({"tenure": [1, 24], "TotalCharges": [50.0, 1200.0], "MonthlyCharges": [50.0, 50.0]})
    engineered = feature_engineering(df)
    assert "tenure_bin" in engineered.columns
    assert "avg_charge_per_month" in engineered.columns


def test_build_preprocessor_returns_columns():
    df = pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [10.0, 20.0], "Contract": ["Month-to-month", "Two year"]})
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    assert "tenure" in num_cols
    assert "Contract" in cat_cols
