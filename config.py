from pathlib import Path

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_splits": 5,
    "rf_params": {
        "clf__n_estimators": [100, 200, 500],
        "clf__max_depth": [5, 10, 20, None],
        "clf__min_samples_split": [2, 5, 10],
    },
    "xgb_params": {
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__max_depth": [3, 5, 7],
        "clf__n_estimators": [100, 300, 500],
    },
    "business_impact": {
        "cost_loss_per_customer": 500,
        "cost_action_per_customer": 50,
        "retention_uplift": 0.3,
        "expected_months": 12,
    },
    "paths": {
        "data": Path("data/telco_customer_churn.csv"),
        "model": Path("models/best_model.joblib"),
        "output_dir": Path("outputs"),
    },
}
