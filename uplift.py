import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlflow_utils import init_mlflow, log_metrics, log_params, log_model, start_run
from preprocessing import build_preprocessor, clean_and_prepare, feature_engineering


class TwoModelUplift:
    def __init__(self, base_estimator: Optional[Any] = None):
        self.base_estimator = base_estimator or RandomForestClassifier(n_estimators=200, random_state=42)
        self.model_treated = clone(self.base_estimator)
        self.model_control = clone(self.base_estimator)
        self.preprocessor = None

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series):
        if self.preprocessor is None:
            self.preprocessor, _, _ = build_preprocessor(X)
        X_transformed = self.preprocessor.fit_transform(X)
        self.model_treated.fit(X_transformed[treatment == 1], outcome[treatment == 1])
        self.model_control.fit(X_transformed[treatment == 0], outcome[treatment == 0])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("Model must be fit before predicting uplift.")
        X_transformed = self.preprocessor.transform(X)
        proba_treatment = self.model_treated.predict_proba(X_transformed)[:, 1]
        proba_control = self.model_control.predict_proba(X_transformed)[:, 1]
        return proba_treatment - proba_control


def load_uplift_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_and_prepare(df)
    df = feature_engineering(df)
    if "treatment" not in df.columns or "Churn" not in df.columns:
        raise ValueError("Uplift dataset must contain 'treatment' and 'Churn' columns.")
    return df


def train_uplift_model(data_path: str, output_dir: Path, experiment_name: str = "uplift-experiments") -> Tuple[TwoModelUplift, Dict[str, float]]:
    df = load_uplift_dataset(data_path)
    X = df.drop(columns=["Churn", "treatment"])
    treatment = df["treatment"].astype(int)
    y = df["Churn"].astype(int)

    X_train, X_test, treat_train, treat_test, y_train, y_test = train_test_split(
        X, treatment, y, test_size=0.2, stratify=treatment, random_state=42
    )

    model = TwoModelUplift()
    model.fit(X_train, treat_train, y_train)

    uplift_scores = model.predict_uplift(X_test)
    metrics = {
        "uplift_mean": float(np.mean(uplift_scores)),
        "uplift_std": float(np.std(uplift_scores)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "uplift_model.joblib"
    log_params({"data_path": data_path})
    log_metrics(metrics)
    log_model(model, artifact_path="uplift_model", registered_model_name="customer-churn-uplift")
    return model, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train an uplift model when treatment labels are available.")
    parser.add_argument("--data", type=str, required=True, help="Input CSV with treatment and churn columns.")
    parser.add_argument("--output", type=str, default="models/uplift")
    args = parser.parse_args()

    init_mlflow(experiment_name="uplift-experiments")
    with start_run(run_name="uplift-training"):
        model, metrics = train_uplift_model(args.data, Path(args.output))
        print("Uplift training finished", metrics)


if __name__ == "__main__":
    main()
