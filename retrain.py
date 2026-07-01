import argparse
import logging
from pathlib import Path

import pandas as pd

from churn_prediction import (build_preprocessor, clean_and_prepare, feature_engineering,
                              load_data, train_and_evaluate)
from monitoring import population_stability_index, run_drift_check
from mlflow_utils import init_mlflow, log_metrics, log_params, start_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def combine_datasets(base_path: str, new_path: str) -> pd.DataFrame:
    base = load_data(base_path)
    new = load_data(new_path)
    combined = pd.concat([base, new], ignore_index=True)
    return combined


def should_retrain(baseline_path: str, candidate_path: str, threshold: float = 0.1) -> bool:
    report = run_drift_check(Path(baseline_path), Path(candidate_path), Path("outputs/drift_report.json"))
    psi_values = [v.get("psi", 0.0) for v in report.values() if isinstance(v, dict)]
    max_psi = max(psi_values) if psi_values else 0.0
    logging.info("Maximum PSI detected: %s", max_psi)
    return max_psi > threshold


def main():
    parser = argparse.ArgumentParser(description="Automated retraining pipeline for new customer data.")
    parser.add_argument("--baseline", type=str, default="data/telco_customer_churn.csv")
    parser.add_argument("--new", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/retrain")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    init_mlflow()
    if should_retrain(args.baseline, args.new, args.threshold):
        logging.info("Data drift threshold exceeded. Retraining model...")
        combined = combine_datasets(args.baseline, args.new)
        combined = clean_and_prepare(combined)
        combined = feature_engineering(combined)
        if "Churn" not in combined.columns:
            raise ValueError("Combined dataset must include a Churn target column.")

        X = combined.drop(columns=["Churn"])
        y = combined["Churn"].astype(int)
        X_train, X_test, y_train, y_test = X, pd.DataFrame(), y, pd.Series()
        preprocessor, num_cols, cat_cols = build_preprocessor(X)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        with start_run(run_name="automated-retrain"):
            results, best_name, best_model = train_and_evaluate(X, X, y, y, preprocessor, num_cols, cat_cols, output_dir)
            logging.info("Retraining complete. Best model: %s", best_name)
            log_params({"baseline": args.baseline, "new_data": args.new, "threshold": args.threshold})
            log_metrics({f"retrain_{k}": v for k, v in results.get(best_name, {}).items()})
    else:
        logging.info("Data drift below threshold; no retraining triggered.")


if __name__ == "__main__":
    main()
