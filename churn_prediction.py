"""
Customer Churn Prediction - End-to-end script
Assumes dataset CSV is at `data/telco_customer_churn.csv`.

Run:
    python churn_prediction.py --data data/telco_customer_churn.csv

Outputs:
- models/best_model.joblib
- outputs/metrics.json
- outputs/roc_curve.png
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import preprocessing as pp
from mlflow_utils import HAS_MLFLOW, init_mlflow, log_artifact, log_metrics, log_model, log_params, start_run, mlflow_is_active

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    },
}

try:
    from xgboost import XGBClassifier  # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    return pp.clean_and_prepare(df)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return pp.feature_engineering(df)


def build_preprocessor(df: pd.DataFrame):
    return pp.build_preprocessor(df)


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    """Return transformed feature names from ColumnTransformer with OneHotEncoder."""
    feature_names = []
    # numeric first
    feature_names.extend(numeric_cols)

    # find the cat transformer
    try:
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat":
                # trans is a Pipeline
                ohe = trans.named_steps.get("ohe")
                if ohe is not None:
                    try:
                        cat_names = ohe.get_feature_names_out(categorical_cols)
                        feature_names.extend(list(cat_names))
                    except Exception:
                        # fallback: use original categorical names
                        feature_names.extend(categorical_cols)
                else:
                    feature_names.extend(categorical_cols)
    except Exception:
        feature_names.extend(categorical_cols)

    return feature_names


def save_prepared_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def prepare_data(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = clean_and_prepare(df)
    df = feature_engineering(df)
    save_prepared_data(df, path)
    return df


def summarize_clv(df: pd.DataFrame) -> Dict[str, float]:
    if "clv" not in df.columns:
        return {}
    return {
        "clv_mean": float(df["clv"].mean()),
        "clv_median": float(df["clv"].median()),
        "clv_std": float(df["clv"].std()),
        "clv_high_risk_mean": float(df.loc[df.get("Churn") == 1, "clv"].mean()) if "Churn" in df.columns else float(df["clv"].mean()),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, numeric_cols, categorical_cols, output_dir: Path):
    results = {}
    models = {}

    # Pipelines
    pipe_lr = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe_rf = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])

    if HAS_XGB:
        pipe_xgb = Pipeline([("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))])

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=CONFIG["cv_splits"], shuffle=True, random_state=CONFIG["random_state"])

    # Hyperparameter distributions
    rf_param_dist = CONFIG["rf_params"]
    xgb_param_dist = CONFIG["xgb_params"]

    # Fit Logistic Regression (baseline)
    pipe_lr.fit(X_train, y_train)
    models["logreg"] = pipe_lr

    # Randomized search for Random Forest
    rsearch_rf = RandomizedSearchCV(pipe_rf, rf_param_dist, n_iter=6, scoring="roc_auc", cv=skf, n_jobs=-1, random_state=CONFIG["random_state"])
    rsearch_rf.fit(X_train, y_train)
    best_rf = rsearch_rf.best_estimator_
    models["rf"] = best_rf
    # record CV mean/std
    rf_mean = rsearch_rf.cv_results_["mean_test_score"][rsearch_rf.best_index_]
    rf_std = rsearch_rf.cv_results_["std_test_score"][rsearch_rf.best_index_]

    # XGBoost tuning if available
    if HAS_XGB:
        rsearch_xgb = RandomizedSearchCV(pipe_xgb, CONFIG["xgb_params"], n_iter=6, scoring="roc_auc", cv=skf, n_jobs=-1, random_state=CONFIG["random_state"])
        rsearch_xgb.fit(X_train, y_train)
        best_xgb = rsearch_xgb.best_estimator_
        models["xgb"] = best_xgb
        xgb_mean = rsearch_xgb.cv_results_["mean_test_score"][rsearch_xgb.best_index_]
        xgb_std = rsearch_xgb.cv_results_["std_test_score"][rsearch_xgb.best_index_]

    # Evaluate models and plot ROC
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        results[name] = {"roc_auc": float(auc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    roc_path = output_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Choose best model by AUC
    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])
    best_model = models[best_name]

    # Save best model
    model_path = output_dir.parent / "models"
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "best_model.joblib"
    joblib.dump(best_model, model_file)

    if mlflow_is_active():
        log_artifact(roc_path)
        log_artifact(metrics_path)
        log_artifact(model_file)

    # Feature importance (if tree-based)
    try:
        feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
        clf = best_model.named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            # Plot top 20 and save
            plt.figure(figsize=(10, 8))
            fi.head(20).plot(kind="barh")
            plt.gca().invert_yaxis()
            plt.title("Top 20 Feature Importances")
            fp = output_dir / "feature_importance.png"
            plt.tight_layout()
            plt.savefig(fp)
        elif hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_).ravel()
            fi = pd.Series(coefs, index=feature_names).sort_values(ascending=False)
            plt.figure(figsize=(10, 8))
            fi.head(20).plot(kind="barh")
            plt.gca().invert_yaxis()
            plt.title("Top 20 Coefficients (abs)")
            fp = output_dir / "feature_importance.png"
            plt.tight_layout()
            plt.savefig(fp)
    except Exception as e:
        logging.warning("Feature importance plot failed: %s", e)

    # Confusion matrix
    try:
        disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
        cm_path = output_dir / "confusion_matrix.png"
        plt.tight_layout()
        disp.figure_.savefig(cm_path)
        plt.close(disp.figure_)
    except Exception as e:
        logging.warning("Confusion matrix save failed: %s", e)

    return results, best_name, best_model


def explain_with_shap(best_model, X_train, numeric_cols, categorical_cols, output_dir: Path):
    if not HAS_SHAP:
        logging.warning("SHAP not available; install shap to get explanations.")
        return

    preproc = best_model.named_steps.get("pre")
    clf = best_model.named_steps.get("clf")

    # Use a manageable sample
    sample = X_train.sample(n=min(200, len(X_train)), random_state=42)
    try:
        X_trans = preproc.transform(sample)
    except Exception:
        X_trans = sample

    def is_tree_model(model):
        return any([
            hasattr(model, "tree_"),
            hasattr(model, "feature_importances_"),
            hasattr(model, "get_booster"),
        ])

    # Use TreeExplainer for tree models, otherwise generic Explainer
    try:
        if is_tree_model(clf):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_trans)
            shap.summary_plot(shap_values, X_trans, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png")
            plt.close()

            probs = best_model.predict_proba(sample)[:, 1]
            idx = np.argmax(probs)
            try:
                shap.plots.waterfall(explainer(X_trans[idx]), show=False)
                plt.tight_layout()
                plt.savefig(output_dir / "shap_waterfall.png")
                plt.close()
            except Exception:
                logging.warning("SHAP waterfall plot failed for sample index %s", idx)

            try:
                feature_names = get_feature_names(preproc, numeric_cols, categorical_cols)
                top_feat = feature_names[0] if feature_names else None
                if top_feat is not None:
                    shap.dependence_plot(top_feat, shap_values, X_trans, show=False)
                    plt.tight_layout()
                    plt.savefig(output_dir / "shap_dependence.png")
                    plt.close()
            except Exception:
                logging.warning("SHAP dependence plot failed")
        else:
            explainer = shap.Explainer(clf, X_trans)
            sv = explainer(X_trans)
            shap.summary_plot(sv, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png")
            plt.close()
    except Exception as e:
        logging.warning("SHAP plotting failed: %s", e)


def business_impact_analysis(best_model, X_test, y_test, results, output_dir: Path):
    cost_loss = CONFIG["business_impact"]["cost_loss_per_customer"]
    cost_action = CONFIG["business_impact"]["cost_action_per_customer"]
    churn_rate = y_test.mean()
    probs = best_model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    expected_retained = tp
    expected_action_cost = (tp + fp) * cost_action
    expected_revenue_saved = expected_retained * cost_loss
    expected_net_savings = expected_revenue_saved - expected_action_cost

    summary = {
        "assumptions": {
            "cost_loss_per_customer": cost_loss,
            "cost_action_per_customer": cost_action,
            "churn_rate": float(churn_rate),
            "decision_threshold": threshold,
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "expected_retained_customers": int(expected_retained),
        "expected_action_cost": float(expected_action_cost),
        "expected_revenue_saved": float(expected_revenue_saved),
        "expected_net_savings": float(expected_net_savings),
    }

    md_lines = [
        "# Business Insights",
        "\n",
        "## Model-driven retention simulation",
        f"- Churn rate in test set: {churn_rate:.2%}",
        f"- Decision threshold: {threshold}",
        f"- Expected retained churn customers: {expected_retained}",
        f"- Expected retention action cost: ${expected_action_cost:,.0f}",
        f"- Expected revenue saved from retention: ${expected_revenue_saved:,.0f}",
        f"- Expected net savings: ${expected_net_savings:,.0f}",
        "\n",
        "## Top churn drivers",
    ]

    try:
        feature_names = get_feature_names(best_model.named_steps["pre"], X_test.select_dtypes(include=["number"]).columns.tolist(), X_test.select_dtypes(include=["object", "category"]).columns.tolist())
        clf = best_model.named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        elif hasattr(clf, "coef_"):
            fi = pd.Series(np.abs(clf.coef_).ravel(), index=feature_names).sort_values(ascending=False)
        else:
            fi = pd.Series(dtype=float)
        for rank, (name, value) in enumerate(fi.head(10).items(), start=1):
            md_lines.append(f"{rank}. {name} — importance {value:.4f}")
    except Exception as e:
        logging.warning("Could not build feature insight section: %s", e)
        md_lines.append("Unable to generate feature importance summary.")

    with open(output_dir / "business_insights.md", "w") as f:
        f.write("\n".join(md_lines))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/telco_customer_churn.csv")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--experiment", type=str, default="customer-churn-experiments")
    parser.add_argument("--register-model", action="store_true", help="Register the best model in MLflow model registry.")
    parser.add_argument("--stage", choices=["all", "preprocess", "train"], default="all")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "preprocess":
        logging.info("Preparing dataset and writing prepared data to outputs/prepared.csv")
        df = load_data(str(data_path))
        prepare_data(df, out_dir / "prepared.csv")
        return

    logging.info("Initializing MLflow experiment %s", args.experiment)
    init_mlflow(experiment_name=args.experiment)

    logging.info("Loading dataset from %s", data_path)
    df = load_data(str(data_path))
    df = clean_and_prepare(df)
    df = feature_engineering(df)

    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a 'Churn' column with Yes/No values or 1/0 target.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    logging.info("Splitting data: test_size=%s", CONFIG["test_size"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["random_state"]
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    with start_run(run_name="churn-training"):
        log_params({
            "test_size": CONFIG["test_size"],
            "random_state": CONFIG["random_state"],
            "cv_splits": CONFIG["cv_splits"],
        })

        results, best_name, best_model = train_and_evaluate(
            X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols, out_dir
        )

        logging.info("Training complete. Best model: %s", best_name)
        logging.info("Results: %s", json.dumps(results, indent=2))

        # SHAP explanation (best-effort)
        try:
            explain_with_shap(best_model, X_train, num_cols, cat_cols, out_dir)
        except Exception as e:
            logging.warning("SHAP explanation failed: %s", e)

        impact = business_impact_analysis(best_model, X_test, y_test, results, out_dir)
        with open(out_dir / "business_impact.json", "w") as f:
            json.dump(impact, f, indent=2)

        if args.register_model:
            log_model(best_model, artifact_path="best_model", registered_model_name="customer-churn")

    logging.info("Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
