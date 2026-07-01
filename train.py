import logging
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import CONFIG
from preprocessing import build_preprocessor, clean_and_prepare, feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_model_pipelines(preprocessor):
    pipelines = {
        "logreg": Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "rf": Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(class_weight="balanced", random_state=CONFIG["random_state"]))]),
    }
    if CONFIG.get("use_xgb", True):
        from xgboost import XGBClassifier
        pipelines["xgb"] = Pipeline([("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=CONFIG["random_state"]))])
    return pipelines


def resample_with_smote(X, y):
    smote = SMOTE(random_state=CONFIG["random_state"])
    return smote.fit_resample(X, y)


def train_models(df):
    df = clean_and_prepare(df)
    df = feature_engineering(df)

    if "Churn" not in df.columns:
        raise ValueError("Churn column required")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["random_state"]
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    pipelines = get_model_pipelines(preprocessor)

    trained = {}
    metrics = {}
    for name, pipe in pipelines.items():
        if name == "rf":
            search = RandomizedSearchCV(pipe, CONFIG["rf_params"], n_iter=6, scoring="roc_auc", cv=StratifiedKFold(n_splits=CONFIG["cv_splits"], shuffle=True, random_state=CONFIG["random_state"]), n_jobs=-1, random_state=CONFIG["random_state"])
            search.fit(X_train, y_train)
            trained[name] = search.best_estimator_
            logging.info("Best RF params: %s", search.best_params_)
        elif name == "xgb":
            from xgboost import XGBClassifier
            search = RandomizedSearchCV(pipe, CONFIG["xgb_params"], n_iter=6, scoring="roc_auc", cv=StratifiedKFold(n_splits=CONFIG["cv_splits"], shuffle=True, random_state=CONFIG["random_state"]), n_jobs=-1, random_state=CONFIG["random_state"])
            search.fit(X_train, y_train)
            trained[name] = search.best_estimator_
            logging.info("Best XGB params: %s", search.best_params_)
        else:
            pipe.fit(X_train, y_train)
            trained[name] = pipe

    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)

    # Choose best model by ROC-AUC on the held-out test set
    from sklearn.metrics import roc_auc_score
    best_model = None
    best_score = -1.0
    for name, model in trained.items():
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        metrics[name] = auc
        logging.info("Model %s ROC-AUC: %.4f", name, auc)
        if auc > best_score:
            best_score = auc
            best_model = model

    joblib.dump(best_model, model_path / "best_model.joblib")
    logging.info("Saved best model with ROC-AUC %.4f", best_score)
    return trained, X_test, y_test
