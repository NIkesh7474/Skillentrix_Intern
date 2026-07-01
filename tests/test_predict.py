import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predict import MODEL_PATH, load_customer


def test_load_customer_json(tmp_path):
    data = {"tenure": 10, "MonthlyCharges": 70}
    file_path = tmp_path / "customer.json"
    file_path.write_text(json.dumps(data))

    loaded = load_customer(file_path)
    assert loaded.loc[0, "tenure"] == 10
    assert loaded.loc[0, "MonthlyCharges"] == 70


def test_predict_cli_with_saved_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best_model.joblib"

    X = pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [10, 20]})
    y = [0, 1]
    clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
    clf.fit(X, y)
    joblib.dump(clf, model_path)

    monkeypatch.setattr("predict.MODEL_PATH", model_path)
    data = {"tenure": 1, "MonthlyCharges": 10}
    file_path = tmp_path / "customer.json"
    file_path.write_text(json.dumps(data))

    loaded = load_customer(file_path)
    assert loaded.loc[0, "tenure"] == 1
    assert loaded.loc[0, "MonthlyCharges"] == 10
