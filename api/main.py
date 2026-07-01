from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from batch_predictor import BatchPredictor
from preprocessing import clean_and_prepare, feature_engineering
from path_utils import resolve_path

MODEL_PATH = resolve_path("models/best_model.joblib")
app = FastAPI(title="Churn Prediction API")

class PredictRequest(BaseModel):
    data: Dict[str, Any]

@app.on_event("startup")
def load_model():
    app.state.model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    df = pd.DataFrame([request.data])
    df = clean_and_prepare(df)
    df = feature_engineering(df)

    predictor = BatchPredictor()
    predictor.model = app.state.model
    df_aligned = predictor._align_columns(df)

    probability = float(predictor.model.predict_proba(df_aligned)[:, 1][0])
    return {"churn_probability": probability}
