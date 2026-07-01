import joblib
from pathlib import Path


class PredictionPipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans)

    def predict(self, X):
        X_trans = self.preprocessor.transform(X)
        return self.model.predict(X_trans)


def save_pipeline(pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path):
    return joblib.load(path)
