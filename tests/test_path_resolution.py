from pathlib import Path

from batch_predictor import BatchPredictor


def test_batch_predictor_resolves_model_path_from_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    predictor = BatchPredictor()

    assert predictor.model_path.is_absolute()
    assert predictor.model_path.exists()
    assert predictor.model_path.name == "best_model.joblib"
