import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold, best_f1


def save_metrics(metrics: Dict[str, float], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def generate_business_report(metrics: Dict[str, float], path: Path, clv_summary: Optional[Dict[str, float]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "<html>",
        "<head><title>Model Business Report</title></head>",
        "<body>",
        "<h1>Model Business Report</h1>",
        "<h2>Metrics</h2>",
        "<ul>",
    ]
    for name, value in metrics.items():
        lines.append(f"<li>{name}: {value:.4f}</li>")
    lines.extend(["</ul>", "<h2>CLV Summary</h2>", "<ul>"])
    if clv_summary is not None:
        for name, value in clv_summary.items():
            lines.append(f"<li>{name}: {value}</li>")
    else:
        lines.append("<li>No CLV data available.</li>")
    lines.extend(["</ul>", "</body>", "</html>"])
    with open(path, "w") as f:
        f.write("\n".join(lines))
