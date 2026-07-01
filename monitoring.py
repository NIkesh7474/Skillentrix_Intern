import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def summarize_distribution(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        else:
            summary[col] = {
                "top": str(df[col].mode().iloc[0]) if not df[col].mode().empty else "",
                "unique": int(df[col].nunique()),
            }
    return summary


def compare_drift(baseline: Dict[str, Dict[str, float]], current: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    drift_report = {}
    for col, base_stats in baseline.items():
        curr_stats = current.get(col)
        if curr_stats is None:
            drift_report[col] = {"status": "missing_in_current"}
            continue
        if "mean" in base_stats and "mean" in curr_stats:
            delta = abs(base_stats["mean"] - curr_stats["mean"])
            drift_report[col] = {
                "baseline_mean": base_stats["mean"],
                "current_mean": curr_stats["mean"],
                "mean_delta": delta,
                "drifted": delta > 0.1 * (abs(base_stats["mean"]) + 1e-6),
            }
        else:
            drift_report[col] = {
                "baseline_top": base_stats.get("top"),
                "current_top": curr_stats.get("top"),
                "drifted": base_stats.get("top") != curr_stats.get("top"),
            }
    return drift_report


def save_drift_report(report: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def run_drift_check(baseline_path: Path, current_path: Path, output_path: Path):
    baseline = pd.read_csv(baseline_path)
    current = pd.read_csv(current_path)
    baseline_summary = summarize_distribution(baseline)
    current_summary = summarize_distribution(current)
    report = compare_drift(baseline_summary, current_summary)
    save_drift_report(report, output_path)
    return report
