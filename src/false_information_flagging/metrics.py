from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_predictions(y_true, y_pred, y_score) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "classification_report": classification_report(y_true, y_pred, target_names=["Fake", "True"], output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def save_artifacts(output_dir: str | Path, config: dict[str, Any], metrics: dict[str, Any]) -> Path:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifact_dir / "metrics.json"
    report_path = artifact_dir / "classification_report.json"
    confusion_path = artifact_dir / "confusion_matrix.csv"
    summary_path = artifact_dir / "summary.md"
    config_path = artifact_dir / "resolved_config.json"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {key: value for key, value in metrics.items() if key != "classification_report"},
            handle,
            indent=2,
        )

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics["classification_report"], handle, indent=2)

    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["actual_fake", "actual_true"],
        columns=["pred_fake", "pred_true"],
    )
    matrix.to_csv(confusion_path)

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summary = "\n".join(
        [
            "# Experiment Summary",
            "",
            f"- Model family: `{config['model']['family']}`",
            f"- Model name: `{config['model']['name']}`",
            f"- Accuracy: `{metrics['accuracy']:.4f}`",
            f"- Precision: `{metrics['precision']:.4f}`",
            f"- Recall: `{metrics['recall']:.4f}`",
            f"- F1: `{metrics['f1']:.4f}`",
            f"- ROC-AUC: `{metrics['roc_auc']:.4f}`",
            "",
            "Use these artifacts as the source of truth for experiment comparisons.",
        ]
    )
    summary_path.write_text(summary, encoding="utf-8")

    return artifact_dir

