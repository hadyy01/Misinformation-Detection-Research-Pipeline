from __future__ import annotations

from pathlib import Path
from typing import Any

from .classical import build_classical_pipeline, predict_scores
from .config import load_config
from .data import load_and_prepare_dataset, split_dataset
from .metrics import evaluate_predictions, save_artifacts
from .neural import train_neural_model
from .transformer import train_transformer_model


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    project_root = (
        resolved_config_path.parent.parent
        if resolved_config_path.parent.name == "configs"
        else resolved_config_path.parent
    )

    df = load_and_prepare_dataset(config, resolved_config_path)
    x_train, x_test, y_train, y_test = split_dataset(df, config)

    model_config = config["model"]
    output_dir = config.get("artifacts", {}).get("output_dir", "artifacts/run")
    resolved_output_dir = (
        Path(output_dir)
        if Path(output_dir).is_absolute()
        else (project_root / output_dir).resolve()
    )

    if model_config["family"] == "classical":
        pipeline = build_classical_pipeline(model_config)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        scores = predict_scores(pipeline, x_test)
    elif model_config["family"] == "neural":
        _, _, predictions, scores = train_neural_model(x_train, y_train, x_test, y_test, config)
    elif model_config["family"] == "transformer":
        _, _, predictions, scores = train_transformer_model(x_train, y_train, x_test, y_test, config)
    else:
        raise ValueError(f"Unsupported model family: {model_config['family']}")

    metrics = evaluate_predictions(y_test, predictions, scores)
    artifact_dir = save_artifacts(resolved_output_dir, config, metrics)

    return {
        "metrics": metrics,
        "artifact_dir": str(artifact_dir),
        "rows": len(df),
    }
