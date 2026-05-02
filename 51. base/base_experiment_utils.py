from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


SEED = 42
COMET_PROJECT_NAME = "proyecto-1-taa"
COMET_WORKSPACE = "equipo-a"
SPECIAL_COLUMNS = ["EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"]

PIPELINE_DIRS = {
    "P00": "P00_global_standard",
    "P01": "P01_jet_onehot_standard",
    "P02": "P02_jet_onehot_robust",
    "P04": "P04_feature_engineering_robust",
}


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "roadmap.md").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


def read_comet_api_key(project_root: Path | None = None) -> str:
    root = find_project_root(project_root)
    api_key = (root / "keys" / ".comet.config").read_text().strip()
    if not api_key:
        raise ValueError("La API key de Comet esta vacia.")
    return api_key


def load_processed_dataset(pipeline_id: str, project_root: Path | None = None) -> dict[str, object]:
    if pipeline_id not in PIPELINE_DIRS:
        raise ValueError(f"Pipeline no soportado en este bloque: {pipeline_id}")
    root = find_project_root(project_root)
    dataset_dir = root / "data" / "preprocesor" / PIPELINE_DIRS[pipeline_id]
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {dataset_dir}")

    return {
        "pipeline_id": pipeline_id,
        "dataset_dir": dataset_dir,
        "X_train": pd.read_csv(dataset_dir / "X_train_processed.csv").astype(np.float32),
        "X_valid": pd.read_csv(dataset_dir / "X_valid_processed.csv").astype(np.float32),
        "y_train": pd.read_csv(dataset_dir / "y_train.csv")["Label"].astype(np.int8),
        "y_valid": pd.read_csv(dataset_dir / "y_valid.csv")["Label"].astype(np.int8),
        "valid_weight": pd.read_csv(dataset_dir / "valid_weight.csv")["Weight"].astype(np.float64),
    }


def ams_score(signal_weight: np.ndarray | float, background_weight: np.ndarray | float, br: float = 10.0) -> np.ndarray:
    signal_arr = np.asarray(signal_weight, dtype=np.float64)
    background_arr = np.asarray(background_weight, dtype=np.float64)
    radicand = 2.0 * (
        (signal_arr + background_arr + br) * np.log1p(signal_arr / (background_arr + br)) - signal_arr
    )
    return np.sqrt(np.clip(radicand, a_min=0.0, a_max=None))


def find_best_ams_threshold(
    y_true: pd.Series | np.ndarray,
    y_scores: np.ndarray,
    weights: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    y_array = np.asarray(y_true, dtype=np.int8)
    score_array = np.asarray(y_scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)

    order = np.argsort(score_array)[::-1]
    y_sorted = y_array[order]
    score_sorted = score_array[order]
    weight_sorted = weight_array[order]

    signal_cumsum = np.cumsum(np.where(y_sorted == 1, weight_sorted, 0.0))
    background_cumsum = np.cumsum(np.where(y_sorted == 0, weight_sorted, 0.0))
    candidate_mask = np.r_[score_sorted[1:] != score_sorted[:-1], True]
    candidate_indices = np.flatnonzero(candidate_mask)
    ams_values = ams_score(signal_cumsum[candidate_indices], background_cumsum[candidate_indices])

    best_position = int(np.argmax(ams_values))
    best_idx = int(candidate_indices[best_position])
    threshold = float(score_sorted[best_idx])
    y_pred = (score_array >= threshold).astype(np.int8)

    return {
        "best_ams": float(ams_values[best_position]),
        "best_threshold": threshold,
        "signal_weight": float(signal_cumsum[best_idx]),
        "background_weight": float(background_cumsum[best_idx]),
        "predicted_signal_count": int(y_pred.sum()),
        "predicted_background_count": int(len(y_pred) - y_pred.sum()),
        "accuracy_at_ams": float(accuracy_score(y_array, y_pred)),
        "precision_at_ams": float(precision_score(y_array, y_pred, zero_division=0)),
        "recall_at_ams": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1_at_ams": float(f1_score(y_array, y_pred, zero_division=0)),
    }


def predict_scores(model: object, X_valid: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_valid)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_valid)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    raise TypeError(f"El modelo {type(model).__name__} no expone predict_proba ni decision_function.")


def make_experiment(
    *,
    api_key: str,
    block_name: str,
    pipeline_id: str,
    model_name: str,
) -> Experiment:
    experiment = Experiment(
        api_key=api_key,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
        auto_param_logging=False,
        auto_metric_logging=False,
        log_code=False,
        log_graph=False,
        parse_args=False,
    )
    experiment.set_name(f"{block_name}_{pipeline_id}_{model_name}")
    experiment.add_tag(block_name)
    experiment.add_tag(pipeline_id)
    experiment.add_tag(model_name)
    experiment.add_tag("base_models")
    experiment.add_tag("ams")
    return experiment


def run_experiment(
    *,
    block_name: str,
    pipeline_id: str,
    model_name: str,
    model: object,
    model_params: dict[str, object],
    seed: int = SEED,
    log_to_comet: bool = True,
) -> dict[str, object]:
    project_root = find_project_root()
    dataset = load_processed_dataset(pipeline_id, project_root)
    X_train = dataset["X_train"]
    X_valid = dataset["X_valid"]
    y_train = dataset["y_train"]
    y_valid = dataset["y_valid"]
    valid_weight = dataset["valid_weight"]

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    valid_scores = predict_scores(model, X_valid)
    default_pred = (valid_scores >= 0.5).astype(np.int8)
    ams_metrics = find_best_ams_threshold(y_valid, valid_scores, valid_weight)
    metrics = {
        **ams_metrics,
        "roc_auc": float(roc_auc_score(y_valid, valid_scores)),
        "default_accuracy": float(accuracy_score(y_valid, default_pred)),
        "default_precision": float(precision_score(y_valid, default_pred, zero_division=0)),
        "default_recall": float(recall_score(y_valid, default_pred, zero_division=0)),
        "default_f1": float(f1_score(y_valid, default_pred, zero_division=0)),
        "fit_seconds": float(fit_seconds),
    }

    row = {
        "block": block_name,
        "pipeline": pipeline_id,
        "model": model_name,
        "seed": int(seed),
        "uses_processed_dataset_only": True,
        "preprocessing_modified": False,
        "weight_used_as_feature": False,
        "hyperparameters_json": json.dumps(model_params, sort_keys=True),
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        **metrics,
    }

    if log_to_comet:
        experiment = make_experiment(
            api_key=read_comet_api_key(project_root),
            block_name=block_name,
            pipeline_id=pipeline_id,
            model_name=model_name,
        )
        try:
            experiment.log_parameters(model_params)
            experiment.log_parameters(
                {
                    "block": block_name,
                    "pipeline": pipeline_id,
                    "model": model_name,
                    "seed": seed,
                    "uses_processed_dataset_only": True,
                    "preprocessing_modified": False,
                    "weight_used_as_feature": False,
                    "dataset_dir": str(dataset["dataset_dir"]),
                    "feature_count": int(X_train.shape[1]),
                    "train_rows": int(len(y_train)),
                    "valid_rows": int(len(y_valid)),
                }
            )
            experiment.log_metrics(metrics)
            experiment.log_metric("ams", metrics["best_ams"])
            experiment.log_metric("best_ams", metrics["best_ams"])
            experiment.log_metric("ams_threshold", metrics["best_threshold"])
        finally:
            experiment.end()

    return row


def save_comparison(rows: list[dict[str, object]], output_name: str) -> pd.DataFrame:
    project_root = find_project_root()
    results_dir = project_root / "51. base" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows).sort_values("best_ams", ascending=False).reset_index(drop=True)
    comparison.to_csv(results_dir / output_name, index=False)
    return comparison
