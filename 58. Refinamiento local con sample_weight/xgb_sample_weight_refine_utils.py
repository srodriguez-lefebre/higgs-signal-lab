from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEED = 42
N_SPLITS = 5
COMET_PROJECT_NAME = "proyecto-1-taa"
COMET_WORKSPACE = "equipo-a"
PIPELINE_ID = "P04"
PIPELINE_DIR = "P04_feature_engineering_robust"

BASE_WEIGHTED_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": SEED,
    "tree_method": "hist",
    "n_jobs": -1,
    "missing": np.nan,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": 1.5346,
    "gamma": 0.0,
    "max_delta_step": 0,
}

REFINE_CONFIGS = [
    {"config_id": "SW00", "reg_alpha": 0.5, "reg_lambda": 1.0},
    {"config_id": "SW01", "reg_alpha": 0.3, "reg_lambda": 1.0},
    {"config_id": "SW02", "reg_alpha": 0.7, "reg_lambda": 1.0},
    {"config_id": "SW03", "reg_alpha": 0.5, "reg_lambda": 0.5},
    {"config_id": "SW04", "reg_alpha": 0.5, "reg_lambda": 2.0},
    {"config_id": "SW05", "reg_alpha": 0.3, "reg_lambda": 2.0},
]


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


def load_p04_full_dataset(project_root: Path | None = None) -> dict[str, object]:
    root = find_project_root(project_root)
    dataset_dir = root / "data" / "preprocesor" / PIPELINE_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {dataset_dir}")

    X_train = pd.read_csv(dataset_dir / "X_train_processed.csv").astype(np.float32)
    X_valid = pd.read_csv(dataset_dir / "X_valid_processed.csv").astype(np.float32)
    y_train = pd.read_csv(dataset_dir / "y_train.csv")["Label"].astype(np.int8)
    y_valid = pd.read_csv(dataset_dir / "y_valid.csv")["Label"].astype(np.int8)
    w_train = pd.read_csv(dataset_dir / "train_weight.csv")["Weight"].astype(np.float64)
    w_valid = pd.read_csv(dataset_dir / "valid_weight.csv")["Weight"].astype(np.float64)

    forbidden = {"Weight", "Label", "EventId", "KaggleSet", "KaggleWeight"}
    leaked = sorted(forbidden.intersection(X_train.columns).union(forbidden.intersection(X_valid.columns)))
    if leaked:
        raise ValueError(f"Columnas prohibidas dentro de X procesado: {leaked}")

    return {
        "pipeline_id": PIPELINE_ID,
        "dataset_dir": dataset_dir,
        "X": pd.concat([X_train, X_valid], ignore_index=True),
        "y": pd.concat([y_train, y_valid], ignore_index=True),
        "weight": pd.concat([w_train, w_valid], ignore_index=True),
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
        "signal_rate": float(y_pred.mean()),
        "accuracy_at_ams": float(accuracy_score(y_array, y_pred)),
        "precision_at_ams": float(precision_score(y_array, y_pred, zero_division=0)),
        "recall_at_ams": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1_at_ams": float(f1_score(y_array, y_pred, zero_division=0)),
    }


def make_experiment(
    *,
    api_key: str,
    folder_name: str,
    config_id: str,
    fold: int,
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
    experiment.set_name(f"{folder_name}_{PIPELINE_ID}_XGBClassifier_sample_weight_{config_id}_fold_{fold}")
    experiment.add_tag(folder_name)
    experiment.add_tag(PIPELINE_ID)
    experiment.add_tag("XGBClassifier")
    experiment.add_tag(config_id)
    experiment.add_tag("sample_weight")
    experiment.add_tag("reg_alpha_reg_lambda_refine")
    experiment.add_tag("cross_validation")
    experiment.add_tag("ams")
    return experiment


def summarize_folds(fold_df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "fold_count": int(len(fold_df)),
        "ams_mean": float(fold_df["best_ams"].mean()),
        "ams_std": float(fold_df["best_ams"].std(ddof=1)),
        "auc_mean": float(fold_df["roc_auc"].mean()),
        "auc_std": float(fold_df["roc_auc"].std(ddof=1)),
        "threshold_mean": float(fold_df["best_threshold"].mean()),
        "threshold_median": float(fold_df["best_threshold"].median()),
        "threshold_std": float(fold_df["best_threshold"].std(ddof=1)),
        "fit_seconds_mean": float(fold_df["fit_seconds"].mean()),
        "signal_rate_mean": float(fold_df["signal_rate"].mean()),
        "precision_at_ams_mean": float(fold_df["precision_at_ams"].mean()),
        "recall_at_ams_mean": float(fold_df["recall_at_ams"].mean()),
        "f1_at_ams_mean": float(fold_df["f1_at_ams"].mean()),
    }


def build_model_params(config: dict[str, object]) -> dict[str, object]:
    params = dict(BASE_WEIGHTED_PARAMS)
    params["reg_alpha"] = config["reg_alpha"]
    params["reg_lambda"] = config["reg_lambda"]
    return params


def run_sample_weight_refinement(
    *,
    folder_name: str = "58. Refinamiento local con sample_weight",
    configs: list[dict[str, object]] | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    log_to_comet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = configs or REFINE_CONFIGS
    project_root = find_project_root()
    results_dir = project_root / folder_name / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_p04_full_dataset(project_root)
    X = dataset["X"]
    y = dataset["y"]
    weights = dataset["weight"]
    api_key = read_comet_api_key(project_root) if log_to_comet else ""

    all_fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_indices = list(cv.split(X, y))

    for config in configs:
        config_id = str(config["config_id"])
        model_params = build_model_params(config)
        config_fold_rows: list[dict[str, object]] = []
        for fold, (train_idx, valid_idx) in enumerate(split_indices, start=1):
            model = XGBClassifier(**model_params)
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_valid = X.iloc[valid_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_valid = y.iloc[valid_idx].reset_index(drop=True)
            w_train = weights.iloc[train_idx].reset_index(drop=True)
            w_valid = weights.iloc[valid_idx].reset_index(drop=True)

            start = time.perf_counter()
            model.fit(X_train, y_train, sample_weight=w_train)
            fit_seconds = time.perf_counter() - start

            class_order = model.classes_.tolist()
            if class_order != [0, 1]:
                raise ValueError(f"Orden de clases inesperado para predict_proba: {class_order}")
            valid_scores = model.predict_proba(X_valid)[:, 1]
            ams_metrics = find_best_ams_threshold(y_valid, valid_scores, w_valid)
            metrics = {
                **ams_metrics,
                "roc_auc": float(roc_auc_score(y_valid, valid_scores)),
                "fit_seconds": float(fit_seconds),
            }
            row = {
                "pipeline": PIPELINE_ID,
                "model": "XGBClassifier",
                "config_id": config_id,
                "fold": int(fold),
                "seed": int(seed),
                "uses_processed_dataset_only": True,
                "preprocessing_modified": False,
                "weight_used_as_feature": False,
                "sample_weight_used": True,
                "predict_proba_signal_column": 1,
                "class_order": json.dumps(class_order),
                "hyperparameters_json": json.dumps(model_params, sort_keys=True),
                "learning_rate": float(model_params["learning_rate"]),
                "n_estimators": int(model_params["n_estimators"]),
                "max_depth": int(model_params["max_depth"]),
                "min_child_weight": int(model_params["min_child_weight"]),
                "subsample": float(model_params["subsample"]),
                "colsample_bytree": float(model_params["colsample_bytree"]),
                "reg_alpha": float(config["reg_alpha"]),
                "reg_lambda": float(config["reg_lambda"]),
                "feature_count": int(X.shape[1]),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                **metrics,
            }
            config_fold_rows.append(row)
            all_fold_rows.append(row)

            if log_to_comet:
                experiment = make_experiment(api_key=api_key, folder_name=folder_name, config_id=config_id, fold=fold)
                try:
                    experiment.log_parameters(model_params)
                    experiment.log_parameters(
                        {
                            "pipeline": PIPELINE_ID,
                            "model": "XGBClassifier",
                            "config_id": config_id,
                            "fold": fold,
                            "seed": seed,
                            "n_splits": n_splits,
                            "uses_processed_dataset_only": True,
                            "preprocessing_modified": False,
                            "weight_used_as_feature": False,
                            "sample_weight_used": True,
                            "predict_proba_signal_column": 1,
                            "class_order": json.dumps(class_order),
                            "dataset_dir": str(dataset["dataset_dir"]),
                            "feature_count": int(X.shape[1]),
                            "train_rows": int(len(train_idx)),
                            "valid_rows": int(len(valid_idx)),
                        }
                    )
                    experiment.log_metrics(metrics)
                    experiment.log_metric("ams", metrics["best_ams"])
                    experiment.log_metric("best_ams", metrics["best_ams"])
                    experiment.log_metric("ams_threshold", metrics["best_threshold"])
                finally:
                    experiment.end()

            print(
                f"{PIPELINE_ID} {config_id} fold={fold} "
                f"AMS={metrics['best_ams']:.6f} AUC={metrics['roc_auc']:.6f} "
                f"threshold={metrics['best_threshold']:.6f} signal_rate={metrics['signal_rate']:.4f} "
                f"fit_seconds={fit_seconds:.2f}"
            )

        config_fold_df = pd.DataFrame(config_fold_rows)
        summary_rows.append(
            {
                "pipeline": PIPELINE_ID,
                "model": "XGBClassifier",
                "config_id": config_id,
                "sample_weight_used": True,
                "learning_rate": float(BASE_WEIGHTED_PARAMS["learning_rate"]),
                "n_estimators": int(BASE_WEIGHTED_PARAMS["n_estimators"]),
                "max_depth": int(BASE_WEIGHTED_PARAMS["max_depth"]),
                "min_child_weight": int(BASE_WEIGHTED_PARAMS["min_child_weight"]),
                "subsample": float(BASE_WEIGHTED_PARAMS["subsample"]),
                "colsample_bytree": float(BASE_WEIGHTED_PARAMS["colsample_bytree"]),
                "reg_alpha": float(config["reg_alpha"]),
                "reg_lambda": float(config["reg_lambda"]),
                "seed": int(seed),
                **summarize_folds(config_fold_df),
            }
        )

    fold_df = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("ams_mean", ascending=False).reset_index(drop=True)
    fold_df.to_csv(results_dir / "p04_xgboost_sample_weight_refine_fold_metrics.csv", index=False)
    summary_df.to_csv(results_dir / "p04_xgboost_sample_weight_refine_summary.csv", index=False)
    return fold_df, summary_df
