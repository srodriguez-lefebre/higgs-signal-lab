from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split
from xgboost import XGBClassifier


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = PROJECT_ROOT / "20.Arbol_i3"
RESULTS_DIR = STAGE_DIR / "results"
MODELS_DIR = STAGE_DIR / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "higgs-boson" / "processed" / "i3_arboles"
COMET_KEY_PATH = PROJECT_ROOT / "keys" / ".comet.config"
COMET_PROJECT_NAME = "Primer Proyecto TAA"
COMET_WORKSPACE = "equipo-a"
SEARCH_NAME = "xgboost_i3_arboles_comet_search"
RESULTS_PATH = RESULTS_DIR / "01_search_results.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_xgboost_i3_arboles.joblib"
BEST_PARAMS_PATH = RESULTS_DIR / "01_best_params.json"

BASE_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": SEED,
    "tree_method": "hist",
    "n_jobs": -1,
    "missing": np.nan,
}

BASELINE_TRIAL: dict[str, object] = {
    "learning_rate": 0.05,
    "n_estimators": 400,
    "max_depth": 6,
    "scale_pos_weight": 1.5346,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "max_delta_step": 0,
}

PARAM_DISTRIBUTIONS: dict[str, list[object]] = {
    "learning_rate": [0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.1],
    "n_estimators": [250, 320, 400, 500, 650, 800],
    "max_depth": [4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 3, 5, 8],
    "gamma": [0.0, 0.05, 0.1, 0.2, 0.4],
    "subsample": [0.75, 0.8, 0.85, 0.9, 0.95],
    "colsample_bytree": [0.75, 0.8, 0.85, 0.9, 0.95],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.3],
    "scale_pos_weight": [1.0, 1.25, 1.5346, 1.75, 2.0],
    "max_delta_step": [0, 1],
}

PARAM_COLUMNS = [
    "learning_rate",
    "n_estimators",
    "max_depth",
    "scale_pos_weight",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_lambda",
    "reg_alpha",
    "max_delta_step",
]

RESULT_COLUMNS = [
    "trial_id",
    "dataset",
    "params_json",
    *PARAM_COLUMNS,
    "best_ams",
    "best_threshold",
    "signal_weight",
    "background_weight",
    "predicted_signal_count",
    "predicted_background_count",
    "accuracy_at_ams",
    "precision_at_ams",
    "recall_at_ams",
    "f1_at_ams",
    "roc_auc",
    "default_accuracy",
    "default_precision",
    "default_recall",
    "default_f1",
    "fit_seconds",
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def read_comet_api_key() -> str:
    api_key = COMET_KEY_PATH.read_text().strip()
    if not api_key:
        raise ValueError(f"La API key de Comet esta vacia: {COMET_KEY_PATH}")
    return api_key


def load_i3_arboles_dataset() -> dict[str, object]:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {PROCESSED_DIR}")

    label_map = {"b": 0, "s": 1}
    X_full = pd.read_csv(PROCESSED_DIR / "X_train_processed.csv").astype(np.float32)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_processed.csv").astype(np.float32)
    y_full = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Label"].map(label_map).astype(np.int8)
    weight_full = pd.read_csv(PROCESSED_DIR / "train_weight.csv")["Weight"].astype(np.float64)
    train_event_id = pd.read_csv(PROCESSED_DIR / "train_event_id.csv")["EventId"].astype(np.int64)
    test_event_id = pd.read_csv(PROCESSED_DIR / "test_event_id.csv")["EventId"].astype(np.int64)

    train_idx, valid_idx = train_test_split(
        np.arange(len(X_full)),
        test_size=0.1,
        random_state=SEED,
        stratify=y_full,
    )

    return {
        "X_train": X_full.iloc[train_idx].reset_index(drop=True),
        "X_valid": X_full.iloc[valid_idx].reset_index(drop=True),
        "X_test": X_test,
        "y_train": y_full.iloc[train_idx].reset_index(drop=True),
        "y_valid": y_full.iloc[valid_idx].reset_index(drop=True),
        "train_weight": weight_full.iloc[train_idx].reset_index(drop=True),
        "valid_weight": weight_full.iloc[valid_idx].reset_index(drop=True),
        "train_event_id": train_event_id.iloc[train_idx].reset_index(drop=True),
        "valid_event_id": train_event_id.iloc[valid_idx].reset_index(drop=True),
        "test_event_id": test_event_id,
        "feature_count": int(X_full.shape[1]),
        "full_train_rows": int(len(X_full)),
        "test_rows": int(len(X_test)),
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


def build_trials(n_iter: int) -> list[dict[str, object]]:
    sampled = list(
        ParameterSampler(
            PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            random_state=SEED,
        )
    )
    return [dict(BASELINE_TRIAL), *[dict(trial) for trial in sampled]]


def load_completed_trial_ids() -> set[int]:
    if not RESULTS_PATH.exists():
        return set()
    results = pd.read_csv(RESULTS_PATH)
    if "trial_id" not in results.columns:
        return set()
    return {int(value) for value in results["trial_id"].dropna().tolist()}


def append_result(row: dict[str, object]) -> None:
    result_df = pd.DataFrame([row], columns=RESULT_COLUMNS)
    result_df.to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)


def make_experiment(api_key: str, trial_id: int) -> Experiment:
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
    experiment.set_name(f"{SEARCH_NAME}_trial_{trial_id:03d}")
    experiment.add_tag("xgboost")
    experiment.add_tag("i3_arboles")
    experiment.add_tag("hyperparam_search")
    experiment.add_tag("ams")
    return experiment


def evaluate_trial(
    dataset: dict[str, object],
    params: dict[str, object],
) -> tuple[XGBClassifier, dict[str, float | int]]:
    model_params = dict(BASE_PARAMS)
    model_params.update(params)
    model = XGBClassifier(**model_params)
    start = time.perf_counter()
    model.fit(dataset["X_train"], dataset["y_train"])
    fit_seconds = time.perf_counter() - start

    valid_scores = model.predict_proba(dataset["X_valid"])[:, 1]
    default_pred = model.predict(dataset["X_valid"])
    ams_metrics = find_best_ams_threshold(dataset["y_valid"], valid_scores, dataset["valid_weight"])
    metrics = {
        **ams_metrics,
        "roc_auc": float(roc_auc_score(dataset["y_valid"], valid_scores)),
        "default_accuracy": float(accuracy_score(dataset["y_valid"], default_pred)),
        "default_precision": float(precision_score(dataset["y_valid"], default_pred, zero_division=0)),
        "default_recall": float(recall_score(dataset["y_valid"], default_pred, zero_division=0)),
        "default_f1": float(f1_score(dataset["y_valid"], default_pred, zero_division=0)),
        "fit_seconds": float(fit_seconds),
    }
    return model, metrics


def update_best_model(model: XGBClassifier, row: dict[str, object], params: dict[str, object]) -> None:
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        best_ams = float(results["best_ams"].max())
    else:
        best_ams = -np.inf
    if float(row["best_ams"]) >= best_ams:
        joblib.dump(model, BEST_MODEL_PATH)
        BEST_PARAMS_PATH.write_text(
            json.dumps(
                {
                    "row": row,
                    "params": params,
                    "base_params": BASE_PARAMS,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )


def run_search(n_iter: int = 35) -> pd.DataFrame:
    ensure_dirs()
    api_key = read_comet_api_key()
    dataset = load_i3_arboles_dataset()
    completed = load_completed_trial_ids()
    trials = build_trials(n_iter)

    for trial_id, params in enumerate(trials):
        if trial_id in completed:
            print(f"trial={trial_id} skip_existing")
            continue

        experiment = make_experiment(api_key, trial_id)
        experiment.log_parameters(params)
        experiment.log_parameters(
            {
                "dataset": "i3_arboles",
                "search_name": SEARCH_NAME,
                "trial_id": trial_id,
                "valid_size": 0.1,
                "split_seed": SEED,
                "feature_count": dataset["feature_count"],
                "train_rows": len(dataset["X_train"]),
                "valid_rows": len(dataset["X_valid"]),
                "full_train_rows": dataset["full_train_rows"],
            }
        )

        try:
            model, metrics = evaluate_trial(dataset, params)
            row = {
                "trial_id": trial_id,
                "dataset": "i3_arboles",
                "params_json": json.dumps(params, sort_keys=True),
                **params,
                **metrics,
            }
            experiment.log_metrics(metrics)
            experiment.log_metric("ams", metrics["best_ams"])
            experiment.log_metric("best_ams", metrics["best_ams"])
            experiment.log_metric("ams_threshold", metrics["best_threshold"])
            append_result(row)
            update_best_model(model, row, params)
            print(
                f"trial={trial_id} ams={metrics['best_ams']:.6f} "
                f"auc={metrics['roc_auc']:.6f} threshold={metrics['best_threshold']:.6f}"
            )
        except Exception as exc:
            experiment.log_other("status", "failed")
            experiment.log_other("error", repr(exc))
            raise
        finally:
            experiment.end()

    results = pd.read_csv(RESULTS_PATH).sort_values("best_ams", ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_DIR / "01_search_results_sorted.csv", index=False)
    return results


def main() -> None:
    results = run_search()
    print("Top 10 trials:")
    print(
        results[
            [
                "trial_id",
                "best_ams",
                "roc_auc",
                "best_threshold",
                "learning_rate",
                "n_estimators",
                "max_depth",
                "min_child_weight",
                "gamma",
                "subsample",
                "colsample_bytree",
                "reg_lambda",
                "reg_alpha",
                "scale_pos_weight",
                "max_delta_step",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
