from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEED = 42
N_SPLITS = 5
COMET_PROJECT_NAME = "proyecto-1-taa"
COMET_WORKSPACE = "equipo-a"
FOLDER_NAME = "59. Política de threshold"
PIPELINE_ID = "P04"
PIPELINE_DIR = "P04_feature_engineering_robust"
MODEL_NAME = "XGBClassifier"
CONFIG_ID = "SW04"

SW04_PARAMS = {
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
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
}


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "roadmap.md").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


def results_dir(project_root: Path | None = None) -> Path:
    root = find_project_root(project_root)
    output_dir = root / FOLDER_NAME / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_comet_api_key(project_root: Path | None = None) -> str:
    root = find_project_root(project_root)
    api_key = (root / "keys" / ".comet.config").read_text().strip()
    if not api_key:
        raise ValueError("La API key de Comet esta vacia.")
    return api_key


def _first_column(path: Path, dtype: str | type | None = None) -> pd.Series:
    series = pd.read_csv(path).iloc[:, 0]
    return series.astype(dtype) if dtype is not None else series


def load_p04_full_dataset(project_root: Path | None = None) -> dict[str, object]:
    root = find_project_root(project_root)
    dataset_dir = root / "data" / "preprocesor" / PIPELINE_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {dataset_dir}")

    X_train = pd.read_csv(dataset_dir / "X_train_processed.csv").astype(np.float32)
    X_valid = pd.read_csv(dataset_dir / "X_valid_processed.csv").astype(np.float32)
    y_train = _first_column(dataset_dir / "y_train.csv", np.int8)
    y_valid = _first_column(dataset_dir / "y_valid.csv", np.int8)
    w_train = _first_column(dataset_dir / "train_weight.csv", np.float64)
    w_valid = _first_column(dataset_dir / "valid_weight.csv", np.float64)
    event_train = _first_column(dataset_dir / "train_event_id.csv", np.int64)
    event_valid = _first_column(dataset_dir / "valid_event_id.csv", np.int64)

    forbidden = {"Weight", "Label", "EventId", "KaggleSet", "KaggleWeight"}
    leaked = sorted(forbidden.intersection(X_train.columns).union(forbidden.intersection(X_valid.columns)))
    if leaked:
        raise ValueError(f"Columnas prohibidas dentro de X procesado: {leaked}")

    return {
        "pipeline_id": PIPELINE_ID,
        "dataset_dir": dataset_dir,
        "X": pd.concat([X_train, X_valid], ignore_index=True),
        "y": pd.concat([y_train, y_valid], ignore_index=True),
        "weights": pd.concat([w_train, w_valid], ignore_index=True),
        "event_id": pd.concat([event_train, event_valid], ignore_index=True),
    }


def ams_score(signal_weight: np.ndarray | float, background_weight: np.ndarray | float, br: float = 10.0) -> np.ndarray:
    signal_arr = np.asarray(signal_weight, dtype=np.float64)
    background_arr = np.asarray(background_weight, dtype=np.float64)
    radicand = 2.0 * (
        (signal_arr + background_arr + br) * np.log1p(signal_arr / (background_arr + br)) - signal_arr
    )
    return np.sqrt(np.clip(radicand, a_min=0.0, a_max=None))


def policy_metrics(
    *,
    y_true: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    policy_id: str,
    rule_used: str,
    threshold: float | None = None,
    selected_mask: np.ndarray | None = None,
) -> dict[str, float | int | str | None]:
    y_array = np.asarray(y_true, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)

    if selected_mask is None:
        if threshold is None:
            raise ValueError("Se necesita threshold o selected_mask.")
        y_pred = (score_array >= threshold).astype(np.int8)
    else:
        y_pred = np.asarray(selected_mask, dtype=bool).astype(np.int8)
        if threshold is None and y_pred.sum() > 0:
            threshold = float(score_array[y_pred == 1].min())

    signal_weight = float(weight_array[(y_array == 1) & (y_pred == 1)].sum())
    background_weight = float(weight_array[(y_array == 0) & (y_pred == 1)].sum())

    return {
        "pipeline": PIPELINE_ID,
        "model": MODEL_NAME,
        "config_id": CONFIG_ID,
        "policy_id": policy_id,
        "rule_used": rule_used,
        "threshold": None if threshold is None else float(threshold),
        "ams": float(ams_score(signal_weight, background_weight)),
        "signal_rate": float(y_pred.mean()),
        "signal_percent": float(100.0 * y_pred.mean()),
        "precision": float(precision_score(y_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_array, y_pred, zero_division=0)),
        "signal_weight": signal_weight,
        "background_weight": background_weight,
        "predicted_signal_count": int(y_pred.sum()),
        "predicted_background_count": int(len(y_pred) - y_pred.sum()),
    }


def find_best_ams_threshold(
    y_true: pd.Series | np.ndarray,
    y_scores: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
) -> dict[str, float | int | str | None]:
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

    metrics = policy_metrics(
        y_true=y_array,
        weights=weight_array,
        scores=score_array,
        policy_id="best_ams_threshold",
        rule_used="threshold que maximiza AMS",
        threshold=threshold,
    )
    metrics.update(
        {
            "best_rank_position": int(best_idx + 1),
            "unique_threshold_candidates": int(len(candidate_indices)),
        }
    )
    return metrics


def _make_comet_experiment(*, api_key: str, name: str, tags: list[str]):
    from comet_ml import Experiment

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
    experiment.set_name(name)
    for tag in tags:
        experiment.add_tag(tag)
    return experiment


def generate_sw04_oof_scores(
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    log_to_comet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = find_project_root()
    output_dir = results_dir(project_root)
    dataset = load_p04_full_dataset(project_root)
    X = dataset["X"]
    y = dataset["y"]
    weights = dataset["weights"]
    event_id = dataset["event_id"]
    api_key = read_comet_api_key(project_root) if log_to_comet else ""

    oof_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = XGBClassifier(**SW04_PARAMS)
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

        fold_best = find_best_ams_threshold(y_valid, valid_scores, w_valid)
        fold_best.update(
            {
                "fold_id": int(fold),
                "seed": int(seed),
                "roc_auc": float(roc_auc_score(y_valid, valid_scores)),
                "fit_seconds": float(fit_seconds),
                "sample_weight_used": True,
                "weight_used_as_feature": False,
                "predict_proba_signal_column": 1,
                "class_order": json.dumps(class_order),
                "hyperparameters_json": json.dumps(SW04_PARAMS, sort_keys=True),
            }
        )
        fold_rows.append(fold_best)

        oof_rows.append(
            pd.DataFrame(
                {
                    "event_id_valid": event_id.iloc[valid_idx].to_numpy(dtype=np.int64),
                    "y_valid": y_valid.to_numpy(dtype=np.int8),
                    "weights_valid": w_valid.to_numpy(dtype=np.float64),
                    "scores_valid": valid_scores.astype(np.float64),
                    "fold_id": int(fold),
                }
            )
        )

        if log_to_comet:
            experiment = _make_comet_experiment(
                api_key=api_key,
                name=f"{FOLDER_NAME}_{PIPELINE_ID}_{MODEL_NAME}_{CONFIG_ID}_oof_fold_{fold}",
                tags=[FOLDER_NAME, PIPELINE_ID, MODEL_NAME, CONFIG_ID, "oof_scores", "threshold_policy"],
            )
            experiment.log_parameters(SW04_PARAMS)
            experiment.log_parameters(
                {
                    "pipeline": PIPELINE_ID,
                    "model": MODEL_NAME,
                    "config_id": CONFIG_ID,
                    "seed": seed,
                    "fold": fold,
                    "n_splits": n_splits,
                    "sample_weight_used": True,
                    "weight_used_as_feature": False,
                }
            )
            experiment.log_metrics(
                {
                    "ams": fold_best["ams"],
                    "auc": fold_best["roc_auc"],
                    "threshold_optimal_ams": fold_best["threshold"],
                    "signal_percent": fold_best["signal_percent"],
                    "precision": fold_best["precision"],
                    "recall": fold_best["recall"],
                    "f1": fold_best["f1"],
                    "signal_weight": fold_best["signal_weight"],
                    "background_weight": fold_best["background_weight"],
                    "fit_seconds": fit_seconds,
                }
            )
            experiment.end()

    oof_df = pd.concat(oof_rows, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)

    oof_path = output_dir / "p04_xgboost_sw04_oof_scores.csv"
    fold_path = output_dir / "p04_xgboost_sw04_fold_thresholds.csv"
    oof_df.to_csv(oof_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    return oof_df, fold_df


def top_percent_mask(scores: pd.Series | np.ndarray, percent: float) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    n_signal = int(np.ceil(len(score_array) * percent / 100.0))
    if n_signal <= 0:
        return np.zeros(len(score_array), dtype=bool)
    order = np.argsort(score_array)[::-1]
    selected = np.zeros(len(score_array), dtype=bool)
    selected[order[:n_signal]] = True
    return selected


def compare_threshold_policies(
    *,
    top_percentages: list[float] | None = None,
    log_to_comet: bool = True,
) -> pd.DataFrame:
    project_root = find_project_root()
    output_dir = results_dir(project_root)
    oof_path = output_dir / "p04_xgboost_sw04_oof_scores.csv"
    fold_path = output_dir / "p04_xgboost_sw04_fold_thresholds.csv"
    if not oof_path.exists() or not fold_path.exists():
        raise FileNotFoundError("Primero hay que generar los OOF scores y thresholds por fold.")

    oof_df = pd.read_csv(oof_path)
    fold_df = pd.read_csv(fold_path)
    y = oof_df["y_valid"]
    weights = oof_df["weights_valid"]
    scores = oof_df["scores_valid"]

    median_threshold = float(fold_df["threshold"].median())
    mean_threshold = float(fold_df["threshold"].mean())
    rows = [
        policy_metrics(
            y_true=y,
            weights=weights,
            scores=scores,
            policy_id="median_fold_threshold",
            rule_used="mediana de thresholds optimos por fold",
            threshold=median_threshold,
        ),
        policy_metrics(
            y_true=y,
            weights=weights,
            scores=scores,
            policy_id="mean_fold_threshold",
            rule_used="promedio de thresholds optimos por fold",
            threshold=mean_threshold,
        ),
    ]

    global_best = find_best_ams_threshold(y, scores, weights)
    global_best["policy_id"] = "global_oof_optimal_threshold"
    global_best["rule_used"] = "threshold optimo global usando todos los scores OOF"
    rows.append(global_best)

    if top_percentages is None:
        fold_signal_percent_median = float(fold_df["signal_percent"].median())
        top_percentages = sorted(
            {
                10.0,
                12.0,
                14.0,
                15.0,
                16.0,
                18.0,
                round(fold_signal_percent_median, 4),
            }
        )

    for percent in top_percentages:
        selected = top_percent_mask(scores, percent)
        rows.append(
            policy_metrics(
                y_true=y,
                weights=weights,
                scores=scores,
                policy_id=f"top_{percent:g}_percent_signal",
                rule_used=f"top {percent:g}% de eventos por score como signal",
                selected_mask=selected,
            )
        )

    comparison_df = pd.DataFrame(rows).sort_values("ams", ascending=False).reset_index(drop=True)
    comparison_path = output_dir / "p04_xgboost_sw04_threshold_policy_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    if log_to_comet:
        api_key = read_comet_api_key(project_root)
        experiment = _make_comet_experiment(
            api_key=api_key,
            name=f"{FOLDER_NAME}_{PIPELINE_ID}_{MODEL_NAME}_{CONFIG_ID}_threshold_policies",
            tags=[FOLDER_NAME, PIPELINE_ID, MODEL_NAME, CONFIG_ID, "threshold_policy", "oof_scores"],
        )
        experiment.log_parameters(SW04_PARAMS)
        experiment.log_parameters(
            {
                "pipeline": PIPELINE_ID,
                "model": MODEL_NAME,
                "config_id": CONFIG_ID,
                "sample_weight_used": True,
                "weight_used_as_feature": False,
                "fold_threshold_median": median_threshold,
                "fold_threshold_mean": mean_threshold,
                "top_percentages": json.dumps(top_percentages),
            }
        )
        for _, row in comparison_df.iterrows():
            prefix = str(row["policy_id"])
            experiment.log_metrics(
                {
                    f"{prefix}_ams": row["ams"],
                    f"{prefix}_signal_percent": row["signal_percent"],
                    f"{prefix}_precision": row["precision"],
                    f"{prefix}_recall": row["recall"],
                    f"{prefix}_f1": row["f1"],
                    f"{prefix}_signal_weight": row["signal_weight"],
                    f"{prefix}_background_weight": row["background_weight"],
                }
            )
            if pd.notna(row["threshold"]):
                experiment.log_metric(f"{prefix}_threshold", row["threshold"])
        experiment.log_table("threshold_policy_comparison.csv", comparison_df)
        experiment.end()

    return comparison_df
