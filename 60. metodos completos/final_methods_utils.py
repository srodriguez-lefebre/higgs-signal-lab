from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEED = 42
N_SPLITS = 5
TOP_SIGNAL_PERCENT = 14.0
COMET_PROJECT_NAME = "proyecto-1-taa"
COMET_WORKSPACE = "equipo-a"
FOLDER_NAME = "60. metodos completos"
PIPELINE_ID = "P04"
PIPELINE_DIR = "P04_feature_engineering_robust"
MODEL_NAME = "XGBClassifier"
CONFIG_ID = "SW04"

REFERENCE_METRICS = {
    "cv_ams_mean": 1.670944,
    "cv_ams_std": 0.078382,
    "cv_auc_mean": 0.898346,
    "cv_threshold_median": 0.054097,
    "cv_threshold_std": 0.012263,
    "cv_signal_percent": 15.42,
    "kaggle_top14_public": 3.62371,
    "kaggle_top14_private": 3.66749,
}

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


@dataclass(frozen=True)
class P04Dataset:
    X_full: pd.DataFrame
    y_full: pd.Series
    weight_full: pd.Series
    X_test: pd.DataFrame
    test_event_id: pd.Series


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


def results_dir(project_root: Path | None = None) -> Path:
    root = find_project_root(project_root)
    output_dir = root / FOLDER_NAME / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _first_column(path: Path, dtype: str | type | None = None) -> pd.Series:
    series = pd.read_csv(path).iloc[:, 0]
    return series.astype(dtype) if dtype is not None else series


def validate_no_forbidden_features(X: pd.DataFrame) -> None:
    forbidden = {"Weight", "Label", "EventId", "KaggleSet", "KaggleWeight"}
    leaked = sorted(forbidden.intersection(X.columns))
    if leaked:
        raise ValueError(f"Columnas prohibidas dentro de X procesado: {leaked}")


def load_p04_dataset(project_root: Path | None = None) -> P04Dataset:
    root = find_project_root(project_root)
    p04_dir = root / "data" / "preprocesor" / PIPELINE_DIR
    if not p04_dir.exists():
        raise FileNotFoundError(f"No existe el dataset P04 procesado: {p04_dir}")

    X_train = pd.read_csv(p04_dir / "X_train_processed.csv").astype(np.float32)
    X_valid = pd.read_csv(p04_dir / "X_valid_processed.csv").astype(np.float32)
    X_test = pd.read_csv(p04_dir / "X_test_processed.csv").astype(np.float32)
    validate_no_forbidden_features(X_train)
    validate_no_forbidden_features(X_valid)
    validate_no_forbidden_features(X_test)

    y_train = _first_column(p04_dir / "y_train.csv", np.int8)
    y_valid = _first_column(p04_dir / "y_valid.csv", np.int8)
    w_train = _first_column(p04_dir / "train_weight.csv", np.float64)
    w_valid = _first_column(p04_dir / "valid_weight.csv", np.float64)
    test_event_id = _first_column(p04_dir / "test_event_id.csv", np.int64)

    return P04Dataset(
        X_full=pd.concat([X_train, X_valid], ignore_index=True),
        y_full=pd.concat([y_train, y_valid], ignore_index=True),
        weight_full=pd.concat([w_train, w_valid], ignore_index=True),
        X_test=X_test,
        test_event_id=test_event_id,
    )


def validate_test_event_id(event_id: pd.Series) -> None:
    if event_id.isna().any():
        raise ValueError("Hay EventId nulos.")
    if event_id.duplicated().any():
        raise ValueError("Hay EventId duplicados.")
    if len(event_id) != 550_000:
        raise ValueError(f"Cantidad inesperada de EventId de test: {len(event_id)}")


def validate_submission(
    *,
    submission: pd.DataFrame,
    expected_event_id: pd.Series,
    expected_signal_percent: float = TOP_SIGNAL_PERCENT,
    tolerance_points: float = 0.01,
) -> dict[str, float | int]:
    if list(submission.columns) != ["EventId", "RankOrder", "Class"]:
        raise ValueError("Columnas invalidas para submission.")
    if len(submission) != len(expected_event_id):
        raise ValueError("Cantidad de filas incorrecta.")
    if not submission["EventId"].reset_index(drop=True).equals(expected_event_id.reset_index(drop=True)):
        raise ValueError("EventId no coincide con el test procesado.")

    ranks = submission["RankOrder"].to_numpy(dtype=np.int64)
    expected_ranks = np.arange(1, len(submission) + 1)
    if not np.array_equal(np.sort(ranks), expected_ranks):
        raise ValueError("RankOrder invalido.")

    classes = set(submission["Class"].unique())
    if not classes <= {"s", "b"}:
        raise ValueError(f"Clases invalidas: {classes}")

    signal_count = int((submission["Class"] == "s").sum())
    background_count = int((submission["Class"] == "b").sum())
    signal_percent = float(signal_count / len(submission) * 100.0)
    if abs(signal_percent - expected_signal_percent) > tolerance_points:
        raise ValueError(
            f"Signal % fuera de tolerancia: {signal_percent:.4f}% "
            f"vs esperado {expected_signal_percent:.4f}%"
        )

    return {
        "rows": int(len(submission)),
        "signal_count": signal_count,
        "background_count": background_count,
        "signal_percent": signal_percent,
        "rank_unique_count": int(submission["RankOrder"].nunique()),
    }


def top_percent_classes(scores: np.ndarray, percent: float) -> np.ndarray:
    n_signal = int(np.ceil(len(scores) * percent / 100.0))
    if n_signal <= 0:
        return np.full(len(scores), "b", dtype=object)
    order = np.argsort(scores)[::-1]
    classes = np.full(len(scores), "b", dtype=object)
    classes[order[:n_signal]] = "s"
    return classes


def build_top_percent_submission(
    *,
    event_id: pd.Series,
    scores: np.ndarray,
    top_signal_percent: float = TOP_SIGNAL_PERCENT,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    validate_test_event_id(event_id)
    classes = top_percent_classes(scores, top_signal_percent)
    submission = pd.DataFrame(
        {
            "EventId": event_id.to_numpy(dtype=np.int64),
            "RankOrder": pd.Series(scores).rank(method="first", ascending=True).astype(np.int64),
            "Class": classes,
        }
    )[["EventId", "RankOrder", "Class"]]
    validation = validate_submission(
        submission=submission,
        expected_event_id=event_id,
        expected_signal_percent=top_signal_percent,
    )
    return submission, validation


def make_model() -> XGBClassifier:
    return XGBClassifier(**SW04_PARAMS)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_submission_to_comet(
    *,
    experiment_name: str,
    scores_path: Path,
    submission_path: Path,
    metrics: dict[str, float | int],
    extra_params: dict[str, object],
    log_to_comet: bool,
) -> None:
    if not log_to_comet:
        return

    from comet_ml import Experiment

    api_key = read_comet_api_key()
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
    experiment.set_name(experiment_name)
    for tag in [FOLDER_NAME, PIPELINE_ID, MODEL_NAME, CONFIG_ID, "top14_threshold_policy"]:
        experiment.add_tag(tag)
    experiment.log_parameters(SW04_PARAMS)
    experiment.log_parameters(
        {
            "pipeline": PIPELINE_ID,
            "model": MODEL_NAME,
            "config_id": CONFIG_ID,
            "seed": SEED,
            "sample_weight_used": True,
            "weight_used_as_feature": False,
            "threshold_policy": "top_14_percent_signal",
            "top_signal_percent": TOP_SIGNAL_PERCENT,
            "reference_metrics_json": json.dumps(REFERENCE_METRICS, sort_keys=True),
            **extra_params,
        }
    )
    experiment.log_metrics(metrics)
    experiment.log_asset(str(scores_path), file_name=scores_path.name)
    experiment.log_asset(str(submission_path), file_name=submission_path.name)
    experiment.end()


def run_cv_ensemble_submission(
    *,
    log_to_comet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    output_dir = results_dir()
    dataset = load_p04_dataset()
    fold_test_scores: list[np.ndarray] = []
    fold_fit_seconds: list[float] = []
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold_id, (train_idx, _) in enumerate(cv.split(dataset.X_full, dataset.y_full), start=1):
        model = make_model()
        start = time.perf_counter()
        model.fit(
            dataset.X_full.iloc[train_idx].reset_index(drop=True),
            dataset.y_full.iloc[train_idx].reset_index(drop=True),
            sample_weight=dataset.weight_full.iloc[train_idx].reset_index(drop=True),
        )
        fold_fit_seconds.append(time.perf_counter() - start)

        class_order = model.classes_.tolist()
        if class_order != [0, 1]:
            raise ValueError(f"Orden de clases inesperado para predict_proba: {class_order}")
        fold_test_scores.append(model.predict_proba(dataset.X_test)[:, 1].astype(np.float64))

    score_matrix = np.column_stack(fold_test_scores)
    mean_scores = score_matrix.mean(axis=1)
    score_columns = {f"score_fold_{idx}": score_matrix[:, idx - 1] for idx in range(1, N_SPLITS + 1)}
    scores_df = pd.DataFrame(
        {
            "EventId": dataset.test_event_id.to_numpy(dtype=np.int64),
            **score_columns,
            "score_mean": mean_scores,
        }
    )
    submission, validation = build_top_percent_submission(
        event_id=dataset.test_event_id,
        scores=mean_scores,
    )

    run_timestamp = timestamp()
    scores_path = output_dir / f"p04_xgboost_sw04_cv_ensemble_test_scores_{run_timestamp}.csv"
    submission_path = output_dir / f"higgs_submission_p04_xgboost_sw04_cv_ensemble_top14pct_{run_timestamp}.csv"
    scores_df.to_csv(scores_path, index=False)
    submission.to_csv(submission_path, index=False)

    metrics = {
        **validation,
        "fit_seconds_total": float(sum(fold_fit_seconds)),
        "fit_seconds_mean": float(np.mean(fold_fit_seconds)),
        "fit_seconds_std": float(np.std(fold_fit_seconds, ddof=1)),
        "test_score_mean": float(np.mean(mean_scores)),
        "test_score_std": float(np.std(mean_scores, ddof=1)),
        "test_score_min": float(np.min(mean_scores)),
        "test_score_max": float(np.max(mean_scores)),
        **REFERENCE_METRICS,
    }
    log_submission_to_comet(
        experiment_name=f"{FOLDER_NAME}_cv_ensemble_P04_XGBoost_SW04_top14",
        scores_path=scores_path,
        submission_path=submission_path,
        metrics=metrics,
        extra_params={
            "method": "cv_ensemble",
            "n_models": N_SPLITS,
            "fold_fit_seconds_json": json.dumps(fold_fit_seconds),
        },
        log_to_comet=log_to_comet,
    )
    return scores_df, submission, metrics


def run_full_train_submission(
    *,
    log_to_comet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    output_dir = results_dir()
    dataset = load_p04_dataset()
    model = make_model()
    start = time.perf_counter()
    model.fit(dataset.X_full, dataset.y_full, sample_weight=dataset.weight_full)
    fit_seconds = time.perf_counter() - start

    class_order = model.classes_.tolist()
    if class_order != [0, 1]:
        raise ValueError(f"Orden de clases inesperado para predict_proba: {class_order}")

    test_scores = model.predict_proba(dataset.X_test)[:, 1].astype(np.float64)
    scores_df = pd.DataFrame(
        {
            "EventId": dataset.test_event_id.to_numpy(dtype=np.int64),
            "score": test_scores,
        }
    )
    submission, validation = build_top_percent_submission(
        event_id=dataset.test_event_id,
        scores=test_scores,
    )

    run_timestamp = timestamp()
    scores_path = output_dir / f"p04_xgboost_sw04_full_train_test_scores_{run_timestamp}.csv"
    submission_path = output_dir / f"higgs_submission_p04_xgboost_sw04_full_train_top14pct_{run_timestamp}.csv"
    scores_df.to_csv(scores_path, index=False)
    submission.to_csv(submission_path, index=False)

    metrics = {
        **validation,
        "fit_seconds": float(fit_seconds),
        "test_score_mean": float(np.mean(test_scores)),
        "test_score_std": float(np.std(test_scores, ddof=1)),
        "test_score_min": float(np.min(test_scores)),
        "test_score_max": float(np.max(test_scores)),
        **REFERENCE_METRICS,
    }
    log_submission_to_comet(
        experiment_name=f"{FOLDER_NAME}_full_train_P04_XGBoost_SW04_top14",
        scores_path=scores_path,
        submission_path=submission_path,
        metrics=metrics,
        extra_params={
            "method": "full_train",
            "n_models": 1,
        },
        log_to_comet=log_to_comet,
    )
    return scores_df, submission, metrics
