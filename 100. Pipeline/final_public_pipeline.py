from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEED = 42
N_SPLITS = 5
PIPELINE_ID = "P04"
MODEL_NAME = "XGBClassifier"
CONFIG_ID = "SW04"
TOP_SIGNAL_PERCENT = 15.75
EXPECTED_TEST_ROWS = 550_000
EXPECTED_SIGNAL_COUNT = 86_625

FINAL_XGB_PARAMS = {
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

REFERENCE_KAGGLE = {
    "public_score": 3.66729,
    "private_score": 3.65941,
    "submission": "higgs_submission_p04_xgboost_sw04_cv_ensemble_top15p75pct_20260503_002739.csv",
}


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "roadmap.md").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


def load_preprocessing_lib(project_root: Path | None = None):
    root = find_project_root(project_root)
    module_path = root / "50. preprocesamiento" / "preprocessing_lib.py"
    spec = importlib.util.spec_from_file_location("final_pipeline_preprocessing_lib", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def output_dir(project_root: Path | None = None) -> Path:
    root = find_project_root(project_root)
    result_dir = root / "100. Pipeline" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def top_percent_classes(scores: np.ndarray, percent: float) -> np.ndarray:
    n_signal = int(np.ceil(len(scores) * percent / 100.0))
    selected_order = np.argsort(scores)[::-1]
    classes = np.full(len(scores), "b", dtype=object)
    classes[selected_order[:n_signal]] = "s"
    return classes


def validate_submission(submission: pd.DataFrame, expected_event_id: pd.Series) -> dict[str, float | int]:
    if list(submission.columns) != ["EventId", "RankOrder", "Class"]:
        raise ValueError("Columnas invalidas para submission.")
    if len(submission) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Cantidad de filas incorrecta: {len(submission)}")
    if not submission["EventId"].reset_index(drop=True).equals(expected_event_id.reset_index(drop=True)):
        raise ValueError("EventId no coincide con el test raw.")
    if submission["EventId"].duplicated().any():
        raise ValueError("EventId duplicado.")

    ranks = submission["RankOrder"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(ranks), np.arange(1, len(submission) + 1)):
        raise ValueError("RankOrder debe ser unico y cubrir 1..n.")
    classes = set(submission["Class"].unique())
    if not classes <= {"s", "b"}:
        raise ValueError(f"Clases invalidas: {classes}")

    signal_count = int((submission["Class"] == "s").sum())
    if signal_count != EXPECTED_SIGNAL_COUNT:
        raise ValueError(f"Signal count incorrecto: {signal_count} != {EXPECTED_SIGNAL_COUNT}")

    return {
        "rows": int(len(submission)),
        "event_id_unique": int(submission["EventId"].nunique()),
        "rank_order_unique": int(submission["RankOrder"].nunique()),
        "signal_count": signal_count,
        "background_count": int((submission["Class"] == "b").sum()),
        "signal_percent": float(signal_count / len(submission) * 100.0),
    }


def make_submission(event_id: pd.Series, scores: np.ndarray) -> tuple[pd.DataFrame, dict[str, float | int]]:
    submission = pd.DataFrame(
        {
            "EventId": event_id.to_numpy(dtype=np.int64),
            "RankOrder": pd.Series(scores).rank(method="first", ascending=True).astype(np.int64),
            "Class": top_percent_classes(scores, TOP_SIGNAL_PERCENT),
        }
    )[["EventId", "RankOrder", "Class"]]
    validation = validate_submission(submission, event_id)
    return submission, validation


def run_final_public_pipeline(project_root: Path | None = None) -> dict[str, object]:
    root = find_project_root(project_root)
    preprocessing = load_preprocessing_lib(root)
    result_dir = output_dir(root)

    common = preprocessing.build_common_data(project_root=root)
    p04 = preprocessing.transform_global_pipeline(PIPELINE_ID, common)

    X_full = pd.concat(
        [p04["X_train_processed"], p04["X_valid_processed"]],
        ignore_index=True,
    )
    y_full = pd.concat([common.y_train, common.y_valid], ignore_index=True)
    weight_full = pd.concat([common.train_weight, common.valid_weight], ignore_index=True)
    X_test = p04["X_test_processed"]

    score_columns: dict[str, np.ndarray] = {}
    fit_seconds: list[float] = []
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold_id, (train_idx, _) in enumerate(cv.split(X_full, y_full), start=1):
        model = XGBClassifier(**FINAL_XGB_PARAMS)
        start = time.perf_counter()
        model.fit(
            X_full.iloc[train_idx].reset_index(drop=True),
            y_full.iloc[train_idx].reset_index(drop=True),
            sample_weight=weight_full.iloc[train_idx].reset_index(drop=True),
        )
        fit_seconds.append(time.perf_counter() - start)

        class_order = model.classes_.tolist()
        if class_order != [0, 1]:
            raise ValueError(f"Orden inesperado en predict_proba: {class_order}")
        score_columns[f"score_fold_{fold_id}"] = model.predict_proba(X_test)[:, 1].astype(np.float64)

    score_matrix = np.column_stack([score_columns[f"score_fold_{fold}"] for fold in range(1, N_SPLITS + 1)])
    mean_scores = score_matrix.mean(axis=1)
    scores_df = pd.DataFrame(
        {
            "EventId": common.test_event_id.to_numpy(dtype=np.int64),
            **score_columns,
            "score_mean": mean_scores,
        }
    )
    submission, validation = make_submission(common.test_event_id, mean_scores)

    scores_path = result_dir / "p04_xgboost_sw04_cv_ensemble_top15p75_test_scores.csv"
    submission_path = result_dir / "higgs_submission_p04_xgboost_sw04_cv_ensemble_top15p75_final_public.csv"
    metadata_path = result_dir / "final_public_pipeline_metadata.json"
    scores_df.to_csv(scores_path, index=False)
    submission.to_csv(submission_path, index=False)

    metadata: dict[str, object] = {
        "pipeline": PIPELINE_ID,
        "model": MODEL_NAME,
        "config_id": CONFIG_ID,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "sample_weight_used": True,
        "weight_used_as_feature": False,
        "threshold_policy": f"top_{TOP_SIGNAL_PERCENT:g}_percent_by_score_rank",
        "top_signal_percent": TOP_SIGNAL_PERCENT,
        "hyperparameters": FINAL_XGB_PARAMS,
        "reference_kaggle": REFERENCE_KAGGLE,
        "raw_train_path": str(root / "data" / "higgs-boson" / "training.csv"),
        "raw_test_path": str(root / "data" / "higgs-boson" / "test.csv"),
        "processed_shapes": {
            "X_train_processed": list(p04["X_train_processed"].shape),
            "X_valid_processed": list(p04["X_valid_processed"].shape),
            "X_full": list(X_full.shape),
            "X_test_processed": list(X_test.shape),
        },
        "validation": validation,
        "fit_seconds": {
            "per_fold": fit_seconds,
            "total": float(sum(fit_seconds)),
            "mean": float(np.mean(fit_seconds)),
        },
        "scores_path": str(scores_path),
        "submission_path": str(submission_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    return {
        "scores": scores_df,
        "submission": submission,
        "metadata": metadata,
        "scores_path": scores_path,
        "submission_path": submission_path,
        "metadata_path": metadata_path,
    }
