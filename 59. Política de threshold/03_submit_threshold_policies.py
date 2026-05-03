from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


SEED = 42
GLOBAL_OOF_THRESHOLD = 0.064023
TOP_SIGNAL_PERCENT = 14.0


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "roadmap.md").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


def validate_submission(submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["EventId", "RankOrder", "Class"]:
        raise ValueError("Columnas invalidas para submission.")
    ranks = submission["RankOrder"].to_numpy(dtype=np.int64)
    expected = np.arange(1, len(submission) + 1)
    if not np.array_equal(np.sort(ranks), expected):
        raise ValueError("RankOrder invalido.")
    classes = set(submission["Class"].unique())
    if not classes <= {"s", "b"}:
        raise ValueError(f"Clases invalidas: {classes}")


def build_submission(
    *,
    event_id: pd.Series,
    scores: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "EventId": event_id.to_numpy(dtype=np.int64),
            "RankOrder": pd.Series(scores).rank(method="first", ascending=True).astype(np.int64),
            "Class": classes,
        }
    )[["EventId", "RankOrder", "Class"]]
    validate_submission(submission)
    return submission


def top_percent_classes(scores: np.ndarray, percent: float) -> np.ndarray:
    n_signal = int(np.ceil(len(scores) * percent / 100.0))
    if n_signal <= 0:
        return np.full(len(scores), "b", dtype=object)

    order = np.argsort(scores)[::-1]
    classes = np.full(len(scores), "b", dtype=object)
    classes[order[:n_signal]] = "s"
    return classes


def main() -> None:
    project_root = find_project_root()
    p04_dir = project_root / "data" / "preprocesor" / "P04_feature_engineering_robust"
    results_dir = project_root / "59. Política de threshold" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(p04_dir / "X_train_processed.csv").astype(np.float32)
    X_valid = pd.read_csv(p04_dir / "X_valid_processed.csv").astype(np.float32)
    y_train = pd.read_csv(p04_dir / "y_train.csv")["Label"].astype(np.int8)
    y_valid = pd.read_csv(p04_dir / "y_valid.csv")["Label"].astype(np.int8)
    w_train = pd.read_csv(p04_dir / "train_weight.csv")["Weight"].astype(np.float64)
    w_valid = pd.read_csv(p04_dir / "valid_weight.csv")["Weight"].astype(np.float64)
    X_test = pd.read_csv(p04_dir / "X_test_processed.csv").astype(np.float32)
    test_event_id = pd.read_csv(p04_dir / "test_event_id.csv")["EventId"].astype(np.int64)

    forbidden = {"Weight", "Label", "EventId", "KaggleSet", "KaggleWeight"}
    leaked = sorted(forbidden.intersection(X_train.columns).union(forbidden.intersection(X_valid.columns)))
    if leaked:
        raise ValueError(f"Columnas prohibidas dentro de X procesado: {leaked}")

    X_full = pd.concat([X_train, X_valid], ignore_index=True)
    y_full = pd.concat([y_train, y_valid], ignore_index=True)
    w_full = pd.concat([w_train, w_valid], ignore_index=True)

    params = {
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
    model = XGBClassifier(**params)
    model.fit(X_full, y_full, sample_weight=w_full)

    class_order = model.classes_.tolist()
    if class_order != [0, 1]:
        raise ValueError(f"Orden de clases inesperado para predict_proba: {class_order}")

    test_scores = model.predict_proba(X_test)[:, 1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    global_classes = np.where(test_scores >= GLOBAL_OOF_THRESHOLD, "s", "b")
    global_submission = build_submission(
        event_id=test_event_id,
        scores=test_scores,
        classes=global_classes,
    )
    global_path = results_dir / f"higgs_submission_p04_xgboost_sw04_global_oof_threshold_{timestamp}.csv"
    global_submission.to_csv(global_path, index=False)

    top_classes = top_percent_classes(test_scores, TOP_SIGNAL_PERCENT)
    top_submission = build_submission(
        event_id=test_event_id,
        scores=test_scores,
        classes=top_classes,
    )
    top_path = results_dir / f"higgs_submission_p04_xgboost_sw04_top14pct_{timestamp}.csv"
    top_submission.to_csv(top_path, index=False)

    print(f"global_path={global_path}")
    print(f"global_threshold={GLOBAL_OOF_THRESHOLD:.6f}")
    print(f"global_signal_count={(global_submission['Class'] == 's').sum()}")
    print(f"global_background_count={(global_submission['Class'] == 'b').sum()}")
    print(f"top_path={top_path}")
    print(f"top_signal_percent={TOP_SIGNAL_PERCENT:.2f}")
    print(f"top_signal_count={(top_submission['Class'] == 's').sum()}")
    print(f"top_background_count={(top_submission['Class'] == 'b').sum()}")


if __name__ == "__main__":
    main()
