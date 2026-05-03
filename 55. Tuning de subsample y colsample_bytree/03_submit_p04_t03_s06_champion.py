from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


SEED = 42
CV_MEDIAN_THRESHOLD = 0.881877


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


def main() -> None:
    project_root = find_project_root()
    p04_dir = project_root / "data" / "preprocesor" / "P04_feature_engineering_robust"
    results_dir = project_root / "55. Tuning de subsample y colsample_bytree" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(p04_dir / "X_train_processed.csv").astype(np.float32)
    X_valid = pd.read_csv(p04_dir / "X_valid_processed.csv").astype(np.float32)
    y_train = pd.read_csv(p04_dir / "y_train.csv")["Label"].astype(np.int8)
    y_valid = pd.read_csv(p04_dir / "y_valid.csv")["Label"].astype(np.int8)
    X_test = pd.read_csv(p04_dir / "X_test_processed.csv").astype(np.float32)
    test_event_id = pd.read_csv(p04_dir / "test_event_id.csv")["EventId"].astype(np.int64)

    X_full = pd.concat([X_train, X_valid], ignore_index=True)
    y_full = pd.concat([y_train, y_valid], ignore_index=True)

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
        "scale_pos_weight": 1.5346,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "max_delta_step": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_full, y_full)

    test_scores = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {
            "EventId": test_event_id.to_numpy(dtype=np.int64),
            "RankOrder": pd.Series(test_scores).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(test_scores >= CV_MEDIAN_THRESHOLD, "s", "b"),
        }
    )[["EventId", "RankOrder", "Class"]]
    validate_submission(submission)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = results_dir / f"higgs_submission_p04_xgboost_t03_s06_champion_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"submission_path={submission_path}")
    print(f"rows={len(submission)}")
    print(f"threshold={CV_MEDIAN_THRESHOLD:.6f}")
    print(f"signal_count={(submission['Class'] == 's').sum()}")
    print(f"background_count={(submission['Class'] == 'b').sum()}")


if __name__ == "__main__":
    main()
