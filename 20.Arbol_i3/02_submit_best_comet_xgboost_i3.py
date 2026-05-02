from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = PROJECT_ROOT / "20.Arbol_i3"
RESULTS_DIR = STAGE_DIR / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "higgs-boson" / "processed" / "i3_arboles"
BEST_PARAMS_PATH = RESULTS_DIR / "01_best_params.json"


def load_best_config() -> tuple[dict[str, object], float]:
    best_config = json.loads(BEST_PARAMS_PATH.read_text())
    return best_config["params"], float(best_config["row"]["best_threshold"])


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    label_map = {"b": 0, "s": 1}
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_processed.csv").astype(np.float32)
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Label"].map(label_map).astype(np.int8)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_processed.csv").astype(np.float32)
    test_event_id = pd.read_csv(PROCESSED_DIR / "test_event_id.csv")["EventId"].astype(np.int64)
    return X_train, y_train, X_test, test_event_id


def build_submission(event_id: pd.Series, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    score_series = pd.Series(scores)
    submission = pd.DataFrame(
        {
            "EventId": event_id.to_numpy(dtype=np.int64),
            "RankOrder": score_series.rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(scores >= threshold, "s", "b"),
        }
    )[["EventId", "RankOrder", "Class"]]
    validate_submission(submission)
    return submission


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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    params, threshold = load_best_config()
    X_train, y_train, X_test, test_event_id = load_data()

    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "tree_method": "hist",
        "n_jobs": -1,
        "missing": np.nan,
        **params,
    }
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    test_scores = model.predict_proba(X_test)[:, 1]
    submission = build_submission(test_event_id, test_scores, threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = RESULTS_DIR / f"higgs_submission_i3_arboles_xgboost_comet_tuned_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"submission_path={submission_path}")
    print(f"rows={len(submission)}")
    print(f"threshold={threshold:.9f}")
    print(f"signal_count={(submission['Class'] == 's').sum()}")
    print(f"background_count={(submission['Class'] == 'b').sum()}")


if __name__ == "__main__":
    main()
