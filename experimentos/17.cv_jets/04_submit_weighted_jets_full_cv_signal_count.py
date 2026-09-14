from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "17.cv_jets" / "results"


def validate_submission(submission: pd.DataFrame) -> dict[str, int]:
    ranks = submission["RankOrder"].to_numpy(dtype=np.int64)
    if list(submission.columns) != ["EventId", "RankOrder", "Class"]:
        raise ValueError("Columnas invalidas.")
    if not submission["EventId"].is_unique:
        raise ValueError("EventId repetido.")
    if ranks.min() != 1 or ranks.max() != len(ranks) or len(np.unique(ranks)) != len(ranks):
        raise ValueError("RankOrder invalido.")
    return {
        "rows": int(len(submission)),
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "signal_count": int((submission["Class"] == "s").sum()),
        "background_count": int((submission["Class"] == "b").sum()),
    }


def build_top_n_submission(test_scores: pd.DataFrame, signal_count: int) -> pd.DataFrame:
    scores = test_scores["score"].to_numpy(dtype=np.float64)
    order = np.argsort(scores)[::-1]
    classes = np.full(len(scores), "b", dtype=object)
    classes[order[:signal_count]] = "s"
    return pd.DataFrame(
        {
            "EventId": test_scores["EventId"].astype(np.int64),
            "RankOrder": pd.Series(scores).rank(method="first", ascending=True).astype(np.int64),
            "Class": classes,
        }
    )[["EventId", "RankOrder", "Class"]]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_scores = pd.read_csv(RESULTS_DIR / "03_weighted_jets_full_test_scores.csv")
    folds = pd.read_csv(RESULTS_DIR / "01_policy_folds.csv")
    selected = folds[(folds["variant"] == "weighted14") & (folds["policy"] == "joint_thresholds_from_independent")]
    fold_rows = int(selected["predicted_signal_count"].median())
    mean_fold_rows = float(selected["predicted_signal_count"].mean())
    scale = len(test_scores) / 50000.0
    signal_counts = {
        "cv_median_count": int(round(fold_rows * scale)),
        "cv_mean_count": int(round(mean_fold_rows * scale)),
    }

    metadata: dict[str, object] = {"signal_counts": signal_counts, "submissions": {}}
    for name, signal_count in signal_counts.items():
        submission = build_top_n_submission(test_scores, signal_count)
        validation = validate_submission(submission)
        path = RESULTS_DIR / f"higgs_submission_weighted_jets_full_{name}.csv"
        submission.to_csv(path, index=False)
        metadata["submissions"][name] = {"path": str(path), "validation": validation}
        print(f"{name}: {validation} -> {path}")

    (RESULTS_DIR / "04_weighted_jets_full_count_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
