from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "16.cv" / "results"


def validate_submission(submission: pd.DataFrame) -> dict[str, int]:
    if list(submission.columns) != ["EventId", "RankOrder", "Class"]:
        raise ValueError("Columnas invalidas.")
    ranks = submission["RankOrder"].to_numpy(dtype=np.int64)
    if not submission["EventId"].is_unique:
        raise ValueError("EventId repetido.")
    if ranks.min() != 1 or ranks.max() != len(ranks) or len(np.unique(ranks)) != len(ranks):
        raise ValueError("RankOrder invalido.")
    invalid = sorted(set(submission["Class"].unique()) - {"s", "b"})
    if invalid:
        raise ValueError(f"Clases invalidas: {invalid}")
    return {
        "rows": int(len(submission)),
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "signal_count": int((submission["Class"] == "s").sum()),
        "background_count": int((submission["Class"] == "b").sum()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_scores = pd.read_csv(PROJECT_ROOT / "15.tune_global" / "results" / "01_test_scores.csv")
    blend_sweep = pd.read_csv(PROJECT_ROOT / "15.tune_global" / "results" / "01_blend_sweep.csv")
    selected = blend_sweep.loc[blend_sweep["blend_key"] == "percentile_rank_i3_0.20_v2_0.80"].iloc[0]

    i3_rank = test_scores["i3_tree_safe_mix_score"].rank(method="first", pct=True).to_numpy(dtype=np.float64)
    v2_rank = test_scores["v2_physics_by_jet_score"].rank(method="first", pct=True).to_numpy(dtype=np.float64)
    ensemble_score = 0.20 * i3_rank + 0.80 * v2_rank
    threshold = float(selected["ams_threshold"])

    submission = pd.DataFrame(
        {
            "EventId": test_scores["EventId"].astype(np.int64),
            "RankOrder": pd.Series(ensemble_score).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(ensemble_score >= threshold, "s", "b"),
        }
    )[["EventId", "RankOrder", "Class"]]
    validation = validate_submission(submission)
    submission_path = RESULTS_DIR / "higgs_submission_cv_selected_tuned_rank_20_80.csv"
    submission.to_csv(submission_path, index=False)
    summary_path = RESULTS_DIR / "02_cv_selected_submission_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "# CV-selected tuned rank 20/80 submission",
                "",
                f"Blend key: {selected['blend_key']}",
                f"Local AMS on original split: {float(selected['ams']):.6f}",
                f"Local AUC on original split: {float(selected['auc']):.6f}",
                f"Threshold: {threshold:.6f}",
                f"Signal rows: {validation['signal_count']}",
                f"Background rows: {validation['background_count']}",
                f"Submission: {submission_path}",
            ]
        )
        + "\n"
    )
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
