from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "17.cv_jets" / "results"


def ams_score(signal_weight: np.ndarray | float, background_weight: np.ndarray | float, br: float = 10.0) -> np.ndarray:
    signal_arr = np.asarray(signal_weight, dtype=np.float64)
    background_arr = np.asarray(background_weight, dtype=np.float64)
    radicand = 2.0 * (
        (signal_arr + background_arr + br) * np.log1p(signal_arr / (background_arr + br)) - signal_arr
    )
    return np.sqrt(np.clip(radicand, a_min=0.0, a_max=None))


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    score_sorted = scores[order]
    weight_sorted = weights[order]
    signal_cumsum = np.cumsum(np.where(y_sorted == 1, weight_sorted, 0.0))
    background_cumsum = np.cumsum(np.where(y_sorted == 0, weight_sorted, 0.0))
    candidate_mask = np.r_[score_sorted[1:] != score_sorted[:-1], True]
    candidate_indices = np.flatnonzero(candidate_mask)
    values = ams_score(signal_cumsum[candidate_indices], background_cumsum[candidate_indices])
    best_pos = int(np.argmax(values))
    best_idx = int(candidate_indices[best_pos])
    y_pred = scores >= float(score_sorted[best_idx])
    return {
        "threshold": float(score_sorted[best_idx]),
        "ams": float(values[best_pos]),
        "signal_weight": float(signal_cumsum[best_idx]),
        "background_weight": float(background_cumsum[best_idx]),
        "predicted_signal_count": int(y_pred.sum()),
    }


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


def rank(values: pd.Series) -> np.ndarray:
    return values.rank(method="first", pct=True).to_numpy(dtype=np.float64)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comb_valid = pd.read_csv(PROJECT_ROOT / "13.comb" / "results" / "01_valid_scores.csv")
    comb_test = pd.read_csv(PROJECT_ROOT / "13.comb" / "results" / "01_test_scores.csv")
    jets_valid = pd.read_csv(PROJECT_ROOT / "14.invest" / "results" / "01_valid_scores.csv")
    jets_test = pd.read_csv(PROJECT_ROOT / "14.invest" / "results" / "01_test_scores.csv")

    if not np.array_equal(comb_valid["EventId"].to_numpy(), jets_valid["EventId"].to_numpy()):
        raise ValueError("Valid EventId no coincide.")
    if not np.array_equal(comb_test["EventId"].to_numpy(), jets_test["EventId"].to_numpy()):
        raise ValueError("Test EventId no coincide.")

    y_true = comb_valid["y_true"].to_numpy(dtype=np.int8)
    weights = comb_valid["weight"].to_numpy(dtype=np.float64)
    i3_valid_rank = rank(comb_valid["i3_tree_safe_mix_score"])
    v2_valid_rank = rank(comb_valid["v2_physics_by_jet_score"])
    jets_valid_rank = rank(jets_valid["score"])
    i3_test_rank = rank(comb_test["i3_tree_safe_mix_score"])
    v2_test_rank = rank(comb_test["v2_physics_by_jet_score"])
    jets_test_rank = rank(jets_test["score"])

    rows: list[dict[str, object]] = []
    for jets_weight in np.linspace(0.0, 0.25, 26):
        remaining = 1.0 - float(jets_weight)
        for i3_share in np.linspace(0.0, 0.6, 61):
            i3_weight = remaining * float(i3_share)
            v2_weight = remaining - i3_weight
            valid_score = i3_weight * i3_valid_rank + v2_weight * v2_valid_rank + float(jets_weight) * jets_valid_rank
            test_score = i3_weight * i3_test_rank + v2_weight * v2_test_rank + float(jets_weight) * jets_test_rank
            info = find_best_threshold(y_true, valid_score, weights)
            key = f"rank_i3_{i3_weight:.3f}_v2_{v2_weight:.3f}_jets_{float(jets_weight):.3f}"
            rows.append(
                {
                    "blend_key": key,
                    "i3_weight": i3_weight,
                    "v2_weight": v2_weight,
                    "jets_weight": float(jets_weight),
                    **info,
                }
            )

    sweep = pd.DataFrame(rows).sort_values("ams", ascending=False).reset_index(drop=True)
    best = sweep.iloc[0]
    best_key = str(best["blend_key"])
    i3_weight = float(best["i3_weight"])
    v2_weight = float(best["v2_weight"])
    jets_weight = float(best["jets_weight"])
    threshold = float(best["threshold"])
    test_score = i3_weight * i3_test_rank + v2_weight * v2_test_rank + jets_weight * jets_test_rank
    submission = pd.DataFrame(
        {
            "EventId": comb_test["EventId"].astype(np.int64),
            "RankOrder": pd.Series(test_score).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(test_score >= threshold, "s", "b"),
        }
    )[["EventId", "RankOrder", "Class"]]
    validation = validate_submission(submission)
    sweep.to_csv(RESULTS_DIR / "02_ensemble_with_weighted_jets_sweep.csv", index=False)
    submission_path = RESULTS_DIR / "higgs_submission_best_plus_weighted_jets.csv"
    submission.to_csv(submission_path, index=False)
    (RESULTS_DIR / "02_ensemble_with_weighted_jets_summary.txt").write_text(
        "\n".join(
            [
                "# Ensemble best + weighted jets",
                "",
                f"Best blend: {best_key}",
                f"Local AMS: {float(best['ams']):.6f}",
                f"Threshold: {threshold:.6f}",
                f"Signal rows: {validation['signal_count']}",
                f"Background rows: {validation['background_count']}",
                f"Submission: {submission_path}",
            ]
        )
        + "\n"
    )
    (RESULTS_DIR / "02_ensemble_with_weighted_jets_metadata.json").write_text(
        json.dumps({"best": best.to_dict(), "submission_validation": validation}, indent=2, ensure_ascii=False)
        + "\n"
    )
    print(sweep.head(15).to_string(index=False))
    print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
