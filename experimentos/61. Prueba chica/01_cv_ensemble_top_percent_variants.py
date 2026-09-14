from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "60. metodos completos"))

from final_methods_utils import build_top_percent_submission  # noqa: E402


def latest_cv_ensemble_scores_path() -> Path:
    source_dir = PROJECT_ROOT / "60. metodos completos" / "results"
    candidates = sorted(source_dir.glob("p04_xgboost_sw04_cv_ensemble_test_scores_*.csv"))
    if not candidates:
        raise FileNotFoundError("No encontre scores CV ensemble en 60. metodos completos/results.")
    return candidates[-1]


def variant_label(percent: float) -> str:
    return f"top{str(percent).replace('.', 'p')}pct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "percentages",
        nargs="+",
        type=float,
        help="Porcentajes top de score a marcar como signal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = CURRENT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = latest_cv_ensemble_scores_path()
    scores_df = pd.read_csv(scores_path)
    required_columns = {"EventId", "score_mean"}
    missing = sorted(required_columns.difference(scores_df.columns))
    if missing:
        raise ValueError(f"Faltan columnas en scores CV ensemble: {missing}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"scores_path={scores_path}")

    for percent in args.percentages:
        submission, validation = build_top_percent_submission(
            event_id=scores_df["EventId"],
            scores=scores_df["score_mean"].to_numpy(),
            top_signal_percent=percent,
        )
        output_path = output_dir / (
            f"higgs_submission_p04_xgboost_sw04_cv_ensemble_{variant_label(percent)}_{run_timestamp}.csv"
        )
        submission.to_csv(output_path, index=False)
        print(f"submission_path={output_path}")
        print(f"top_signal_percent={percent:.2f}")
        print(f"signal_count={validation['signal_count']}")
        print(f"background_count={validation['background_count']}")
        print(f"signal_percent={validation['signal_percent']:.6f}")


if __name__ == "__main__":
    main()
