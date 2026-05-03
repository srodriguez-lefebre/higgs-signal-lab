from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from final_methods_utils import (
    TOP_SIGNAL_PERCENT,
    build_top_percent_submission,
    find_project_root,
    results_dir,
)


TOP_PERCENT_VARIANTS = [13.5, 14.5, 13.0, 15.0]


def latest_full_train_scores_path(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("p04_xgboost_sw04_full_train_test_scores_*.csv"))
    if not candidates:
        raise FileNotFoundError("No encontre scores full train en results.")
    return candidates[-1]


def variant_label(percent: float) -> str:
    return f"top{str(percent).replace('.', 'p')}pct"


def main() -> None:
    output_dir = results_dir(find_project_root())
    scores_path = latest_full_train_scores_path(output_dir)
    scores_df = pd.read_csv(scores_path)
    if list(scores_df.columns) != ["EventId", "score"]:
        raise ValueError(f"Columnas inesperadas en scores full train: {list(scores_df.columns)}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"scores_path={scores_path}")
    print(f"base_top_signal_percent={TOP_SIGNAL_PERCENT:.2f}")

    for percent in TOP_PERCENT_VARIANTS:
        submission, validation = build_top_percent_submission(
            event_id=scores_df["EventId"],
            scores=scores_df["score"].to_numpy(),
            top_signal_percent=percent,
        )
        output_path = output_dir / (
            f"higgs_submission_p04_xgboost_sw04_full_train_{variant_label(percent)}_{run_timestamp}.csv"
        )
        submission.to_csv(output_path, index=False)
        print(f"submission_path={output_path}")
        print(f"top_signal_percent={percent:.2f}")
        print(f"signal_count={validation['signal_count']}")
        print(f"background_count={validation['background_count']}")
        print(f"signal_percent={validation['signal_percent']:.6f}")


if __name__ == "__main__":
    main()
