from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
from sklearn.metrics import roc_auc_score

from xgboost_by_jets_plan_utils import (
    BEST_XGB_PARAMS,
    JET_GROUPS,
    RESULTS_DIR,
    class_ratio_for_group,
    cumulative_table_for_scores,
    dump_json,
    ensure_plan_dirs,
    fit_xgb_group,
    get_artifacts,
    label_to_binary,
    make_params,
    best_threshold_from_table,
)


def candidate_stages(group_ratio: float) -> list[tuple[str, list[dict[str, object]]]]:
    """Returns small, local tuning stages around the current baseline."""
    baseline_spw = float(BEST_XGB_PARAMS["scale_pos_weight"])
    midpoint_spw = (baseline_spw + group_ratio) / 2.0
    return [
        (
            "class_weight",
            [
                {"scale_pos_weight": group_ratio},
                {"scale_pos_weight": midpoint_spw},
                {"scale_pos_weight": 1.0},
            ],
        ),
        (
            "structure",
            [
                {"max_depth": 4},
                {"max_depth": 5},
                {"max_depth": 7},
                {"min_child_weight": 3},
                {"min_child_weight": 5},
                {"gamma": 0.1},
                {"gamma": 0.3},
            ],
        ),
        (
            "capacity",
            [
                {"learning_rate": 0.03, "n_estimators": 650},
                {"learning_rate": 0.04, "n_estimators": 500},
                {"learning_rate": 0.07, "n_estimators": 320},
                {"learning_rate": 0.10, "n_estimators": 240},
            ],
        ),
        (
            "sampling",
            [
                {"subsample": 0.8, "colsample_bytree": 0.9},
                {"subsample": 0.9, "colsample_bytree": 0.8},
                {"subsample": 0.8, "colsample_bytree": 0.8},
                {"subsample": 0.7, "colsample_bytree": 0.9},
            ],
        ),
        (
            "regularization",
            [
                {"reg_lambda": 2.0},
                {"reg_lambda": 5.0},
                {"reg_lambda": 10.0},
                {"reg_alpha": 0.05},
                {"reg_alpha": 0.1},
                {"reg_alpha": 0.5},
            ],
        ),
    ]


def evaluate_candidate(
    group: dict[str, object],
    params: dict[str, object],
    jet_value: int,
    stage: str,
    candidate_name: str,
) -> dict[str, object]:
    """Fits and evaluates a single candidate for one jet."""
    _, valid_scores, fit_seconds = fit_xgb_group(group, params)
    y_valid = label_to_binary(group["y_valid"])
    table = cumulative_table_for_scores(y_valid, valid_scores, group["valid_weight"])
    best_threshold = best_threshold_from_table(table)
    return {
        "jet_num": jet_value,
        "stage": stage,
        "candidate": candidate_name,
        "auc": float(roc_auc_score(y_valid, valid_scores)),
        "roc_auc": float(roc_auc_score(y_valid, valid_scores)),
        "ams": float(best_threshold["ams"]),
        "ams_threshold": float(best_threshold["threshold"]),
        "fit_seconds": float(fit_seconds),
        "params_json": json.dumps(params, sort_keys=True),
    }


def tune_one_jet(group: dict[str, object], jet_value: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Runs cautious staged tuning for one jet and returns the best params plus all rows."""
    group_ratio = class_ratio_for_group(group)
    current_params = make_params()
    rows: list[dict[str, object]] = []

    baseline_row = evaluate_candidate(group, current_params, jet_value, "baseline", "shared_baseline")
    rows.append(baseline_row)
    current_best_ams = float(baseline_row["ams"])

    for stage, overrides_list in candidate_stages(group_ratio):
        stage_rows: list[dict[str, object]] = []
        for overrides in overrides_list:
            candidate_params = deepcopy(current_params)
            candidate_params.update(overrides)
            candidate_name = ",".join(f"{key}={value}" for key, value in sorted(overrides.items()))
            row = evaluate_candidate(group, candidate_params, jet_value, stage, candidate_name)
            rows.append(row)
            stage_rows.append(row)

        best_stage_row = max(stage_rows, key=lambda row: float(row["ams"]))
        if float(best_stage_row["ams"]) > current_best_ams:
            current_params = json.loads(str(best_stage_row["params_json"]))
            current_best_ams = float(best_stage_row["ams"])

    return current_params, rows


def main() -> None:
    ensure_plan_dirs()
    artifacts = get_artifacts()
    all_rows: list[dict[str, object]] = []
    best_params_by_jet: dict[int, dict[str, object]] = {}

    for jet_value in JET_GROUPS:
        group = artifacts["group_artifacts"][jet_value]
        best_params, rows = tune_one_jet(group, jet_value)
        best_params_by_jet[jet_value] = best_params
        all_rows.extend(rows)
        best_row = max((row for row in rows), key=lambda row: float(row["ams"]))
        print(
            f"jet={jet_value} best_ams={float(best_row['ams']):.6f} "
            f"best_candidate={best_row['stage']}:{best_row['candidate']}"
        )

    runs_df = pd.DataFrame(all_rows)
    best_rows_df = (
        runs_df.sort_values(["jet_num", "ams"], ascending=[True, False])
        .groupby("jet_num", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    runs_df.to_csv(RESULTS_DIR / "06_tuning_xgboost_por_jets_runs.csv", index=False)
    best_rows_df.to_csv(RESULTS_DIR / "06_tuning_xgboost_por_jets_best_rows.csv", index=False)
    dump_json(
        RESULTS_DIR / "06_tuning_xgboost_por_jets_best_params.json",
        {
            "best_params_by_jet": best_params_by_jet,
            "selection_metric": "per_jet_ams",
            "note": "Staged cautious search around the previous shared XGBoost baseline.",
        },
    )

    print("Mejores filas por jet:")
    print(best_rows_df[["jet_num", "stage", "candidate", "roc_auc", "ams", "ams_threshold"]].to_string(index=False))


if __name__ == "__main__":
    main()
