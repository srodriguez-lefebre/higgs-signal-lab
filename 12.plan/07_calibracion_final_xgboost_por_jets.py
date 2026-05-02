from __future__ import annotations

from xgboost_by_jets_plan_utils import (
    RESULTS_DIR,
    build_validation_predictions,
    dump_json,
    ensure_plan_dirs,
    get_artifacts,
    load_best_params_by_jet,
    threshold_policy_comparison,
    train_models_from_params,
)


def main() -> None:
    ensure_plan_dirs()
    best_params_path = RESULTS_DIR / "06_tuning_xgboost_por_jets_best_params.json"
    params_by_jet = load_best_params_by_jet(best_params_path)

    artifacts = get_artifacts()
    models_by_jet, group_metrics_df = train_models_from_params(artifacts, params_by_jet)
    valid_predictions_df = build_validation_predictions(artifacts, models_by_jet)
    comparison_df, threshold_details = threshold_policy_comparison(valid_predictions_df)

    group_metrics_df.to_csv(RESULTS_DIR / "07_calibracion_final_group_metrics.csv", index=False)
    valid_predictions_df.to_csv(RESULTS_DIR / "07_calibracion_final_valid_scores.csv", index=False)
    comparison_df.to_csv(RESULTS_DIR / "07_calibracion_final_policy_comparison.csv", index=False)
    dump_json(RESULTS_DIR / "07_calibracion_final_threshold_details.json", threshold_details)

    print("Comparacion final de politicas de threshold sobre modelos tuneados:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
