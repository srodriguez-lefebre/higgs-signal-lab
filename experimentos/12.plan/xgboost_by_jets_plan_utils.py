from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

PLAN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAN_DIR.parent
MIX_DIR = PROJECT_ROOT / "11.Mezcla"
RESULTS_DIR = PLAN_DIR / "results"
MODELS_DIR = PLAN_DIR / "models"

if str(MIX_DIR) not in sys.path:
    sys.path.insert(0, str(MIX_DIR))

from i3_tree_safe_mix import get_higgs_data_dir  # noqa: E402
from i3_tree_safe_mix_by_jets import JET_GROUPS, run_i3_tree_safe_mix_by_jets  # noqa: E402
from i3_tree_safe_mix_by_jets_xgboost import (  # noqa: E402
    BEST_XGB_PARAMS,
    ams_score,
    build_higgs_submission,
    save_xgboost_by_jets_bundle,
    validate_higgs_submission,
)


def ensure_plan_dirs() -> None:
    """Creates the output directories used by the plan scripts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: object) -> object:
    """Converts numpy/pandas scalars recursively so json.dump is stable."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    return value


def dump_json(path: Path, payload: object) -> None:
    """Writes JSON with stable formatting."""
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, object]:
    """Loads a JSON object from disk."""
    return json.loads(path.read_text())


def label_to_binary(labels: pd.Series) -> pd.Series:
    """Maps Higgs labels to 0/1."""
    return labels.map({"b": 0, "s": 1}).astype(np.int8)


def get_artifacts() -> dict[str, object]:
    """Builds the fixed preprocessed data for the plan."""
    return run_i3_tree_safe_mix_by_jets(project_root=PROJECT_ROOT)


def class_ratio_for_group(group: dict[str, object]) -> float:
    """Returns negative/positive ratio for scale_pos_weight candidates."""
    y_train = label_to_binary(group["y_train"])
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if positive_count == 0:
        return float(BEST_XGB_PARAMS["scale_pos_weight"])
    return negative_count / positive_count


def make_params(overrides: dict[str, object] | None = None) -> dict[str, object]:
    """Creates an XGBoost parameter dictionary from the shared baseline."""
    params = dict(BEST_XGB_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def fit_xgb_group(
    group: dict[str, object],
    params: dict[str, object],
) -> tuple[XGBClassifier, np.ndarray, float]:
    """Fits one XGBoost model for a jet group and returns validation scores."""
    model = XGBClassifier(**params)
    y_train = label_to_binary(group["y_train"])
    start_time = time.perf_counter()
    model.fit(group["X_train_processed"], y_train)
    fit_seconds = time.perf_counter() - start_time
    valid_scores = model.predict_proba(group["X_valid_processed"])[:, 1]
    return model, valid_scores, fit_seconds


def cumulative_table_for_scores(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    weights: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Builds cumulative signal/background weights for descending thresholds."""
    y_array = np.asarray(y_true, dtype=np.int8)
    score_array = np.asarray(scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    order = np.argsort(score_array)[::-1]
    y_sorted = y_array[order]
    score_sorted = score_array[order]
    weight_sorted = weight_array[order]

    signal_cumsum = np.cumsum(np.where(y_sorted == 1, weight_sorted, 0.0))
    background_cumsum = np.cumsum(np.where(y_sorted == 0, weight_sorted, 0.0))
    candidate_mask = np.r_[score_sorted[1:] != score_sorted[:-1], True]
    candidate_indices = np.flatnonzero(candidate_mask)

    return pd.DataFrame(
        {
            "threshold": score_sorted[candidate_indices],
            "signal_weight": signal_cumsum[candidate_indices],
            "background_weight": background_cumsum[candidate_indices],
            "predicted_signal_count": candidate_indices + 1,
        }
    )


def best_threshold_from_table(table: pd.DataFrame) -> dict[str, float | int]:
    """Finds the best AMS row from a cumulative threshold table."""
    ams_values = ams_score(table["signal_weight"].to_numpy(), table["background_weight"].to_numpy())
    best_idx = int(np.argmax(ams_values))
    row = table.iloc[best_idx]
    return {
        "threshold": float(row["threshold"]),
        "ams": float(ams_values[best_idx]),
        "signal_weight": float(row["signal_weight"]),
        "background_weight": float(row["background_weight"]),
        "predicted_signal_count": int(row["predicted_signal_count"]),
    }


def metrics_from_thresholds(
    valid_predictions_df: pd.DataFrame,
    thresholds_by_jet: dict[int, float] | None = None,
    global_threshold: float | None = None,
    label: str = "policy",
) -> dict[str, float | int | str]:
    """Evaluates validation predictions using either one global threshold or thresholds by jet."""
    if thresholds_by_jet is None and global_threshold is None:
        raise ValueError("Se requiere global_threshold o thresholds_by_jet.")

    scores = valid_predictions_df["score"].to_numpy(dtype=np.float64)
    y_true = valid_predictions_df["y_true"].to_numpy(dtype=np.int8)
    weights = valid_predictions_df["weight"].to_numpy(dtype=np.float64)

    if thresholds_by_jet is not None:
        thresholds = valid_predictions_df["jet_num"].map(thresholds_by_jet).to_numpy(dtype=np.float64)
        y_pred = (scores >= thresholds).astype(np.int8)
        threshold_repr = json.dumps({str(key): value for key, value in thresholds_by_jet.items()}, sort_keys=True)
    else:
        y_pred = (scores >= float(global_threshold)).astype(np.int8)
        threshold_repr = f"{float(global_threshold):.12f}"

    signal_weight = float(weights[(y_pred == 1) & (y_true == 1)].sum())
    background_weight = float(weights[(y_pred == 1) & (y_true == 0)].sum())

    return {
        "policy": label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, scores)),
        "ams": float(ams_score(signal_weight, background_weight)),
        "signal_weight": signal_weight,
        "background_weight": background_weight,
        "predicted_signal_count": int(y_pred.sum()),
        "thresholds": threshold_repr,
    }


def build_validation_predictions(
    artifacts: dict[str, object],
    models_by_jet: dict[int, XGBClassifier],
) -> pd.DataFrame:
    """Scores validation rows with their jet-specific model and restores global order."""
    parts: list[pd.DataFrame] = []
    for jet_value in JET_GROUPS:
        group = artifacts["group_artifacts"][jet_value]
        scores = models_by_jet[jet_value].predict_proba(group["X_valid_processed"])[:, 1]
        parts.append(
            pd.DataFrame(
                {
                    "EventId": group["valid_event_id"].astype(np.int64),
                    "score": scores,
                    "jet_num": int(jet_value),
                    "y_true": label_to_binary(group["y_valid"]).to_numpy(),
                    "weight": group["valid_weight"].astype(np.float64).to_numpy(),
                }
            )
        )

    valid_predictions_df = pd.concat(parts, ignore_index=True)
    global_order = artifacts["global_valid_event_id"].astype(np.int64)
    return valid_predictions_df.set_index("EventId").loc[global_order].reset_index()


def build_test_predictions(
    artifacts: dict[str, object],
    models_by_jet: dict[int, XGBClassifier],
) -> pd.DataFrame:
    """Scores test rows with their jet-specific model and restores global order."""
    parts: list[pd.DataFrame] = []
    for jet_value in JET_GROUPS:
        group = artifacts["group_artifacts"][jet_value]
        scores = models_by_jet[jet_value].predict_proba(group["X_test_processed"])[:, 1]
        parts.append(
            pd.DataFrame(
                {
                    "EventId": group["test_event_id"].astype(np.int64),
                    "score": scores,
                    "jet_num": int(jet_value),
                }
            )
        )

    test_predictions_df = pd.concat(parts, ignore_index=True)
    global_order = artifacts["global_test_event_id"].astype(np.int64)
    return test_predictions_df.set_index("EventId").loc[global_order].reset_index()


def threshold_tables_by_jet(valid_predictions_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Builds cumulative threshold tables for every jet."""
    tables: dict[int, pd.DataFrame] = {}
    for jet_value in JET_GROUPS:
        subset = valid_predictions_df.loc[valid_predictions_df["jet_num"] == jet_value]
        tables[jet_value] = cumulative_table_for_scores(
            y_true=subset["y_true"],
            scores=subset["score"].to_numpy(),
            weights=subset["weight"],
        )
    return tables


def best_global_threshold(valid_predictions_df: pd.DataFrame) -> dict[str, float | int]:
    """Finds the best single threshold across all routed validation rows."""
    table = cumulative_table_for_scores(
        y_true=valid_predictions_df["y_true"],
        scores=valid_predictions_df["score"].to_numpy(),
        weights=valid_predictions_df["weight"],
    )
    return best_threshold_from_table(table)


def independent_thresholds_by_jet(valid_predictions_df: pd.DataFrame) -> dict[int, float]:
    """Finds the best threshold inside each jet independently."""
    tables = threshold_tables_by_jet(valid_predictions_df)
    return {jet_value: float(best_threshold_from_table(table)["threshold"]) for jet_value, table in tables.items()}


def optimize_joint_thresholds(
    valid_predictions_df: pd.DataFrame,
    starting_thresholds: dict[int, float] | None = None,
    max_iter: int = 8,
) -> tuple[dict[int, float], pd.DataFrame]:
    """Optimizes four jet thresholds with coordinate descent against global AMS."""
    tables = threshold_tables_by_jet(valid_predictions_df)
    if starting_thresholds is None:
        global_info = best_global_threshold(valid_predictions_df)
        starting_thresholds = {jet_value: float(global_info["threshold"]) for jet_value in JET_GROUPS}

    thresholds = dict(starting_thresholds)
    trace_rows: list[dict[str, float | int | str]] = []

    for iteration in range(max_iter):
        changed = False
        for jet_value in JET_GROUPS:
            fixed_weights = {"signal": 0.0, "background": 0.0}
            for other_jet in JET_GROUPS:
                if other_jet == jet_value:
                    continue
                other_table = tables[other_jet]
                other_threshold = thresholds[other_jet]
                eligible = other_table.loc[other_table["threshold"] >= other_threshold]
                if eligible.empty:
                    continue
                row = eligible.iloc[-1]
                fixed_weights["signal"] += float(row["signal_weight"])
                fixed_weights["background"] += float(row["background_weight"])

            candidate_table = tables[jet_value]
            signal = fixed_weights["signal"] + candidate_table["signal_weight"].to_numpy(dtype=np.float64)
            background = fixed_weights["background"] + candidate_table["background_weight"].to_numpy(dtype=np.float64)
            ams_values = ams_score(signal, background)
            best_idx = int(np.argmax(ams_values))
            old_threshold = thresholds[jet_value]
            new_threshold = float(candidate_table.iloc[best_idx]["threshold"])
            thresholds[jet_value] = new_threshold
            changed = changed or not np.isclose(old_threshold, new_threshold)
            trace_rows.append(
                {
                    "iteration": iteration,
                    "jet_num": jet_value,
                    "old_threshold": float(old_threshold),
                    "new_threshold": new_threshold,
                    "ams_after_step": float(ams_values[best_idx]),
                }
            )

        if not changed:
            break

    return thresholds, pd.DataFrame(trace_rows)


def threshold_policy_comparison(valid_predictions_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compares global, independent-by-jet and jointly optimized threshold policies."""
    global_info = best_global_threshold(valid_predictions_df)
    independent_thresholds = independent_thresholds_by_jet(valid_predictions_df)
    joint_from_global, trace_global = optimize_joint_thresholds(
        valid_predictions_df,
        starting_thresholds={jet_value: float(global_info["threshold"]) for jet_value in JET_GROUPS},
    )
    joint_from_independent, trace_independent = optimize_joint_thresholds(
        valid_predictions_df,
        starting_thresholds=independent_thresholds,
    )

    rows = [
        metrics_from_thresholds(
            valid_predictions_df,
            global_threshold=float(global_info["threshold"]),
            label="global_single_threshold",
        ),
        metrics_from_thresholds(
            valid_predictions_df,
            thresholds_by_jet=independent_thresholds,
            label="independent_threshold_by_jet",
        ),
        metrics_from_thresholds(
            valid_predictions_df,
            thresholds_by_jet=joint_from_global,
            label="joint_thresholds_from_global",
        ),
        metrics_from_thresholds(
            valid_predictions_df,
            thresholds_by_jet=joint_from_independent,
            label="joint_thresholds_from_independent",
        ),
    ]
    comparison_df = pd.DataFrame(rows).sort_values("ams", ascending=False).reset_index(drop=True)

    details = {
        "global_threshold": float(global_info["threshold"]),
        "independent_thresholds": independent_thresholds,
        "joint_thresholds_from_global": joint_from_global,
        "joint_thresholds_from_independent": joint_from_independent,
        "trace_from_global": trace_global.to_dict(orient="records"),
        "trace_from_independent": trace_independent.to_dict(orient="records"),
        "best_policy": comparison_df.iloc[0].to_dict(),
    }
    return comparison_df, details


def train_models_from_params(
    artifacts: dict[str, object],
    params_by_jet: dict[int, dict[str, object]],
) -> tuple[dict[int, XGBClassifier], pd.DataFrame]:
    """Trains one XGBoost model per jet using explicit params."""
    models: dict[int, XGBClassifier] = {}
    rows: list[dict[str, float | int | str]] = []
    for jet_value in JET_GROUPS:
        group = artifacts["group_artifacts"][jet_value]
        model, scores, fit_seconds = fit_xgb_group(group, params_by_jet[jet_value])
        models[jet_value] = model
        y_valid = label_to_binary(group["y_valid"])
        valid_pred_default = model.predict(group["X_valid_processed"])
        table = cumulative_table_for_scores(y_valid, scores, group["valid_weight"])
        best_threshold = best_threshold_from_table(table)
        rows.append(
            {
                "jet_num": int(jet_value),
                "train_rows": int(len(group["X_train_processed"])),
                "valid_rows": int(len(group["X_valid_processed"])),
                "feature_count": int(group["X_train_processed"].shape[1]),
                "accuracy_default": float(accuracy_score(y_valid, valid_pred_default)),
                "precision_default": float(precision_score(y_valid, valid_pred_default, zero_division=0)),
                "recall_default": float(recall_score(y_valid, valid_pred_default, zero_division=0)),
                "f1_default": float(f1_score(y_valid, valid_pred_default, zero_division=0)),
                "auc": float(roc_auc_score(y_valid, scores)),
                "ams": float(best_threshold["ams"]),
                "ams_threshold": float(best_threshold["threshold"]),
                "fit_seconds": float(fit_seconds),
            }
        )
    return models, pd.DataFrame(rows)


def baseline_params_by_jet() -> dict[int, dict[str, object]]:
    """Returns the shared baseline params for every jet."""
    return {jet_value: make_params() for jet_value in JET_GROUPS}


def load_best_params_by_jet(path: Path) -> dict[int, dict[str, object]]:
    """Loads best params produced by the tuning script."""
    raw = load_json(path)
    return {int(jet_value): dict(params) for jet_value, params in raw["best_params_by_jet"].items()}


def save_final_artifacts(
    artifacts: dict[str, object],
    models_by_jet: dict[int, XGBClassifier],
    group_metrics_df: pd.DataFrame,
    overall_metrics_df: pd.DataFrame,
    threshold_details: dict[str, object],
    params_by_jet: dict[int, dict[str, object]],
    submission_label: str,
) -> dict[str, Path]:
    """Saves the final bundle, predictions, metrics and Kaggle submission."""
    ensure_plan_dirs()
    bundle_dir = MODELS_DIR / f"{submission_label}_bundle"
    training_outputs = {
        "group_models": models_by_jet,
        "group_metrics_df": group_metrics_df,
        "overall_metrics_df": overall_metrics_df,
        "global_threshold_info": {
            "ams_threshold": float(threshold_details["selected_thresholds"].get("global", np.nan)),
            "ams": float(overall_metrics_df.iloc[0]["ams"]),
        },
    }
    save_xgboost_by_jets_bundle(
        artifacts=artifacts,
        training_outputs=training_outputs,
        output_dir=bundle_dir,
        xgb_params={"params_by_jet": params_by_jet},
    )

    test_predictions_df = build_test_predictions(artifacts, models_by_jet)
    selected_thresholds = threshold_details["selected_thresholds"]
    if "global" in selected_thresholds:
        class_threshold = float(selected_thresholds["global"])
        submission_df = build_higgs_submission(
            event_ids=test_predictions_df["EventId"],
            scores=test_predictions_df["score"].to_numpy(),
            threshold=class_threshold,
        )
    else:
        thresholds_by_jet = {int(key): float(value) for key, value in selected_thresholds.items()}
        threshold_array = test_predictions_df["jet_num"].map(thresholds_by_jet).to_numpy(dtype=np.float64)
        score_array = test_predictions_df["score"].to_numpy(dtype=np.float64)
        submission_df = pd.DataFrame(
            {
                "EventId": test_predictions_df["EventId"].astype(np.int64),
                "RankOrder": pd.Series(score_array).rank(method="first", ascending=True).astype(np.int64),
                "Class": np.where(score_array >= threshold_array, "s", "b"),
            }
        )[["EventId", "RankOrder", "Class"]]

    validation = validate_higgs_submission(submission_df)
    submission_path = RESULTS_DIR / f"higgs_submission_{submission_label}.csv"
    submission_df.to_csv(submission_path, index=False)

    return {
        "bundle_dir": bundle_dir,
        "submission_path": submission_path,
        "test_predictions_path": RESULTS_DIR / f"{submission_label}_test_scores.csv",
        "validation_path": RESULTS_DIR / f"{submission_label}_submission_validation.json",
    } | _save_prediction_artifacts(test_predictions_df, validation, submission_label)


def _save_prediction_artifacts(
    test_predictions_df: pd.DataFrame,
    validation: dict[str, int],
    submission_label: str,
) -> dict[str, Path]:
    test_predictions_path = RESULTS_DIR / f"{submission_label}_test_scores.csv"
    validation_path = RESULTS_DIR / f"{submission_label}_submission_validation.json"
    test_predictions_df.to_csv(test_predictions_path, index=False)
    dump_json(validation_path, validation)
    return {
        "test_predictions_path": test_predictions_path,
        "validation_path": validation_path,
    }


def reference_best_tree_row() -> pd.Series:
    """Loads the previous best tree row from 10.try/06."""
    reference_path = PROJECT_ROOT / "10.try" / "results" / "best_nn_vs_best_tree_comparison.csv"
    reference_df = pd.read_csv(reference_path)
    return reference_df.loc[reference_df["model_family"] == "tree"].iloc[0]


def selected_thresholds_from_policy(policy_row: pd.Series) -> dict[str, float]:
    """Parses the threshold representation from a policy comparison row."""
    policy = str(policy_row["policy"])
    raw_thresholds = str(policy_row["thresholds"])
    if policy == "global_single_threshold":
        return {"global": float(raw_thresholds)}
    parsed = json.loads(raw_thresholds)
    return {str(key): float(value) for key, value in parsed.items()}
