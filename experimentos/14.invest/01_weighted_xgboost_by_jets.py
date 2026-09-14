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


SEED = 42
JET_GROUPS = [0, 1, 2, 3]
BASE_XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": SEED,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "max_depth": 6,
    "scale_pos_weight": 1.5346,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "n_jobs": -1,
    "missing": np.nan,
}

INVEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INVEST_DIR.parent
MIX_DIR = PROJECT_ROOT / "11.Mezcla"
RESULTS_DIR = INVEST_DIR / "results"
MODELS_DIR = INVEST_DIR / "models"

if str(MIX_DIR) not in sys.path:
    sys.path.insert(0, str(MIX_DIR))

from i3_tree_safe_mix_by_jets import run_i3_tree_safe_mix_by_jets  # noqa: E402


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: object) -> object:
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
    if isinstance(value, Path):
        return str(value)
    return value


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def label_to_binary(labels: pd.Series) -> pd.Series:
    return labels.map({"b": 0, "s": 1}).astype(np.int8)


def ams_score(signal_weight: np.ndarray | float, background_weight: np.ndarray | float, br: float = 10.0) -> np.ndarray:
    signal_arr = np.asarray(signal_weight, dtype=np.float64)
    background_arr = np.asarray(background_weight, dtype=np.float64)
    radicand = 2.0 * (
        (signal_arr + background_arr + br) * np.log1p(signal_arr / (background_arr + br)) - signal_arr
    )
    return np.sqrt(np.clip(radicand, a_min=0.0, a_max=None))


def cumulative_table(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    weights: pd.Series | np.ndarray,
) -> pd.DataFrame:
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


def metrics_for_scores(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    weights: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    y_array = np.asarray(y_true, dtype=np.int8)
    table = cumulative_table(y_array, scores, weights)
    best = best_threshold_from_table(table)
    y_pred = (np.asarray(scores, dtype=np.float64) >= float(best["threshold"])).astype(np.int8)
    return {
        **best,
        "accuracy": float(accuracy_score(y_array, y_pred)),
        "precision": float(precision_score(y_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_array, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_array, scores)),
    }


def group_ratios(group: dict[str, object]) -> dict[str, float]:
    y_train = label_to_binary(group["y_train"])
    train_weight = group["train_weight"].astype(np.float64)
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())
    pos_weight = float(train_weight.loc[pos_mask].sum())
    neg_weight = float(train_weight.loc[neg_mask].sum())
    return {
        "count_ratio": neg_count / pos_count,
        "weight_ratio": neg_weight / pos_weight,
    }


def candidate_variants(group: dict[str, object]) -> list[dict[str, object]]:
    ratios = group_ratios(group)
    count_ratio = ratios["count_ratio"]
    weight_ratio = ratios["weight_ratio"]
    return [
        {
            "variant": "unweighted_shared_baseline",
            "use_sample_weight": False,
            "overrides": {"scale_pos_weight": 1.5346},
        },
        {
            "variant": "sample_weight_only",
            "use_sample_weight": True,
            "overrides": {"scale_pos_weight": 1.0},
        },
        {
            "variant": "sample_weight_count_spw",
            "use_sample_weight": True,
            "overrides": {"scale_pos_weight": count_ratio},
        },
        {
            "variant": "sample_weight_weight_spw",
            "use_sample_weight": True,
            "overrides": {"scale_pos_weight": weight_ratio},
        },
        {
            "variant": "sample_weight_count_spw_delta1",
            "use_sample_weight": True,
            "overrides": {"scale_pos_weight": count_ratio, "max_delta_step": 1},
        },
        {
            "variant": "sample_weight_weight_spw_delta1",
            "use_sample_weight": True,
            "overrides": {"scale_pos_weight": weight_ratio, "max_delta_step": 1},
        },
        {
            "variant": "sample_weight_only_delta1",
            "use_sample_weight": True,
            "overrides": {"scale_pos_weight": 1.0, "max_delta_step": 1},
        },
    ]


def make_params(overrides: dict[str, object]) -> dict[str, object]:
    params = dict(BASE_XGB_PARAMS)
    params.update(overrides)
    return params


def fit_candidate(
    group: dict[str, object],
    jet_value: int,
    variant: dict[str, object],
) -> tuple[XGBClassifier, dict[str, object]]:
    params = make_params(dict(variant["overrides"]))
    model = XGBClassifier(**params)
    y_train = label_to_binary(group["y_train"])
    y_valid = label_to_binary(group["y_valid"])
    train_weight = group["train_weight"].astype(np.float64)
    valid_weight = group["valid_weight"].astype(np.float64)

    fit_kwargs: dict[str, object] = {}
    if bool(variant["use_sample_weight"]):
        fit_kwargs["sample_weight"] = train_weight

    start = time.perf_counter()
    model.fit(group["X_train_processed"], y_train, **fit_kwargs)
    fit_seconds = time.perf_counter() - start
    valid_scores = model.predict_proba(group["X_valid_processed"])[:, 1]
    metrics = metrics_for_scores(y_valid, valid_scores, valid_weight)
    row = {
        "jet_num": int(jet_value),
        "variant": str(variant["variant"]),
        "use_sample_weight": bool(variant["use_sample_weight"]),
        "params_json": json.dumps(to_jsonable(params), sort_keys=True),
        "fit_seconds": float(fit_seconds),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "feature_count": int(group["X_train_processed"].shape[1]),
        **metrics,
    }
    return model, row


def train_best_models_by_jet(
    artifacts: dict[str, object],
) -> tuple[dict[int, XGBClassifier], pd.DataFrame, pd.DataFrame, dict[int, dict[str, object]]]:
    best_models: dict[int, XGBClassifier] = {}
    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    best_params_by_jet: dict[int, dict[str, object]] = {}

    for jet_value in JET_GROUPS:
        group = artifacts["group_artifacts"][jet_value]
        candidates = candidate_variants(group)
        group_results: list[tuple[XGBClassifier, dict[str, object]]] = []
        for variant in candidates:
            model, row = fit_candidate(group, jet_value, variant)
            all_rows.append(row)
            group_results.append((model, row))

        best_model, best_row = max(group_results, key=lambda item: float(item[1]["ams"]))
        best_models[jet_value] = best_model
        best_rows.append(best_row)
        best_params_by_jet[jet_value] = json.loads(str(best_row["params_json"]))
        print(
            f"jet={jet_value} best_variant={best_row['variant']} "
            f"ams={float(best_row['ams']):.6f} auc={float(best_row['auc']):.6f}"
        )

    return (
        best_models,
        pd.DataFrame(all_rows),
        pd.DataFrame(best_rows).sort_values("jet_num").reset_index(drop=True),
        best_params_by_jet,
    )


def build_validation_predictions(artifacts: dict[str, object], models_by_jet: dict[int, XGBClassifier]) -> pd.DataFrame:
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
    valid_predictions = pd.concat(parts, ignore_index=True)
    global_order = artifacts["global_valid_event_id"].astype(np.int64)
    return valid_predictions.set_index("EventId").loc[global_order].reset_index()


def build_test_predictions(artifacts: dict[str, object], models_by_jet: dict[int, XGBClassifier]) -> pd.DataFrame:
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
    test_predictions = pd.concat(parts, ignore_index=True)
    global_order = artifacts["global_test_event_id"].astype(np.int64)
    return test_predictions.set_index("EventId").loc[global_order].reset_index()


def threshold_tables_by_jet(valid_predictions: pd.DataFrame) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    for jet_value in JET_GROUPS:
        subset = valid_predictions.loc[valid_predictions["jet_num"] == jet_value]
        tables[jet_value] = cumulative_table(subset["y_true"], subset["score"].to_numpy(), subset["weight"])
    return tables


def threshold_row(
    valid_predictions: pd.DataFrame,
    label: str,
    global_threshold: float | None = None,
    thresholds_by_jet: dict[int, float] | None = None,
) -> dict[str, object]:
    scores = valid_predictions["score"].to_numpy(dtype=np.float64)
    y_true = valid_predictions["y_true"].to_numpy(dtype=np.int8)
    weights = valid_predictions["weight"].to_numpy(dtype=np.float64)

    if thresholds_by_jet is not None:
        thresholds = valid_predictions["jet_num"].map(thresholds_by_jet).to_numpy(dtype=np.float64)
        y_pred = (scores >= thresholds).astype(np.int8)
        thresholds_repr = json.dumps({str(key): float(value) for key, value in thresholds_by_jet.items()}, sort_keys=True)
    elif global_threshold is not None:
        y_pred = (scores >= float(global_threshold)).astype(np.int8)
        thresholds_repr = f"{float(global_threshold):.12f}"
    else:
        raise ValueError("Debe indicarse global_threshold o thresholds_by_jet.")

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
        "thresholds": thresholds_repr,
    }


def optimize_joint_thresholds(
    valid_predictions: pd.DataFrame,
    starting_thresholds: dict[int, float],
    max_iter: int = 8,
) -> dict[int, float]:
    tables = threshold_tables_by_jet(valid_predictions)
    thresholds = dict(starting_thresholds)

    for _ in range(max_iter):
        changed = False
        for jet_value in JET_GROUPS:
            fixed_signal = 0.0
            fixed_background = 0.0
            for other_jet in JET_GROUPS:
                if other_jet == jet_value:
                    continue
                table = tables[other_jet]
                eligible = table.loc[table["threshold"] >= thresholds[other_jet]]
                if not eligible.empty:
                    row = eligible.iloc[-1]
                    fixed_signal += float(row["signal_weight"])
                    fixed_background += float(row["background_weight"])

            candidate_table = tables[jet_value]
            ams_values = ams_score(
                fixed_signal + candidate_table["signal_weight"].to_numpy(dtype=np.float64),
                fixed_background + candidate_table["background_weight"].to_numpy(dtype=np.float64),
            )
            best_idx = int(np.argmax(ams_values))
            new_threshold = float(candidate_table.iloc[best_idx]["threshold"])
            changed = changed or not np.isclose(thresholds[jet_value], new_threshold)
            thresholds[jet_value] = new_threshold
        if not changed:
            break
    return thresholds


def compare_threshold_policies(valid_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    global_table = cumulative_table(valid_predictions["y_true"], valid_predictions["score"].to_numpy(), valid_predictions["weight"])
    global_best = best_threshold_from_table(global_table)
    tables = threshold_tables_by_jet(valid_predictions)
    independent = {jet_value: float(best_threshold_from_table(table)["threshold"]) for jet_value, table in tables.items()}
    joint_from_global = optimize_joint_thresholds(
        valid_predictions,
        {jet_value: float(global_best["threshold"]) for jet_value in JET_GROUPS},
    )
    joint_from_independent = optimize_joint_thresholds(valid_predictions, independent)

    rows = [
        threshold_row(valid_predictions, "global_single_threshold", global_threshold=float(global_best["threshold"])),
        threshold_row(valid_predictions, "independent_threshold_by_jet", thresholds_by_jet=independent),
        threshold_row(valid_predictions, "joint_thresholds_from_global", thresholds_by_jet=joint_from_global),
        threshold_row(valid_predictions, "joint_thresholds_from_independent", thresholds_by_jet=joint_from_independent),
    ]
    comparison = pd.DataFrame(rows).sort_values("ams", ascending=False).reset_index(drop=True)
    details = {
        "global_threshold": float(global_best["threshold"]),
        "independent_thresholds": independent,
        "joint_thresholds_from_global": joint_from_global,
        "joint_thresholds_from_independent": joint_from_independent,
        "best_policy": comparison.iloc[0].to_dict(),
    }
    return comparison, details


def parse_thresholds(policy_row: pd.Series) -> dict[str, float]:
    if str(policy_row["policy"]) == "global_single_threshold":
        return {"global": float(policy_row["thresholds"])}
    parsed = json.loads(str(policy_row["thresholds"]))
    return {str(key): float(value) for key, value in parsed.items()}


def build_submission(test_predictions: pd.DataFrame, selected_thresholds: dict[str, float]) -> pd.DataFrame:
    scores = test_predictions["score"].to_numpy(dtype=np.float64)
    if "global" in selected_thresholds:
        y_pred_signal = scores >= float(selected_thresholds["global"])
    else:
        thresholds = test_predictions["jet_num"].astype(str).map(selected_thresholds).to_numpy(dtype=np.float64)
        y_pred_signal = scores >= thresholds

    return pd.DataFrame(
        {
            "EventId": test_predictions["EventId"].astype(np.int64),
            "RankOrder": pd.Series(scores).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(y_pred_signal, "s", "b"),
        }
    )[["EventId", "RankOrder", "Class"]]


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


def save_bundle(
    artifacts: dict[str, object],
    models_by_jet: dict[int, XGBClassifier],
    best_params_by_jet: dict[int, dict[str, object]],
    policy_details: dict[str, object],
) -> Path:
    bundle_dir = MODELS_DIR / "weighted_xgboost_by_jets_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        bundle_dir / "metadata.json",
        {
            "bundle_type": "weighted_xgboost_by_jets",
            "best_params_by_jet": best_params_by_jet,
            "policy_details": policy_details,
        },
    )
    for jet_value in JET_GROUPS:
        group_dir = bundle_dir / f"jet_{jet_value}"
        group_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models_by_jet[jet_value], group_dir / "xgboost_model.joblib")
        joblib.dump(artifacts["group_artifacts"][jet_value]["preprocessing_pipeline"], group_dir / "preprocessing_pipeline.joblib")
    return bundle_dir


def run_investigation() -> dict[str, object]:
    ensure_dirs()
    artifacts = run_i3_tree_safe_mix_by_jets(project_root=PROJECT_ROOT)
    models_by_jet, candidate_runs, best_by_jet, best_params_by_jet = train_best_models_by_jet(artifacts)
    valid_predictions = build_validation_predictions(artifacts, models_by_jet)
    policy_comparison, policy_details = compare_threshold_policies(valid_predictions)
    best_policy = policy_comparison.iloc[0]
    selected_thresholds = parse_thresholds(best_policy)
    policy_details["selected_thresholds"] = selected_thresholds
    test_predictions = build_test_predictions(artifacts, models_by_jet)
    submission = build_submission(test_predictions, selected_thresholds)
    submission_validation = validate_submission(submission)
    bundle_dir = save_bundle(artifacts, models_by_jet, best_params_by_jet, policy_details)

    candidate_runs.to_csv(RESULTS_DIR / "01_candidate_runs.csv", index=False)
    best_by_jet.to_csv(RESULTS_DIR / "01_best_by_jet.csv", index=False)
    valid_predictions.to_csv(RESULTS_DIR / "01_valid_scores.csv", index=False)
    test_predictions.to_csv(RESULTS_DIR / "01_test_scores.csv", index=False)
    policy_comparison.to_csv(RESULTS_DIR / "01_policy_comparison.csv", index=False)
    submission_path = RESULTS_DIR / "higgs_submission_weighted_xgboost_by_jets.csv"
    submission.to_csv(submission_path, index=False)
    dump_json(RESULTS_DIR / "01_metadata.json", {
        "policy_details": policy_details,
        "best_params_by_jet": best_params_by_jet,
        "submission_validation": submission_validation,
        "bundle_dir": bundle_dir,
        "submission_path": submission_path,
    })

    summary_lines = [
        "# Weighted XGBoost por jets",
        "",
        f"Mejor politica: {best_policy['policy']}",
        f"AMS validacion: {float(best_policy['ams']):.6f}",
        f"AUC validacion: {float(best_policy['auc']):.6f}",
        f"Precision validacion: {float(best_policy['precision']):.6f}",
        f"Recall validacion: {float(best_policy['recall']):.6f}",
        f"Thresholds: {best_policy['thresholds']}",
        f"Signal rows submission: {submission_validation['signal_count']}",
        f"Background rows submission: {submission_validation['background_count']}",
        "",
        f"Submission: {submission_path}",
        f"Bundle: {bundle_dir}",
    ]
    (RESULTS_DIR / "01_summary.txt").write_text("\n".join(summary_lines) + "\n")

    return {
        "candidate_runs": candidate_runs,
        "best_by_jet": best_by_jet,
        "valid_predictions": valid_predictions,
        "test_predictions": test_predictions,
        "policy_comparison": policy_comparison,
        "metadata": {
            "policy_details": policy_details,
            "best_params_by_jet": best_params_by_jet,
            "submission_validation": submission_validation,
            "bundle_dir": str(bundle_dir),
            "submission_path": str(submission_path),
        },
    }


def main() -> None:
    outputs = run_investigation()
    print("Mejores variantes por jet:")
    print(outputs["best_by_jet"][["jet_num", "variant", "use_sample_weight", "auc", "ams", "threshold"]].to_string(index=False))
    print("\nPoliticas de threshold:")
    print(outputs["policy_comparison"].to_string(index=False))
    print(f"\nSubmission: {outputs['metadata']['submission_path']}")


if __name__ == "__main__":
    main()
