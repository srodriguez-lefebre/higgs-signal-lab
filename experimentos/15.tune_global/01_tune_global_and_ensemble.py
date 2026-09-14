from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


SEED = 42
DATASETS = ["i3_tree_safe_mix", "v2_physics_by_jet"]
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
MODELS_DIR = SCRIPT_DIR / "models"


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


def load_processed_dataset(processed_name: str) -> dict[str, object]:
    processed_dir = PROJECT_ROOT / "data" / "higgs-boson" / "processed" / processed_name
    if not processed_dir.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {processed_dir}")

    label_map = {"b": 0, "s": 1}
    metadata_path = processed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return {
        "processed_name": processed_name,
        "processed_dir": processed_dir,
        "metadata": metadata,
        "X_train": pd.read_csv(processed_dir / "X_train_processed.csv").astype(np.float32),
        "X_valid": pd.read_csv(processed_dir / "X_valid_processed.csv").astype(np.float32),
        "X_test": pd.read_csv(processed_dir / "X_test_processed.csv").astype(np.float32),
        "y_train": pd.read_csv(processed_dir / "y_train.csv")["Label"].map(label_map).astype(np.int8),
        "y_valid": pd.read_csv(processed_dir / "y_valid.csv")["Label"].map(label_map).astype(np.int8),
        "train_weight": pd.read_csv(processed_dir / "train_weight.csv")["Weight"].astype(np.float64),
        "valid_weight": pd.read_csv(processed_dir / "valid_weight.csv")["Weight"].astype(np.float64),
        "train_event_id": pd.read_csv(processed_dir / "train_event_id.csv")["EventId"].astype(np.int64),
        "valid_event_id": pd.read_csv(processed_dir / "valid_event_id.csv")["EventId"].astype(np.int64),
        "test_event_id": pd.read_csv(processed_dir / "test_event_id.csv")["EventId"].astype(np.int64),
    }


def assert_same_split(left: dict[str, object], right: dict[str, object]) -> None:
    for key in ["train_event_id", "valid_event_id", "test_event_id", "y_valid", "valid_weight"]:
        if not np.array_equal(left[key].to_numpy(), right[key].to_numpy()):
            raise ValueError(f"Los datasets no comparten el mismo split: {key}")


def ams_score(signal_weight: np.ndarray | float, background_weight: np.ndarray | float, br: float = 10.0) -> np.ndarray:
    signal_arr = np.asarray(signal_weight, dtype=np.float64)
    background_arr = np.asarray(background_weight, dtype=np.float64)
    radicand = 2.0 * (
        (signal_arr + background_arr + br) * np.log1p(signal_arr / (background_arr + br)) - signal_arr
    )
    return np.sqrt(np.clip(radicand, a_min=0.0, a_max=None))


def find_best_ams_threshold(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    weights: pd.Series | np.ndarray,
) -> dict[str, float | int]:
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
    ams_values = ams_score(signal_cumsum[candidate_indices], background_cumsum[candidate_indices])
    best_position = int(np.argmax(ams_values))
    best_idx = int(candidate_indices[best_position])
    threshold = float(score_sorted[best_idx])
    y_pred = (score_array >= threshold).astype(np.int8)

    return {
        "ams_threshold": threshold,
        "ams": float(ams_values[best_position]),
        "signal_weight": float(signal_cumsum[best_idx]),
        "background_weight": float(background_cumsum[best_idx]),
        "predicted_signal_count": int(y_pred.sum()),
        "accuracy": float(accuracy_score(y_array, y_pred)),
        "precision": float(precision_score(y_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_array, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_array, score_array)),
    }


def class_ratios(dataset: dict[str, object]) -> dict[str, float]:
    y_train = dataset["y_train"]
    train_weight = dataset["train_weight"]
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    return {
        "count_ratio": float(neg_mask.sum() / pos_mask.sum()),
        "weight_ratio": float(train_weight.loc[neg_mask].sum() / train_weight.loc[pos_mask].sum()),
    }


def staged_candidates(dataset: dict[str, object]) -> list[tuple[str, list[dict[str, object]]]]:
    ratios = class_ratios(dataset)
    return [
        (
            "class_weight",
            [
                {"scale_pos_weight": 1.0},
                {"scale_pos_weight": ratios["count_ratio"]},
                {"scale_pos_weight": ratios["weight_ratio"]},
                {"scale_pos_weight": 1.5346, "max_delta_step": 1},
                {"scale_pos_weight": ratios["count_ratio"], "max_delta_step": 1},
            ],
        ),
        (
            "structure",
            [
                {"max_depth": 4},
                {"max_depth": 5},
                {"max_depth": 7},
                {"max_depth": 8},
                {"min_child_weight": 3},
                {"min_child_weight": 5},
                {"gamma": 0.1},
                {"gamma": 0.3},
            ],
        ),
        (
            "capacity",
            [
                {"learning_rate": 0.03, "n_estimators": 700},
                {"learning_rate": 0.04, "n_estimators": 550},
                {"learning_rate": 0.07, "n_estimators": 330},
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


def fit_candidate(
    dataset: dict[str, object],
    params: dict[str, object],
    dataset_name: str,
    stage: str,
    candidate: str,
) -> tuple[XGBClassifier, dict[str, object], np.ndarray]:
    model = XGBClassifier(**params)
    start = time.perf_counter()
    model.fit(dataset["X_train"], dataset["y_train"])
    fit_seconds = time.perf_counter() - start
    valid_scores = model.predict_proba(dataset["X_valid"])[:, 1]
    metrics = find_best_ams_threshold(dataset["y_valid"], valid_scores, dataset["valid_weight"])
    row = {
        "dataset": dataset_name,
        "stage": stage,
        "candidate": candidate,
        "params_json": json.dumps(to_jsonable(params), sort_keys=True),
        "fit_seconds": float(fit_seconds),
        "feature_count": int(dataset["X_train"].shape[1]),
        **metrics,
    }
    return model, row, valid_scores


def tune_dataset(dataset_name: str, dataset: dict[str, object]) -> tuple[XGBClassifier, dict[str, object], list[dict[str, object]]]:
    current_params = dict(BASE_XGB_PARAMS)
    rows: list[dict[str, object]] = []
    best_model, baseline_row, _ = fit_candidate(dataset, current_params, dataset_name, "baseline", "shared_baseline")
    rows.append(baseline_row)
    best_row = baseline_row
    best_ams = float(baseline_row["ams"])

    for stage, overrides_list in staged_candidates(dataset):
        stage_results: list[tuple[XGBClassifier, dict[str, object]]] = []
        for overrides in overrides_list:
            params = deepcopy(current_params)
            params.update(overrides)
            candidate_name = ",".join(f"{key}={value}" for key, value in sorted(overrides.items()))
            model, row, _ = fit_candidate(dataset, params, dataset_name, stage, candidate_name)
            rows.append(row)
            stage_results.append((model, row))

        stage_model, stage_row = max(stage_results, key=lambda item: float(item[1]["ams"]))
        if float(stage_row["ams"]) > best_ams:
            current_params = json.loads(str(stage_row["params_json"]))
            best_model = stage_model
            best_row = stage_row
            best_ams = float(stage_row["ams"])

    print(
        f"{dataset_name}: best_stage={best_row['stage']} candidate={best_row['candidate']} "
        f"ams={float(best_row['ams']):.6f} auc={float(best_row['auc']):.6f}"
    )
    return best_model, best_row, rows


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(method="first", pct=True).to_numpy(dtype=np.float64)


def build_submission(event_ids: pd.Series, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=np.float64)
    return pd.DataFrame(
        {
            "EventId": event_ids.astype(np.int64),
            "RankOrder": pd.Series(score_array).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(score_array >= threshold, "s", "b"),
        }
    )[["EventId", "RankOrder", "Class"]]


def validate_submission(submission: pd.DataFrame) -> dict[str, int]:
    ranks = submission["RankOrder"].to_numpy(dtype=np.int64)
    if list(submission.columns) != ["EventId", "RankOrder", "Class"]:
        raise ValueError("Columnas invalidas.")
    if not submission["EventId"].is_unique:
        raise ValueError("EventId repetido.")
    if ranks.min() != 1 or ranks.max() != len(ranks) or len(np.unique(ranks)) != len(ranks):
        raise ValueError("RankOrder invalido.")
    invalid_classes = sorted(set(submission["Class"].unique()) - {"s", "b"})
    if invalid_classes:
        raise ValueError(f"Clases invalidas: {invalid_classes}")
    return {
        "rows": int(len(submission)),
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "signal_count": int((submission["Class"] == "s").sum()),
        "background_count": int((submission["Class"] == "b").sum()),
    }


def run_tuning_and_ensemble() -> dict[str, object]:
    ensure_dirs()
    datasets = {name: load_processed_dataset(name) for name in DATASETS}
    assert_same_split(datasets["i3_tree_safe_mix"], datasets["v2_physics_by_jet"])

    models: dict[str, XGBClassifier] = {}
    best_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    valid_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for dataset_name, dataset in datasets.items():
        model, best_row, rows = tune_dataset(dataset_name, dataset)
        models[dataset_name] = model
        best_rows.append(best_row)
        all_rows.extend(rows)
        valid_scores[dataset_name] = model.predict_proba(dataset["X_valid"])[:, 1]
        test_scores[dataset_name] = model.predict_proba(dataset["X_test"])[:, 1]
        joblib.dump(model, MODELS_DIR / f"{dataset_name}_tuned_xgboost.joblib")

    base_dataset = datasets["i3_tree_safe_mix"]
    blend_rows: list[dict[str, object]] = []
    blend_valid_scores: dict[str, np.ndarray] = {}
    blend_test_scores: dict[str, np.ndarray] = {}

    for blend_type in ["raw_probability", "percentile_rank"]:
        if blend_type == "raw_probability":
            i3_valid = valid_scores["i3_tree_safe_mix"]
            v2_valid = valid_scores["v2_physics_by_jet"]
            i3_test = test_scores["i3_tree_safe_mix"]
            v2_test = test_scores["v2_physics_by_jet"]
        else:
            i3_valid = percentile_rank(valid_scores["i3_tree_safe_mix"])
            v2_valid = percentile_rank(valid_scores["v2_physics_by_jet"])
            i3_test = percentile_rank(test_scores["i3_tree_safe_mix"])
            v2_test = percentile_rank(test_scores["v2_physics_by_jet"])

        for i3_weight in np.linspace(0.0, 1.0, 101):
            i3_weight = float(i3_weight)
            v2_weight = 1.0 - i3_weight
            ensemble_valid = i3_weight * i3_valid + v2_weight * v2_valid
            ensemble_test = i3_weight * i3_test + v2_weight * v2_test
            metrics = find_best_ams_threshold(base_dataset["y_valid"], ensemble_valid, base_dataset["valid_weight"])
            key = f"{blend_type}_i3_{i3_weight:.2f}_v2_{v2_weight:.2f}"
            blend_valid_scores[key] = ensemble_valid
            blend_test_scores[key] = ensemble_test
            blend_rows.append(
                {
                    "blend_key": key,
                    "blend_type": blend_type,
                    "i3_weight": i3_weight,
                    "v2_weight": v2_weight,
                    **metrics,
                }
            )

    tune_runs_df = pd.DataFrame(all_rows)
    best_models_df = pd.DataFrame(best_rows).sort_values("dataset").reset_index(drop=True)
    blend_df = pd.DataFrame(blend_rows).sort_values("ams", ascending=False).reset_index(drop=True)
    best_blend = blend_df.iloc[0].to_dict()
    best_key = str(best_blend["blend_key"])

    valid_scores_df = pd.DataFrame(
        {
            "EventId": base_dataset["valid_event_id"],
            "y_true": base_dataset["y_valid"],
            "weight": base_dataset["valid_weight"],
            "i3_tree_safe_mix_score": valid_scores["i3_tree_safe_mix"],
            "v2_physics_by_jet_score": valid_scores["v2_physics_by_jet"],
            "best_ensemble_score": blend_valid_scores[best_key],
        }
    )
    test_scores_df = pd.DataFrame(
        {
            "EventId": base_dataset["test_event_id"],
            "i3_tree_safe_mix_score": test_scores["i3_tree_safe_mix"],
            "v2_physics_by_jet_score": test_scores["v2_physics_by_jet"],
            "best_ensemble_score": blend_test_scores[best_key],
        }
    )

    submission = build_submission(
        event_ids=base_dataset["test_event_id"],
        scores=blend_test_scores[best_key],
        threshold=float(best_blend["ams_threshold"]),
    )
    submission_validation = validate_submission(submission)

    tune_runs_df.to_csv(RESULTS_DIR / "01_tune_runs.csv", index=False)
    best_models_df.to_csv(RESULTS_DIR / "01_best_models.csv", index=False)
    blend_df.to_csv(RESULTS_DIR / "01_blend_sweep.csv", index=False)
    valid_scores_df.to_csv(RESULTS_DIR / "01_valid_scores.csv", index=False)
    test_scores_df.to_csv(RESULTS_DIR / "01_test_scores.csv", index=False)
    submission_path = RESULTS_DIR / "higgs_submission_tuned_i3_plus_tuned_v2.csv"
    submission.to_csv(submission_path, index=False)

    metadata = {
        "best_blend": best_blend,
        "submission_validation": submission_validation,
        "submission_path": submission_path,
        "best_models": best_models_df.to_dict(orient="records"),
    }
    dump_json(RESULTS_DIR / "01_metadata.json", metadata)

    summary_lines = [
        "# Tuned global ensemble",
        "",
        f"Best blend: {best_key}",
        f"AMS validacion: {float(best_blend['ams']):.6f}",
        f"AUC validacion: {float(best_blend['auc']):.6f}",
        f"Threshold: {float(best_blend['ams_threshold']):.6f}",
        f"Signal rows submission: {submission_validation['signal_count']}",
        f"Background rows submission: {submission_validation['background_count']}",
        "",
        f"Submission: {submission_path}",
    ]
    (RESULTS_DIR / "01_summary.txt").write_text("\n".join(summary_lines) + "\n")

    return {
        "tune_runs": tune_runs_df,
        "best_models": best_models_df,
        "blend_sweep": blend_df,
        "valid_scores": valid_scores_df,
        "test_scores": test_scores_df,
        "metadata": metadata,
    }


def main() -> None:
    outputs = run_tuning_and_ensemble()
    print("Mejores modelos globales:")
    print(outputs["best_models"][["dataset", "stage", "candidate", "auc", "ams", "ams_threshold"]].to_string(index=False))
    print("\nMejores blends:")
    print(outputs["blend_sweep"].head(10).to_string(index=False))
    print(f"\nSubmission: {outputs['metadata']['submission_path']}")


if __name__ == "__main__":
    main()
