from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


SEED = 42
BEST_XGB_PARAMS: dict[str, object] = {
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

DATASETS = ["i3_tree_safe_mix", "v2_physics_by_jet"]


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "roadmap.md").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


def load_processed_dataset(project_root: Path, processed_name: str) -> dict[str, object]:
    processed_dir = project_root / "data" / "higgs-boson" / "processed" / processed_name
    if not processed_dir.exists():
        raise FileNotFoundError(f"No existe el directorio procesado: {processed_dir}")

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
        left_values = left[key].to_numpy()
        right_values = right[key].to_numpy()
        if not np.array_equal(left_values, right_values):
            raise ValueError(f"Los datasets no comparten el mismo split/campo: {key}")


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


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(method="first", pct=True).to_numpy(dtype=np.float64)


def train_model(dataset: dict[str, object]) -> tuple[XGBClassifier, dict[str, float | int]]:
    model = XGBClassifier(**BEST_XGB_PARAMS)
    start = time.perf_counter()
    model.fit(dataset["X_train"], dataset["y_train"])
    fit_seconds = time.perf_counter() - start
    valid_scores = model.predict_proba(dataset["X_valid"])[:, 1]
    metrics = find_best_ams_threshold(dataset["y_valid"], valid_scores, dataset["valid_weight"])
    metrics["fit_seconds"] = float(fit_seconds)
    metrics["feature_count"] = int(dataset["X_train"].shape[1])
    return model, metrics


def build_submission(event_ids: pd.Series, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=np.float64)
    submission = pd.DataFrame(
        {
            "EventId": event_ids.astype(np.int64),
            "RankOrder": pd.Series(score_array).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(score_array >= threshold, "s", "b"),
        }
    )
    return submission[["EventId", "RankOrder", "Class"]]


def validate_submission(submission: pd.DataFrame) -> dict[str, int]:
    if list(submission.columns) != ["EventId", "RankOrder", "Class"]:
        raise ValueError("Columnas invalidas para Kaggle.")
    ranks = submission["RankOrder"].to_numpy(dtype=np.int64)
    if not submission["EventId"].is_unique:
        raise ValueError("EventId repetido.")
    if len(np.unique(ranks)) != len(ranks) or ranks.min() != 1 or ranks.max() != len(ranks):
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


def main() -> None:
    project_root = find_project_root()
    output_dir = project_root / "13.comb"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    i3_data = load_processed_dataset(project_root, "i3_tree_safe_mix")
    v2_data = load_processed_dataset(project_root, "v2_physics_by_jet")
    assert_same_split(i3_data, v2_data)

    datasets = {"i3_tree_safe_mix": i3_data, "v2_physics_by_jet": v2_data}
    models: dict[str, XGBClassifier] = {}
    model_rows: list[dict[str, object]] = []
    valid_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for dataset_name, dataset in datasets.items():
        model, metrics = train_model(dataset)
        models[dataset_name] = model
        valid_scores[dataset_name] = model.predict_proba(dataset["X_valid"])[:, 1]
        test_scores[dataset_name] = model.predict_proba(dataset["X_test"])[:, 1]
        model_rows.append({"dataset": dataset_name, **metrics})
        joblib.dump(model, models_dir / f"{dataset_name}_xgboost.joblib")

    y_valid = i3_data["y_valid"]
    valid_weight = i3_data["valid_weight"]
    valid_event_id = i3_data["valid_event_id"]
    test_event_id = i3_data["test_event_id"]

    blend_rows: list[dict[str, object]] = []
    blend_scores_valid: dict[str, np.ndarray] = {}
    blend_scores_test: dict[str, np.ndarray] = {}

    for blend_type in ["raw_probability", "percentile_rank"]:
        if blend_type == "raw_probability":
            i3_valid_base = valid_scores["i3_tree_safe_mix"]
            v2_valid_base = valid_scores["v2_physics_by_jet"]
            i3_test_base = test_scores["i3_tree_safe_mix"]
            v2_test_base = test_scores["v2_physics_by_jet"]
        else:
            i3_valid_base = percentile_rank(valid_scores["i3_tree_safe_mix"])
            v2_valid_base = percentile_rank(valid_scores["v2_physics_by_jet"])
            i3_test_base = percentile_rank(test_scores["i3_tree_safe_mix"])
            v2_test_base = percentile_rank(test_scores["v2_physics_by_jet"])

        for i3_weight in np.linspace(0.0, 1.0, 21):
            v2_weight = 1.0 - float(i3_weight)
            ensemble_valid = float(i3_weight) * i3_valid_base + v2_weight * v2_valid_base
            ensemble_test = float(i3_weight) * i3_test_base + v2_weight * v2_test_base
            metrics = find_best_ams_threshold(y_valid, ensemble_valid, valid_weight)
            key = f"{blend_type}_i3_{float(i3_weight):.2f}_v2_{v2_weight:.2f}"
            blend_scores_valid[key] = ensemble_valid
            blend_scores_test[key] = ensemble_test
            blend_rows.append(
                {
                    "blend_key": key,
                    "blend_type": blend_type,
                    "i3_weight": float(i3_weight),
                    "v2_weight": v2_weight,
                    **metrics,
                }
            )

    model_metrics_df = pd.DataFrame(model_rows).sort_values("ams", ascending=False).reset_index(drop=True)
    blend_metrics_df = pd.DataFrame(blend_rows).sort_values("ams", ascending=False).reset_index(drop=True)
    best_blend = blend_metrics_df.iloc[0].to_dict()
    best_key = str(best_blend["blend_key"])

    valid_scores_df = pd.DataFrame(
        {
            "EventId": valid_event_id,
            "y_true": y_valid,
            "weight": valid_weight,
            "i3_tree_safe_mix_score": valid_scores["i3_tree_safe_mix"],
            "v2_physics_by_jet_score": valid_scores["v2_physics_by_jet"],
            "best_ensemble_score": blend_scores_valid[best_key],
        }
    )
    test_scores_df = pd.DataFrame(
        {
            "EventId": test_event_id,
            "i3_tree_safe_mix_score": test_scores["i3_tree_safe_mix"],
            "v2_physics_by_jet_score": test_scores["v2_physics_by_jet"],
            "best_ensemble_score": blend_scores_test[best_key],
        }
    )

    submission = build_submission(
        event_ids=test_event_id,
        scores=blend_scores_test[best_key],
        threshold=float(best_blend["ams_threshold"]),
    )
    submission_validation = validate_submission(submission)

    model_metrics_path = results_dir / "01_model_metrics.csv"
    blend_metrics_path = results_dir / "01_blend_sweep_metrics.csv"
    valid_scores_path = results_dir / "01_valid_scores.csv"
    test_scores_path = results_dir / "01_test_scores.csv"
    submission_path = results_dir / "higgs_submission_i3_tree_safe_mix_plus_v2_physics_by_jet.csv"
    summary_path = results_dir / "01_summary.txt"
    metadata_path = results_dir / "01_metadata.json"

    model_metrics_df.to_csv(model_metrics_path, index=False)
    blend_metrics_df.to_csv(blend_metrics_path, index=False)
    valid_scores_df.to_csv(valid_scores_path, index=False)
    test_scores_df.to_csv(test_scores_path, index=False)
    submission.to_csv(submission_path, index=False)
    metadata = {
        "best_blend": best_blend,
        "submission_validation": submission_validation,
        "params": BEST_XGB_PARAMS,
        "datasets": DATASETS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    summary_lines = [
        "# Ensemble i3_tree_safe_mix + v2_physics_by_jet",
        "",
        f"Mejor blend: {best_key}",
        f"AMS validacion: {float(best_blend['ams']):.6f}",
        f"AUC validacion: {float(best_blend['auc']):.6f}",
        f"Threshold AMS: {float(best_blend['ams_threshold']):.6f}",
        f"Signal rows submission: {submission_validation['signal_count']}",
        f"Background rows submission: {submission_validation['background_count']}",
        "",
        f"Submission: {submission_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print("Metricas individuales:")
    print(model_metrics_df.to_string(index=False))
    print("\nMejores blends:")
    print(blend_metrics_df.head(10).to_string(index=False))
    print(f"\nSubmission: {submission_path}")


if __name__ == "__main__":
    main()
