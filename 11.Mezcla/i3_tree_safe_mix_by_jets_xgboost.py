from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from i3_tree_safe_mix import JET_NUM_COLUMN
from i3_tree_safe_mix_by_jets import JET_GROUPS, _normalize_jet_num


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


def ams_score(
    signal_weight: np.ndarray | float,
    background_weight: np.ndarray | float,
    br: float = 10.0,
) -> np.ndarray:
    """Computes the AMS metric used by the Higgs challenge."""
    signal_arr = np.asarray(signal_weight, dtype=np.float64)
    background_arr = np.asarray(background_weight, dtype=np.float64)
    radicand = 2.0 * (
        (signal_arr + background_arr + br) * np.log1p(signal_arr / (background_arr + br)) - signal_arr
    )
    return np.sqrt(np.clip(radicand, a_min=0.0, a_max=None))


def find_best_ams_threshold(
    y_true: pd.Series | np.ndarray,
    y_scores: np.ndarray,
    weights: pd.Series | np.ndarray,
    br: float = 10.0,
) -> dict[str, float | int]:
    """Finds the score threshold that maximizes AMS on a labeled validation set."""
    y_array = np.asarray(y_true, dtype=np.int8)
    score_array = np.asarray(y_scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)

    if not (len(y_array) == len(score_array) == len(weight_array)):
        raise ValueError("y_true, y_scores y weights deben tener la misma longitud.")

    order = np.argsort(score_array)[::-1]
    y_sorted = y_array[order]
    score_sorted = score_array[order]
    weight_sorted = weight_array[order]

    signal_cumsum = np.cumsum(np.where(y_sorted == 1, weight_sorted, 0.0))
    background_cumsum = np.cumsum(np.where(y_sorted == 0, weight_sorted, 0.0))

    candidate_mask = np.r_[score_sorted[1:] != score_sorted[:-1], True]
    candidate_indices = np.flatnonzero(candidate_mask)
    ams_values = ams_score(signal_cumsum[candidate_indices], background_cumsum[candidate_indices], br=br)

    best_position = int(np.argmax(ams_values))
    best_idx = int(candidate_indices[best_position])
    best_threshold = float(score_sorted[best_idx])
    best_pred = (score_array >= best_threshold).astype(np.int8)

    return {
        "ams_threshold": best_threshold,
        "ams": float(ams_values[best_position]),
        "signal_weight": float(signal_cumsum[best_idx]),
        "background_weight": float(background_cumsum[best_idx]),
        "predicted_signal_count": int(best_pred.sum()),
        "predicted_background_count": int(len(best_pred) - best_pred.sum()),
        "precision": float(precision_score(y_array, best_pred, zero_division=0)),
        "recall": float(recall_score(y_array, best_pred, zero_division=0)),
        "f1": float(f1_score(y_array, best_pred, zero_division=0)),
        "candidate_threshold_count": int(len(candidate_indices)),
    }


def fit_group_model(
    group: dict[str, object],
    xgb_params: dict[str, object] | None = None,
) -> tuple[XGBClassifier, dict[str, float | int]]:
    """Fits one XGBoost model for a single PRI_jet_num group."""
    params = dict(xgb_params or BEST_XGB_PARAMS)
    X_train = group["X_train_processed"]
    X_valid = group["X_valid_processed"]
    y_train = group["y_train"].map({"b": 0, "s": 1}).astype(np.int8)
    y_valid = group["y_valid"].map({"b": 0, "s": 1}).astype(np.int8)
    valid_weight = group["valid_weight"].astype(np.float64)

    model = XGBClassifier(**params)
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start_time

    valid_score = model.predict_proba(X_valid)[:, 1]
    valid_pred = model.predict(X_valid)
    ams_info = find_best_ams_threshold(y_valid, valid_score, valid_weight)

    metrics = {
        "jet_num": int(group["jet_value"]),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "feature_count": int(X_train.shape[1]),
        "train_nan_count": int(X_train.isna().sum().sum()),
        "valid_nan_count": int(X_valid.isna().sum().sum()),
        "accuracy": float(accuracy_score(y_valid, valid_pred)),
        "precision": float(precision_score(y_valid, valid_pred, zero_division=0)),
        "recall": float(recall_score(y_valid, valid_pred, zero_division=0)),
        "f1": float(f1_score(y_valid, valid_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_valid, valid_score)),
        "ams": float(ams_info["ams"]),
        "ams_threshold": float(ams_info["ams_threshold"]),
        "fit_seconds": float(fit_seconds),
    }
    return model, metrics


def build_routed_prediction_frame(
    artifacts: dict[str, object],
    group_models: dict[int, XGBClassifier],
    split_name: str,
) -> pd.DataFrame:
    """Routes a split through the correct preprocessed group/model and rejoins original order."""
    prediction_parts: list[pd.DataFrame] = []

    for jet_value in JET_GROUPS:
        group = artifacts["group_artifacts"][jet_value]
        model = group_models[jet_value]
        event_id_key = f"{split_name}_event_id"
        processed_key = f"X_{split_name}_processed"

        scores = model.predict_proba(group[processed_key])[:, 1]
        part = pd.DataFrame(
            {
                "EventId": group[event_id_key].astype(np.int64),
                "score": scores,
                "jet_num": int(jet_value),
            }
        )

        if split_name == "valid":
            part["y_true"] = group["y_valid"].map({"b": 0, "s": 1}).astype(np.int8).to_numpy()
            part["weight"] = group["valid_weight"].astype(np.float64).to_numpy()

        prediction_parts.append(part)

    prediction_df = pd.concat(prediction_parts, ignore_index=True)
    global_order = artifacts[f"global_{split_name}_event_id"].astype(np.int64)
    prediction_df = prediction_df.set_index("EventId").loc[global_order].reset_index()
    return prediction_df


def train_xgboost_by_jets(
    artifacts: dict[str, object],
    xgb_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fits one XGBoost model per jet group and evaluates the routed ensemble on validation."""
    group_models: dict[int, XGBClassifier] = {}
    group_metrics_rows: list[dict[str, float | int]] = []

    for jet_value in JET_GROUPS:
        group = artifacts["group_artifacts"][jet_value]
        model, metrics = fit_group_model(group, xgb_params=xgb_params)
        group_models[jet_value] = model
        group_metrics_rows.append(metrics)

    group_metrics_df = pd.DataFrame(group_metrics_rows).sort_values("jet_num").reset_index(drop=True)
    valid_predictions_df = build_routed_prediction_frame(artifacts, group_models, split_name="valid")
    best_global_threshold = find_best_ams_threshold(
        valid_predictions_df["y_true"],
        valid_predictions_df["score"],
        valid_predictions_df["weight"],
    )
    valid_pred_global = (
        valid_predictions_df["score"].to_numpy(dtype=np.float64) >= best_global_threshold["ams_threshold"]
    ).astype(np.int8)

    overall_metrics_df = pd.DataFrame(
        [
            {
                "dataset": "xgboost_i3_tree_safe_mix_by_jets",
                "valid_rows": int(len(valid_predictions_df)),
                "accuracy": float(accuracy_score(valid_predictions_df["y_true"], valid_pred_global)),
                "precision": float(precision_score(valid_predictions_df["y_true"], valid_pred_global, zero_division=0)),
                "recall": float(recall_score(valid_predictions_df["y_true"], valid_pred_global, zero_division=0)),
                "f1": float(f1_score(valid_predictions_df["y_true"], valid_pred_global, zero_division=0)),
                "auc": float(roc_auc_score(valid_predictions_df["y_true"], valid_predictions_df["score"])),
                "ams": float(best_global_threshold["ams"]),
                "ams_threshold": float(best_global_threshold["ams_threshold"]),
                "predicted_signal_count": int(valid_pred_global.sum()),
            }
        ]
    )

    test_scores_df = build_routed_prediction_frame(artifacts, group_models, split_name="test")

    return {
        "group_models": group_models,
        "group_metrics_df": group_metrics_df,
        "valid_predictions_df": valid_predictions_df,
        "overall_metrics_df": overall_metrics_df,
        "global_threshold_info": best_global_threshold,
        "test_scores_df": test_scores_df,
    }


def build_higgs_submission(
    event_ids: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Builds a Kaggle-ready Higgs submission."""
    score_array = np.asarray(scores, dtype=np.float64)
    if len(event_ids) != len(score_array):
        raise ValueError("event_ids y scores deben tener la misma longitud.")

    submission_df = pd.DataFrame(
        {
            "EventId": event_ids.astype(np.int64),
            "RankOrder": pd.Series(score_array).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(score_array >= threshold, "s", "b"),
        }
    )
    return submission_df[["EventId", "RankOrder", "Class"]]


def validate_higgs_submission(submission_df: pd.DataFrame) -> dict[str, int]:
    """Validates that the submission matches the competition format."""
    expected_columns = ["EventId", "RankOrder", "Class"]
    if list(submission_df.columns) != expected_columns:
        raise ValueError(f"Las columnas del submission deben ser exactamente {expected_columns}.")

    if not submission_df["EventId"].is_unique:
        raise ValueError("EventId debe ser único en el submission.")

    rank_array = submission_df["RankOrder"].to_numpy(dtype=np.int64, copy=False)
    expected_rank = np.arange(1, len(submission_df) + 1, dtype=np.int64)
    if rank_array.min() != 1 or rank_array.max() != len(submission_df):
        raise ValueError("RankOrder debe ir de 1 a N.")
    if len(np.unique(rank_array)) != len(submission_df):
        raise ValueError("RankOrder debe ser único.")
    if not np.array_equal(np.sort(rank_array), expected_rank):
        raise ValueError("RankOrder debe contener todos los enteros de 1 a N.")

    invalid_classes = sorted(set(submission_df["Class"].unique()) - {"s", "b"})
    if invalid_classes:
        raise ValueError(f"Class solo puede contener 's' o 'b'. Se encontró: {invalid_classes}")

    return {
        "rows": int(len(submission_df)),
        "rank_min": int(rank_array.min()),
        "rank_max": int(rank_array.max()),
        "signal_count": int((submission_df["Class"] == "s").sum()),
        "background_count": int((submission_df["Class"] == "b").sum()),
    }


def save_xgboost_by_jets_bundle(
    artifacts: dict[str, object],
    training_outputs: dict[str, object],
    output_dir: Path,
    xgb_params: dict[str, object] | None = None,
) -> list[Path]:
    """Saves the trained per-jet ensemble and its routing metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    metadata = {
        "bundle_type": "xgboost_by_jets",
        "variant_key": artifacts["metadata"]["variant_key"],
        "variant_title": artifacts["metadata"]["variant_title"],
        "jet_groups": JET_GROUPS,
        "jet_num_column": JET_NUM_COLUMN,
        "xgb_params": xgb_params or BEST_XGB_PARAMS,
        "global_threshold_info": training_outputs["global_threshold_info"],
        "group_metrics": training_outputs["group_metrics_df"].to_dict(orient="records"),
        "overall_metrics": training_outputs["overall_metrics_df"].iloc[0].to_dict(),
    }
    metadata_path = output_dir / "bundle_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    saved_paths.append(metadata_path)

    group_metrics_path = output_dir / "group_metrics.csv"
    training_outputs["group_metrics_df"].to_csv(group_metrics_path, index=False)
    saved_paths.append(group_metrics_path)

    overall_metrics_path = output_dir / "overall_metrics.csv"
    training_outputs["overall_metrics_df"].to_csv(overall_metrics_path, index=False)
    saved_paths.append(overall_metrics_path)

    for jet_value in JET_GROUPS:
        group_dir = output_dir / f"jet_{jet_value}"
        group_dir.mkdir(parents=True, exist_ok=True)

        model_path = group_dir / "xgboost_model.joblib"
        joblib.dump(training_outputs["group_models"][jet_value], model_path)
        saved_paths.append(model_path)

        preprocessor_path = group_dir / "preprocessing_pipeline.joblib"
        joblib.dump(artifacts["group_artifacts"][jet_value]["preprocessing_pipeline"], preprocessor_path)
        saved_paths.append(preprocessor_path)

        processed_feature_names_path = group_dir / "processed_feature_names.csv"
        pd.DataFrame(
            {"feature": artifacts["group_artifacts"][jet_value]["processed_feature_names"]}
        ).to_csv(processed_feature_names_path, index=False)
        saved_paths.append(processed_feature_names_path)

    return saved_paths


def load_xgboost_by_jets_bundle(bundle_dir: Path) -> dict[str, object]:
    """Loads the trained bundle generated by save_xgboost_by_jets_bundle."""
    metadata_path = bundle_dir / "bundle_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró el metadata del bundle: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    group_models: dict[int, XGBClassifier] = {}
    group_preprocessors: dict[int, object] = {}

    for jet_value in metadata["jet_groups"]:
        group_dir = bundle_dir / f"jet_{jet_value}"
        group_models[int(jet_value)] = joblib.load(group_dir / "xgboost_model.joblib")
        group_preprocessors[int(jet_value)] = joblib.load(group_dir / "preprocessing_pipeline.joblib")

    return {
        "metadata": metadata,
        "group_models": group_models,
        "group_preprocessors": group_preprocessors,
    }


def predict_scores_with_bundle(
    bundle: dict[str, object],
    raw_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Routes raw rows by PRI_jet_num, applies the right preprocessor/model and returns scores."""
    if JET_NUM_COLUMN not in raw_frame.columns:
        raise ValueError(f"Falta la columna requerida para enrutar: {JET_NUM_COLUMN}")

    frame = raw_frame.reset_index(drop=True).copy()
    jet_num = _normalize_jet_num(frame[JET_NUM_COLUMN])
    score_array = np.zeros(len(frame), dtype=np.float64)

    for jet_value in JET_GROUPS:
        mask = jet_num == jet_value
        if not mask.any():
            continue
        group_frame = frame.loc[mask].reset_index(drop=True)
        preprocessor = bundle["group_preprocessors"][jet_value]
        model = bundle["group_models"][jet_value]
        group_processed = preprocessor.transform(group_frame)
        group_scores = model.predict_proba(group_processed)[:, 1]
        score_array[np.flatnonzero(mask.to_numpy())] = group_scores

    return pd.DataFrame(
        {
            "score": score_array,
            "jet_num": jet_num.astype(int),
        }
    )
