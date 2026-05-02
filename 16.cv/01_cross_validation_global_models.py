from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SEED = 42
N_SPLITS = 5
DATASETS = ["i3_tree_safe_mix", "v2_physics_by_jet"]
MODEL_VARIANTS = ["baseline", "tuned"]
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

STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR.parent
RESULTS_DIR = STAGE_DIR / "results"


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar modulo: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


v2_module = load_module("cv_higgs_preprocessing_v2", PROJECT_ROOT / "10.try" / "higgs_preprocessing_v2.py")
i3_module = load_module("cv_i3_tree_safe_mix", PROJECT_ROOT / "11.Mezcla" / "i3_tree_safe_mix.py")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    return value


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def load_tuned_params() -> dict[str, dict[str, object]]:
    best_models_path = PROJECT_ROOT / "15.tune_global" / "results" / "01_best_models.csv"
    best_models = pd.read_csv(best_models_path)
    params_by_dataset: dict[str, dict[str, object]] = {}
    for _, row in best_models.iterrows():
        params_by_dataset[str(row["dataset"])] = json.loads(str(row["params_json"]))
    return params_by_dataset


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


def build_preprocessed_fold(
    dataset_name: str,
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if dataset_name == "i3_tree_safe_mix":
        pipeline = i3_module.build_i3_tree_safe_mix_pipeline()
    elif dataset_name == "v2_physics_by_jet":
        pipeline = v2_module.build_preprocessing_pipeline("physics_by_jet")
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")

    pipeline.fit(X_train_raw)
    X_train_processed = pipeline.transform(X_train_raw)
    X_valid_processed = pipeline.transform(X_valid_raw)
    if not isinstance(X_train_processed, pd.DataFrame):
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
        X_valid_processed = pd.DataFrame(X_valid_processed, columns=feature_names)
    return (
        X_train_processed.astype(np.float32).reset_index(drop=True),
        X_valid_processed.astype(np.float32).reset_index(drop=True),
        int(X_train_processed.shape[1]),
    )


def fit_and_score(
    params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    model = XGBClassifier(**params)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    scores = model.predict_proba(X_valid)[:, 1]
    return scores, fit_seconds


def evaluate_cv() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    tuned_params = load_tuned_params()
    train_data, _ = v2_module.load_higgs_data(project_root=PROJECT_ROOT)
    base = v2_module.split_higgs_features(train_data, train_data.drop(columns=["Weight", "Label"]))
    X_full = base["X_train_full"].reset_index(drop=True)
    y_full_label = base["y_train_full"].reset_index(drop=True)
    y_full = y_full_label.map({"b": 0, "s": 1}).astype(np.int8)
    weight_full = base["train_weight"].reset_index(drop=True).astype(np.float64)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    model_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_full, y_full), start=1):
        X_train_raw = X_full.iloc[train_idx].reset_index(drop=True)
        X_valid_raw = X_full.iloc[valid_idx].reset_index(drop=True)
        y_train = y_full.iloc[train_idx].reset_index(drop=True)
        y_valid = y_full.iloc[valid_idx].reset_index(drop=True)
        valid_weight = weight_full.iloc[valid_idx].reset_index(drop=True)

        fold_scores: dict[tuple[str, str], np.ndarray] = {}
        for dataset_name in DATASETS:
            X_train_processed, X_valid_processed, feature_count = build_preprocessed_fold(
                dataset_name,
                X_train_raw,
                X_valid_raw,
            )
            for variant in MODEL_VARIANTS:
                params = dict(BASE_XGB_PARAMS) if variant == "baseline" else dict(tuned_params[dataset_name])
                scores, fit_seconds = fit_and_score(params, X_train_processed, y_train, X_valid_processed)
                metrics = find_best_ams_threshold(y_valid, scores, valid_weight)
                fold_scores[(dataset_name, variant)] = scores
                row = {
                    "fold": fold,
                    "dataset": dataset_name,
                    "variant": variant,
                    "feature_count": feature_count,
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
                model_rows.append(row)
                print(
                    f"fold={fold} dataset={dataset_name} variant={variant} "
                    f"ams={float(metrics['ams']):.6f} auc={float(metrics['auc']):.6f}"
                )

        ensemble_configs = [
            {
                "ensemble": "baseline_13comb_rank_30_70",
                "left": ("i3_tree_safe_mix", "baseline"),
                "right": ("v2_physics_by_jet", "baseline"),
                "i3_weight": 0.30,
                "v2_weight": 0.70,
                "blend_type": "percentile_rank",
            },
            {
                "ensemble": "tuned_best_v2_only",
                "left": ("i3_tree_safe_mix", "tuned"),
                "right": ("v2_physics_by_jet", "tuned"),
                "i3_weight": 0.00,
                "v2_weight": 1.00,
                "blend_type": "raw_probability",
            },
            {
                "ensemble": "tuned_rank_20_80",
                "left": ("i3_tree_safe_mix", "tuned"),
                "right": ("v2_physics_by_jet", "tuned"),
                "i3_weight": 0.20,
                "v2_weight": 0.80,
                "blend_type": "percentile_rank",
            },
        ]
        for config in ensemble_configs:
            left_scores = fold_scores[config["left"]]
            right_scores = fold_scores[config["right"]]
            if config["blend_type"] == "percentile_rank":
                left_scores = percentile_rank(left_scores)
                right_scores = percentile_rank(right_scores)
            ensemble_score = float(config["i3_weight"]) * left_scores + float(config["v2_weight"]) * right_scores
            metrics = find_best_ams_threshold(y_valid, ensemble_score, valid_weight)
            ensemble_rows.append({"fold": fold, **config, **metrics})
            print(
                f"fold={fold} ensemble={config['ensemble']} "
                f"ams={float(metrics['ams']):.6f} auc={float(metrics['auc']):.6f}"
            )

    model_folds = pd.DataFrame(model_rows)
    ensemble_folds = pd.DataFrame(ensemble_rows)
    model_summary = (
        model_folds.groupby(["dataset", "variant"])
        .agg(
            ams_mean=("ams", "mean"),
            ams_std=("ams", "std"),
            ams_min=("ams", "min"),
            ams_max=("ams", "max"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .reset_index()
        .sort_values("ams_mean", ascending=False)
    )
    ensemble_summary = (
        ensemble_folds.groupby(["ensemble", "blend_type", "i3_weight", "v2_weight"])
        .agg(
            ams_mean=("ams", "mean"),
            ams_std=("ams", "std"),
            ams_min=("ams", "min"),
            ams_max=("ams", "max"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
        )
        .reset_index()
        .sort_values("ams_mean", ascending=False)
    )

    model_folds.to_csv(RESULTS_DIR / "01_model_folds.csv", index=False)
    ensemble_folds.to_csv(RESULTS_DIR / "01_ensemble_folds.csv", index=False)
    model_summary.to_csv(RESULTS_DIR / "01_model_summary.csv", index=False)
    ensemble_summary.to_csv(RESULTS_DIR / "01_ensemble_summary.csv", index=False)

    summary_lines = [
        "# Cross-validation global",
        "",
        f"Folds: {N_SPLITS}",
        "",
        "## Best model summary",
        model_summary.to_string(index=False),
        "",
        "## Best ensemble summary",
        ensemble_summary.to_string(index=False),
    ]
    (RESULTS_DIR / "01_summary.txt").write_text("\n".join(summary_lines) + "\n")
    dump_json(
        RESULTS_DIR / "01_metadata.json",
        {
            "n_splits": N_SPLITS,
            "seed": SEED,
            "datasets": DATASETS,
            "variants": MODEL_VARIANTS,
            "note": "Each fold refits preprocessing on fold train only before training XGBoost.",
        },
    )

    return {
        "model_folds": model_folds,
        "ensemble_folds": ensemble_folds,
        "model_summary": model_summary,
        "ensemble_summary": ensemble_summary,
    }


def main() -> None:
    outputs = evaluate_cv()
    print("\nModel summary:")
    print(outputs["model_summary"].to_string(index=False))
    print("\nEnsemble summary:")
    print(outputs["ensemble_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
