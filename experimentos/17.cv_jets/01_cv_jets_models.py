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
JET_GROUPS = [0, 1, 2, 3]
BASELINE_PARAMS: dict[str, object] = {
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

MIX_DIR = PROJECT_ROOT / "11.Mezcla"
TRY_DIR = PROJECT_ROOT / "10.try"
for module_dir in [MIX_DIR, TRY_DIR]:
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar modulo: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mix_module = load_module("cv_jets_i3_tree_safe_mix_by_jets", PROJECT_ROOT / "11.Mezcla" / "i3_tree_safe_mix_by_jets.py")
v2_module = load_module("cv_jets_higgs_preprocessing_v2", PROJECT_ROOT / "10.try" / "higgs_preprocessing_v2.py")


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


def load_plan12_params() -> dict[int, dict[str, object]]:
    raw = json.loads((PROJECT_ROOT / "12.plan" / "results" / "06_tuning_xgboost_por_jets_best_params.json").read_text())
    return {int(key): dict(value) for key, value in raw["best_params_by_jet"].items()}


def load_weighted14_params() -> dict[int, dict[str, object]]:
    metadata = json.loads((PROJECT_ROOT / "14.invest" / "models" / "weighted_xgboost_by_jets_bundle" / "metadata.json").read_text())
    return {int(key): dict(value) for key, value in metadata["best_params_by_jet"].items()}


def label_to_binary(labels: pd.Series) -> pd.Series:
    if labels.dtype == object:
        return labels.map({"b": 0, "s": 1}).astype(np.int8)
    return labels.astype(np.int8)


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


def best_threshold(table: pd.DataFrame) -> dict[str, float | int]:
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


def threshold_row(
    predictions: pd.DataFrame,
    policy: str,
    global_threshold: float | None = None,
    thresholds_by_jet: dict[int, float] | None = None,
) -> dict[str, object]:
    scores = predictions["score"].to_numpy(dtype=np.float64)
    y_true = predictions["y_true"].to_numpy(dtype=np.int8)
    weights = predictions["weight"].to_numpy(dtype=np.float64)
    if thresholds_by_jet is not None:
        thresholds = predictions["jet_num"].map(thresholds_by_jet).to_numpy(dtype=np.float64)
        y_pred = (scores >= thresholds).astype(np.int8)
        threshold_repr = json.dumps({str(key): float(value) for key, value in thresholds_by_jet.items()}, sort_keys=True)
    elif global_threshold is not None:
        y_pred = (scores >= float(global_threshold)).astype(np.int8)
        threshold_repr = f"{float(global_threshold):.12f}"
    else:
        raise ValueError("Falta threshold.")

    signal_weight = float(weights[(y_pred == 1) & (y_true == 1)].sum())
    background_weight = float(weights[(y_pred == 1) & (y_true == 0)].sum())
    return {
        "policy": policy,
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


def threshold_tables_by_jet(predictions: pd.DataFrame) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    for jet_value in JET_GROUPS:
        subset = predictions.loc[predictions["jet_num"] == jet_value]
        tables[jet_value] = cumulative_table(subset["y_true"], subset["score"].to_numpy(), subset["weight"])
    return tables


def optimize_joint_thresholds(predictions: pd.DataFrame, starting_thresholds: dict[int, float], max_iter: int = 8) -> dict[int, float]:
    tables = threshold_tables_by_jet(predictions)
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
            table = tables[jet_value]
            values = ams_score(
                fixed_signal + table["signal_weight"].to_numpy(dtype=np.float64),
                fixed_background + table["background_weight"].to_numpy(dtype=np.float64),
            )
            best_idx = int(np.argmax(values))
            new_threshold = float(table.iloc[best_idx]["threshold"])
            changed = changed or not np.isclose(thresholds[jet_value], new_threshold)
            thresholds[jet_value] = new_threshold
        if not changed:
            break
    return thresholds


def compare_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    global_info = best_threshold(cumulative_table(predictions["y_true"], predictions["score"].to_numpy(), predictions["weight"]))
    tables = threshold_tables_by_jet(predictions)
    independent = {jet: float(best_threshold(table)["threshold"]) for jet, table in tables.items()}
    joint = optimize_joint_thresholds(predictions, independent)
    rows = [
        threshold_row(predictions, "global_single_threshold", global_threshold=float(global_info["threshold"])),
        threshold_row(predictions, "independent_threshold_by_jet", thresholds_by_jet=independent),
        threshold_row(predictions, "joint_thresholds_from_independent", thresholds_by_jet=joint),
    ]
    return pd.DataFrame(rows).sort_values("ams", ascending=False).reset_index(drop=True)


def prepare_fold_group(
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    train_weight: pd.Series,
    valid_weight: pd.Series,
    valid_event_id: pd.Series,
    jet_value: int,
) -> dict[str, object]:
    train_jet = mix_module._normalize_jet_num(X_train_raw["PRI_jet_num"])
    valid_jet = mix_module._normalize_jet_num(X_valid_raw["PRI_jet_num"])
    train_mask = train_jet == jet_value
    valid_mask = valid_jet == jet_value

    X_train_group_raw = X_train_raw.loc[train_mask].reset_index(drop=True)
    X_valid_group_raw = X_valid_raw.loc[valid_mask].reset_index(drop=True)
    pipeline = mix_module._build_group_pipeline()
    pipeline.fit(X_train_group_raw)
    X_train_processed = pipeline.transform(X_train_group_raw).reset_index(drop=True)
    X_valid_processed = pipeline.transform(X_valid_group_raw).reset_index(drop=True)
    return {
        "jet_value": jet_value,
        "X_train_processed": X_train_processed,
        "X_valid_processed": X_valid_processed,
        "y_train": y_train.loc[train_mask].reset_index(drop=True),
        "y_valid": y_valid.loc[valid_mask].reset_index(drop=True),
        "train_weight": train_weight.loc[train_mask].reset_index(drop=True),
        "valid_weight": valid_weight.loc[valid_mask].reset_index(drop=True),
        "valid_event_id": valid_event_id.loc[valid_mask].reset_index(drop=True),
    }


def fit_group_model(group: dict[str, object], params: dict[str, object], use_sample_weight: bool = False) -> tuple[np.ndarray, float]:
    model = XGBClassifier(**params)
    y_train = label_to_binary(group["y_train"])
    fit_kwargs: dict[str, object] = {}
    if use_sample_weight:
        fit_kwargs["sample_weight"] = group["train_weight"].astype(np.float64)
    start = time.perf_counter()
    model.fit(group["X_train_processed"], y_train, **fit_kwargs)
    fit_seconds = time.perf_counter() - start
    scores = model.predict_proba(group["X_valid_processed"])[:, 1]
    return scores, fit_seconds


def evaluate_cv_jets() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    plan12_params = load_plan12_params()
    weighted14_params = load_weighted14_params()
    variant_specs = {
        "baseline_shared": {
            "params_by_jet": {jet: dict(BASELINE_PARAMS) for jet in JET_GROUPS},
            "use_sample_weight": False,
        },
        "plan12_tuned": {
            "params_by_jet": plan12_params,
            "use_sample_weight": False,
        },
        "weighted14": {
            "params_by_jet": weighted14_params,
            "use_sample_weight": True,
        },
    }

    train_data, _ = v2_module.load_higgs_data(project_root=PROJECT_ROOT)
    base = v2_module.split_higgs_features(train_data, train_data.drop(columns=["Weight", "Label"]))
    X_full = base["X_train_full"].reset_index(drop=True)
    y_full_label = base["y_train_full"].reset_index(drop=True)
    y_full = label_to_binary(y_full_label)
    weight_full = base["train_weight"].reset_index(drop=True).astype(np.float64)
    event_id_full = base["train_event_id"].reset_index(drop=True).astype(np.int64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    policy_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_full, y_full), start=1):
        X_train_raw = X_full.iloc[train_idx].reset_index(drop=True)
        X_valid_raw = X_full.iloc[valid_idx].reset_index(drop=True)
        y_train = y_full_label.iloc[train_idx].reset_index(drop=True)
        y_valid = y_full_label.iloc[valid_idx].reset_index(drop=True)
        train_weight = weight_full.iloc[train_idx].reset_index(drop=True)
        valid_weight = weight_full.iloc[valid_idx].reset_index(drop=True)
        valid_event_id = event_id_full.iloc[valid_idx].reset_index(drop=True)

        groups = {
            jet: prepare_fold_group(
                X_train_raw,
                X_valid_raw,
                y_train,
                y_valid,
                train_weight,
                valid_weight,
                valid_event_id,
                jet,
            )
            for jet in JET_GROUPS
        }

        for variant_name, spec in variant_specs.items():
            parts: list[pd.DataFrame] = []
            for jet_value, group in groups.items():
                params = spec["params_by_jet"][jet_value]
                scores, fit_seconds = fit_group_model(group, params, use_sample_weight=bool(spec["use_sample_weight"]))
                y_valid_group = label_to_binary(group["y_valid"])
                metrics = find_group_metrics(y_valid_group, scores, group["valid_weight"])
                group_rows.append(
                    {
                        "fold": fold,
                        "variant": variant_name,
                        "jet_num": jet_value,
                        "fit_seconds": fit_seconds,
                        "train_rows": int(len(group["y_train"])),
                        "valid_rows": int(len(group["y_valid"])),
                        **metrics,
                    }
                )
                parts.append(
                    pd.DataFrame(
                        {
                            "EventId": group["valid_event_id"].astype(np.int64),
                            "score": scores,
                            "jet_num": jet_value,
                            "y_true": y_valid_group.to_numpy(),
                            "weight": group["valid_weight"].to_numpy(dtype=np.float64),
                        }
                    )
                )
            predictions = pd.concat(parts, ignore_index=True).set_index("EventId").loc[valid_event_id].reset_index()
            policies = compare_policies(predictions)
            for _, row in policies.iterrows():
                policy_rows.append({"fold": fold, "variant": variant_name, **row.to_dict()})
            best = policies.iloc[0]
            print(
                f"fold={fold} variant={variant_name} best_policy={best['policy']} "
                f"ams={float(best['ams']):.6f} auc={float(best['auc']):.6f}"
            )

    policy_folds = pd.DataFrame(policy_rows)
    group_folds = pd.DataFrame(group_rows)
    policy_summary = (
        policy_folds.groupby(["variant", "policy"])
        .agg(
            ams_mean=("ams", "mean"),
            ams_std=("ams", "std"),
            ams_min=("ams", "min"),
            ams_max=("ams", "max"),
            auc_mean=("auc", "mean"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
        )
        .reset_index()
        .sort_values("ams_mean", ascending=False)
    )
    group_summary = (
        group_folds.groupby(["variant", "jet_num"])
        .agg(
            ams_mean=("ams", "mean"),
            ams_std=("ams", "std"),
            auc_mean=("auc", "mean"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
        )
        .reset_index()
    )
    policy_folds.to_csv(RESULTS_DIR / "01_policy_folds.csv", index=False)
    group_folds.to_csv(RESULTS_DIR / "01_group_folds.csv", index=False)
    policy_summary.to_csv(RESULTS_DIR / "01_policy_summary.csv", index=False)
    group_summary.to_csv(RESULTS_DIR / "01_group_summary.csv", index=False)
    (RESULTS_DIR / "01_summary.txt").write_text(
        "\n".join(
            [
                "# CV jets",
                "",
                "## Policy summary",
                policy_summary.to_string(index=False),
                "",
                "## Group summary",
                group_summary.to_string(index=False),
            ]
        )
        + "\n"
    )
    dump_json(
        RESULTS_DIR / "01_metadata.json",
        {
            "n_splits": N_SPLITS,
            "seed": SEED,
            "variants": list(variant_specs),
            "note": "Each fold splits globally first, then fits one preprocessor per jet on fold train only.",
        },
    )
    return {"policy_folds": policy_folds, "group_folds": group_folds, "policy_summary": policy_summary, "group_summary": group_summary}


def find_group_metrics(y_true: pd.Series, scores: np.ndarray, weights: pd.Series) -> dict[str, float | int]:
    info = best_threshold(cumulative_table(y_true, scores, weights))
    y_array = np.asarray(y_true, dtype=np.int8)
    y_pred = (scores >= float(info["threshold"])).astype(np.int8)
    return {
        **info,
        "accuracy": float(accuracy_score(y_array, y_pred)),
        "precision": float(precision_score(y_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_array, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_array, scores)),
    }


def main() -> None:
    outputs = evaluate_cv_jets()
    print("\nPolicy summary:")
    print(outputs["policy_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
