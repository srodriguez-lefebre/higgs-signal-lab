from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = PROJECT_ROOT / "17.cv_jets"
RESULTS_DIR = STAGE_DIR / "results"
MODELS_DIR = STAGE_DIR / "models" / "weighted_jets_full_cv_thresholds"
JET_GROUPS = [0, 1, 2, 3]

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


mix_module = load_module("full_cv_i3_tree_safe_mix_by_jets", MIX_DIR / "i3_tree_safe_mix_by_jets.py")
v2_module = load_module("full_cv_higgs_preprocessing_v2", TRY_DIR / "higgs_preprocessing_v2.py")


def load_weighted14_config() -> tuple[dict[int, dict[str, object]], dict[int, bool]]:
    metadata = json.loads((PROJECT_ROOT / "14.invest" / "models" / "weighted_xgboost_by_jets_bundle" / "metadata.json").read_text())
    params_by_jet = {int(key): dict(value) for key, value in metadata["best_params_by_jet"].items()}
    best_by_jet = pd.read_csv(PROJECT_ROOT / "14.invest" / "results" / "01_best_by_jet.csv")
    sample_weight_by_jet = {
        int(row["jet_num"]): bool(row["use_sample_weight"])
        for _, row in best_by_jet.iterrows()
    }
    return params_by_jet, sample_weight_by_jet


def cv_thresholds(policy: str = "joint_thresholds_from_independent") -> dict[str, dict[int, float]]:
    folds = pd.read_csv(RESULTS_DIR / "01_policy_folds.csv")
    selected = folds[(folds["variant"] == "weighted14") & (folds["policy"] == policy)]
    threshold_rows = [json.loads(raw) for raw in selected["thresholds"]]
    values_by_jet: dict[int, list[float]] = {jet: [] for jet in JET_GROUPS}
    for row in threshold_rows:
        for jet in JET_GROUPS:
            values_by_jet[jet].append(float(row[str(jet)]))
    return {
        "cv_median": {jet: float(np.median(values)) for jet, values in values_by_jet.items()},
        "cv_mean": {jet: float(np.mean(values)) for jet, values in values_by_jet.items()},
    }


def build_submission(test_predictions: pd.DataFrame, thresholds_by_jet: dict[int, float]) -> pd.DataFrame:
    scores = test_predictions["score"].to_numpy(dtype=np.float64)
    thresholds = test_predictions["jet_num"].map(thresholds_by_jet).to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "EventId": test_predictions["EventId"].astype(np.int64),
            "RankOrder": pd.Series(scores).rank(method="first", ascending=True).astype(np.int64),
            "Class": np.where(scores >= thresholds, "s", "b"),
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
    return {
        "rows": int(len(submission)),
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "signal_count": int((submission["Class"] == "s").sum()),
        "background_count": int((submission["Class"] == "b").sum()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    params_by_jet, sample_weight_by_jet = load_weighted14_config()
    threshold_sets = cv_thresholds()

    train_data, test_data = v2_module.load_higgs_data(project_root=PROJECT_ROOT)
    base = v2_module.split_higgs_features(train_data, test_data)
    X_full = base["X_train_full"].reset_index(drop=True)
    y_full = base["y_train_full"].map({"b": 0, "s": 1}).astype(np.int8).reset_index(drop=True)
    weight_full = base["train_weight"].reset_index(drop=True).astype(np.float64)
    X_test = base["X_test_raw"].reset_index(drop=True)
    test_event_id = base["test_event_id"].reset_index(drop=True).astype(np.int64)
    train_jet = mix_module._normalize_jet_num(X_full["PRI_jet_num"])
    test_jet = mix_module._normalize_jet_num(X_test["PRI_jet_num"])

    parts: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for jet_value in JET_GROUPS:
        train_mask = train_jet == jet_value
        test_mask = test_jet == jet_value
        X_train_group = X_full.loc[train_mask].reset_index(drop=True)
        X_test_group = X_test.loc[test_mask].reset_index(drop=True)
        y_train_group = y_full.loc[train_mask].reset_index(drop=True)
        weight_group = weight_full.loc[train_mask].reset_index(drop=True)
        pipeline = mix_module._build_group_pipeline()
        pipeline.fit(X_train_group)
        X_train_processed = pipeline.transform(X_train_group).reset_index(drop=True)
        X_test_processed = pipeline.transform(X_test_group).reset_index(drop=True)

        model = XGBClassifier(**params_by_jet[jet_value])
        fit_kwargs: dict[str, object] = {}
        if sample_weight_by_jet[jet_value]:
            fit_kwargs["sample_weight"] = weight_group
        start = time.perf_counter()
        model.fit(X_train_processed, y_train_group, **fit_kwargs)
        fit_seconds = time.perf_counter() - start
        scores = model.predict_proba(X_test_processed)[:, 1]

        group_dir = MODELS_DIR / f"jet_{jet_value}"
        group_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, group_dir / "xgboost_model.joblib")
        joblib.dump(pipeline, group_dir / "preprocessing_pipeline.joblib")
        fit_rows.append(
            {
                "jet_num": jet_value,
                "train_rows": int(len(X_train_group)),
                "test_rows": int(len(X_test_group)),
                "feature_count": int(X_train_processed.shape[1]),
                "use_sample_weight": sample_weight_by_jet[jet_value],
                "fit_seconds": float(fit_seconds),
            }
        )
        parts.append(
            pd.DataFrame(
                {
                    "EventId": test_event_id.loc[test_mask].reset_index(drop=True),
                    "jet_num": jet_value,
                    "score": scores,
                }
            )
        )

    test_predictions = pd.concat(parts, ignore_index=True).set_index("EventId").loc[test_event_id].reset_index()
    test_predictions.to_csv(RESULTS_DIR / "03_weighted_jets_full_test_scores.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(RESULTS_DIR / "03_weighted_jets_full_fit_summary.csv", index=False)

    metadata: dict[str, object] = {
        "threshold_sets": threshold_sets,
        "sample_weight_by_jet": sample_weight_by_jet,
        "submissions": {},
    }
    for threshold_name, thresholds in threshold_sets.items():
        submission = build_submission(test_predictions, thresholds)
        validation = validate_submission(submission)
        submission_path = RESULTS_DIR / f"higgs_submission_weighted_jets_full_{threshold_name}.csv"
        submission.to_csv(submission_path, index=False)
        metadata["submissions"][threshold_name] = {
            "path": str(submission_path),
            "thresholds": thresholds,
            "validation": validation,
        }
        print(f"{threshold_name}: {validation} -> {submission_path}")

    (RESULTS_DIR / "03_weighted_jets_full_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
