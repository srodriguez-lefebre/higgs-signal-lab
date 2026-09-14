from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from i3_tree_safe_mix import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_VALID_SIZE,
    build_raw_missing_summary,
    build_split_summary,
    create_train_valid_split,
    find_project_root,
    get_higgs_data_dir,
    load_higgs_data,
    split_higgs_features,
    I3TreeSafeMixPreprocessor,
)

JET_GROUPS = [0, 1, 2, 3]


def _normalize_jet_num(series: pd.Series) -> pd.Series:
    values = series.astype("Int64")
    invalid = values[~values.isin(JET_GROUPS)]
    if not invalid.empty:
        unique_invalid = sorted({int(value) for value in invalid.dropna().tolist()})
        raise ValueError(f"Se encontraron valores de PRI_jet_num fuera de {JET_GROUPS}: {unique_invalid}")
    return values.astype(int)


def _slice_group(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return frame.loc[mask].reset_index(drop=True)


def _slice_group_series(series: pd.Series, mask: pd.Series) -> pd.Series:
    return series.loc[mask].reset_index(drop=True)


def _build_group_pipeline() -> I3TreeSafeMixPreprocessor:
    return I3TreeSafeMixPreprocessor()


def run_i3_tree_safe_mix_by_jets(
    project_root: Path | None = None,
    valid_size: float = DEFAULT_VALID_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Builds four jet-specific datasets using the same preprocessing strategy as 11.Mezcla/01."""
    train_data, test_data = load_higgs_data(project_root=project_root)
    base = split_higgs_features(train_data, test_data)
    split = create_train_valid_split(
        X_full=base["X_train_full"],
        y_full=base["y_train_full"],
        train_event_id=base["train_event_id"],
        train_weight=base["train_weight"],
        valid_size=valid_size,
        random_state=random_state,
    )

    X_train_global_raw = split["X_train_raw"].reset_index(drop=True)
    X_valid_global_raw = split["X_valid_raw"].reset_index(drop=True)
    X_test_global_raw = base["X_test_raw"].reset_index(drop=True)

    train_jet_num = _normalize_jet_num(X_train_global_raw["PRI_jet_num"])
    valid_jet_num = _normalize_jet_num(X_valid_global_raw["PRI_jet_num"])
    test_jet_num = _normalize_jet_num(X_test_global_raw["PRI_jet_num"])

    group_artifacts: dict[int, dict[str, object]] = {}
    group_summary_rows: list[dict[str, object]] = []

    for jet_value in JET_GROUPS:
        train_mask = train_jet_num == jet_value
        valid_mask = valid_jet_num == jet_value
        test_mask = test_jet_num == jet_value

        X_train_group_raw = _slice_group(X_train_global_raw, train_mask)
        X_valid_group_raw = _slice_group(X_valid_global_raw, valid_mask)
        X_test_group_raw = _slice_group(X_test_global_raw, test_mask)

        y_train_group = _slice_group_series(split["y_train"], train_mask)
        y_valid_group = _slice_group_series(split["y_valid"], valid_mask)
        train_weight_group = _slice_group_series(split["train_weight"], train_mask)
        valid_weight_group = _slice_group_series(split["valid_weight"], valid_mask)
        train_event_id_group = _slice_group_series(split["train_event_id"], train_mask)
        valid_event_id_group = _slice_group_series(split["valid_event_id"], valid_mask)
        test_event_id_group = _slice_group_series(base["test_event_id"], test_mask)

        if X_train_group_raw.empty:
            raise ValueError(f"El subconjunto de train para PRI_jet_num={jet_value} quedó vacío.")

        pipeline = _build_group_pipeline()
        pipeline.fit(X_train_group_raw)

        X_train_group_processed = pipeline.transform(X_train_group_raw).reset_index(drop=True)
        X_valid_group_processed = pipeline.transform(X_valid_group_raw).reset_index(drop=True)
        X_test_group_processed = pipeline.transform(X_test_group_raw).reset_index(drop=True)

        preprocess_step = pipeline
        # Pipeline.fit/transform above operates directly on the custom transformer.
        if hasattr(pipeline, "get_feature_names_out"):
            processed_feature_names = pipeline.get_feature_names_out().tolist()
        else:
            processed_feature_names = list(X_train_group_processed.columns)

        group_artifacts[jet_value] = {
            "jet_value": jet_value,
            "preprocessing_pipeline": pipeline,
            "processed_feature_names": processed_feature_names,
            "engineered_feature_columns": pipeline.engineered_feature_columns_,
            "rank_source_columns": pipeline.rank_columns_,
            "missing_indicator_source_columns": pipeline.missing_indicator_adder_.source_columns_,
            "X_train_raw": X_train_group_raw,
            "X_valid_raw": X_valid_group_raw,
            "X_test_raw": X_test_group_raw,
            "X_train_processed": X_train_group_processed,
            "X_valid_processed": X_valid_group_processed,
            "X_test_processed": X_test_group_processed,
            "y_train": y_train_group,
            "y_valid": y_valid_group,
            "train_weight": train_weight_group,
            "valid_weight": valid_weight_group,
            "train_event_id": train_event_id_group,
            "valid_event_id": valid_event_id_group,
            "test_event_id": test_event_id_group,
            "train_missing_summary_df": build_raw_missing_summary(X_train_group_raw),
            "valid_missing_summary_df": build_raw_missing_summary(X_valid_group_raw),
            "test_missing_summary_df": build_raw_missing_summary(X_test_group_raw),
        }

        group_summary_rows.append(
            {
                "jet_num": jet_value,
                "train_rows": int(len(X_train_group_raw)),
                "valid_rows": int(len(X_valid_group_raw)),
                "test_rows": int(len(X_test_group_raw)),
                "processed_feature_count": int(len(processed_feature_names)),
                "engineered_feature_count": int(len(pipeline.engineered_feature_columns_)),
                "missing_indicator_count": int(len(pipeline.missing_indicator_adder_.source_columns_)),
                "train_nan_count": int(X_train_group_processed.isna().sum().sum()),
                "valid_nan_count": int(X_valid_group_processed.isna().sum().sum()),
                "test_nan_count": int(X_test_group_processed.isna().sum().sum()),
            }
        )

    split_summary_df = build_split_summary(
        y_train=split["y_train"],
        y_valid=split["y_valid"],
        train_weight=split["train_weight"],
        valid_weight=split["valid_weight"],
        test_rows=len(base["X_test_raw"]),
    )
    group_summary_df = pd.DataFrame(group_summary_rows).sort_values("jet_num").reset_index(drop=True)

    metadata = {
        "variant_key": "i3_tree_safe_mix_by_jets",
        "variant_title": "Mezcla i3 + tree_safe separada por jets",
        "description": (
            "Usa el mismo preprocesador de 11.Mezcla/01, pero ajusta un pipeline independiente por cada "
            "valor de PRI_jet_num = 0, 1, 2 y 3. El ruteo de inferencia se hace con la cantidad de jets cruda."
        ),
        "valid_size": float(valid_size),
        "random_state": int(random_state),
        "raw_feature_count": int(len(base["feature_columns"])),
        "jet_groups": JET_GROUPS,
        "group_count": len(JET_GROUPS),
    }

    return {
        "metadata": metadata,
        "feature_columns": base["feature_columns"],
        "split_summary_df": split_summary_df,
        "group_summary_df": group_summary_df,
        "group_artifacts": group_artifacts,
        "global_train_event_id": split["train_event_id"].reset_index(drop=True),
        "global_valid_event_id": split["valid_event_id"].reset_index(drop=True),
        "global_test_event_id": base["test_event_id"].reset_index(drop=True),
        "global_y_train": split["y_train"].reset_index(drop=True),
        "global_y_valid": split["y_valid"].reset_index(drop=True),
        "global_train_weight": split["train_weight"].reset_index(drop=True),
        "global_valid_weight": split["valid_weight"].reset_index(drop=True),
    }


def save_i3_tree_safe_mix_by_jets_artifacts(
    artifacts: dict[str, object],
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    """Saves per-jet processed datasets and fitted preprocessors to disk."""
    if output_dir is None:
        output_dir = get_higgs_data_dir(project_root) / "processed" / "i3_tree_safe_mix_by_jets"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(artifacts["metadata"], indent=2, ensure_ascii=False))
    saved_paths.append(metadata_path)

    split_summary_path = output_dir / "split_summary.csv"
    artifacts["split_summary_df"].to_csv(split_summary_path, index=False)
    saved_paths.append(split_summary_path)

    group_summary_path = output_dir / "group_summary.csv"
    artifacts["group_summary_df"].to_csv(group_summary_path, index=False)
    saved_paths.append(group_summary_path)

    for jet_value, group in artifacts["group_artifacts"].items():
        group_dir = output_dir / f"jet_{jet_value}"
        group_dir.mkdir(parents=True, exist_ok=True)

        files_to_save: dict[str, pd.DataFrame] = {
            "X_train_processed.csv": group["X_train_processed"],
            "X_valid_processed.csv": group["X_valid_processed"],
            "X_test_processed.csv": group["X_test_processed"],
            "y_train.csv": group["y_train"].to_frame(name="Label"),
            "y_valid.csv": group["y_valid"].to_frame(name="Label"),
            "train_weight.csv": group["train_weight"].to_frame(name="Weight"),
            "valid_weight.csv": group["valid_weight"].to_frame(name="Weight"),
            "train_event_id.csv": group["train_event_id"].to_frame(name="EventId"),
            "valid_event_id.csv": group["valid_event_id"].to_frame(name="EventId"),
            "test_event_id.csv": group["test_event_id"].to_frame(name="EventId"),
            "processed_feature_names.csv": pd.DataFrame({"feature": group["processed_feature_names"]}),
            "engineered_feature_columns.csv": pd.DataFrame({"feature": group["engineered_feature_columns"]}),
            "rank_source_columns.csv": pd.DataFrame({"feature": group["rank_source_columns"]}),
            "missing_indicator_source_columns.csv": pd.DataFrame(
                {"feature": group["missing_indicator_source_columns"]}
            ),
            "train_missing_summary.csv": group["train_missing_summary_df"],
            "valid_missing_summary.csv": group["valid_missing_summary_df"],
            "test_missing_summary.csv": group["test_missing_summary_df"],
        }

        for filename, dataframe in files_to_save.items():
            file_path = group_dir / filename
            dataframe.to_csv(file_path, index=False)
            saved_paths.append(file_path)

        pipeline_path = group_dir / "preprocessing_pipeline.joblib"
        joblib.dump(group["preprocessing_pipeline"], pipeline_path)
        saved_paths.append(pipeline_path)

    return saved_paths
