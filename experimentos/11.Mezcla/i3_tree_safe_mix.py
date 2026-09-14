from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _load_base_module():
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        module_path = candidate / "10.try" / "higgs_preprocessing_v2.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("_higgs_preprocessing_v2_base", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"No se pudo crear el spec para: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("No se pudo ubicar 10.try/higgs_preprocessing_v2.py desde 11.Mezcla.")


_BASE = _load_base_module()

MISSING_MARKER = _BASE.MISSING_MARKER
JET_NUM_COLUMN = _BASE.JET_NUM_COLUMN
DEFAULT_VALID_SIZE = _BASE.DEFAULT_VALID_SIZE
DEFAULT_RANDOM_STATE = _BASE.DEFAULT_RANDOM_STATE
EPSILON = _BASE.EPSILON

find_project_root = _BASE.find_project_root
get_higgs_data_dir = _BASE.get_higgs_data_dir
load_higgs_data = _BASE.load_higgs_data
split_higgs_features = _BASE.split_higgs_features
create_train_valid_split = _BASE.create_train_valid_split
build_raw_missing_summary = _BASE.build_raw_missing_summary
build_split_summary = _BASE.build_split_summary
wrap_delta_phi = _BASE.wrap_delta_phi
safe_divide = _BASE.safe_divide
MissingIndicatorAdder = _BASE.MissingIndicatorAdder


MIX_SQUARED_COLUMNS = [
    "PRI_tau_pt",
    "PRI_lep_pt",
    "PRI_met",
    "DER_mass_MMC",
    "PRI_jet_leading_pt",
    "HT",
    "jets_pt_sum",
    "deltaR_tau_lep",
]

MIX_RANK_COLUMNS = [
    "DER_mass_MMC",
    "PRI_tau_pt",
    "PRI_lep_pt",
    "PRI_met",
    "HT",
    "deltaR_tau_lep",
    "jets_pt_sum",
]


def add_i3_tree_safe_core_features(
    data: pd.DataFrame,
    jet_num_column: str = JET_NUM_COLUMN,
    epsilon: float = EPSILON,
    squared_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Builds the core engineered features inspired by i3 while preserving NaN."""
    required_columns = [
        "DER_mass_MMC",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_jet_leading_pt",
        "PRI_jet_subleading_pt",
        jet_num_column,
    ]
    missing_required = [column for column in required_columns if column not in data.columns]
    if missing_required:
        raise ValueError(f"Faltan columnas requeridas para generar features mezcla: {missing_required}")

    squared_columns = squared_columns or MIX_SQUARED_COLUMNS

    X = data.copy()

    leading_pt = X["PRI_jet_leading_pt"].fillna(0.0)
    subleading_pt = X["PRI_jet_subleading_pt"].fillna(0.0)
    visible_energy = X["PRI_tau_pt"] + X["PRI_lep_pt"]
    jets_pt_sum = leading_pt + subleading_pt
    ht = visible_energy + X["PRI_met"] + jets_pt_sum
    delta_eta_tau_lep = X["PRI_tau_eta"] - X["PRI_lep_eta"]
    delta_phi_tau_lep = wrap_delta_phi(X["PRI_tau_phi"] - X["PRI_lep_phi"])
    abs_phi_diff_tau_lep = delta_phi_tau_lep.abs()
    has_jet = (X[jet_num_column].fillna(0) > 0).astype(np.float32)

    X["ratio_tau_lep_pt"] = safe_divide(X["PRI_tau_pt"], X["PRI_lep_pt"], epsilon=epsilon)
    X["ratio_tau_met"] = safe_divide(X["PRI_tau_pt"], X["PRI_met"], epsilon=epsilon)
    X["ratio_lep_met"] = safe_divide(X["PRI_lep_pt"], X["PRI_met"], epsilon=epsilon)
    X["ratio_mass_pt"] = safe_divide(X["DER_mass_MMC"], visible_energy, epsilon=epsilon)
    X["jet_lep_ratio"] = safe_divide(leading_pt, X["PRI_lep_pt"], epsilon=epsilon)
    X["HT"] = ht
    X["visible_energy"] = visible_energy
    X["met_fraction"] = safe_divide(X["PRI_met"], ht, epsilon=epsilon)
    X["abs_eta_diff_tau_lep"] = delta_eta_tau_lep.abs()
    X["delta_phi_tau_lep"] = delta_phi_tau_lep
    X["abs_phi_diff_tau_lep"] = abs_phi_diff_tau_lep
    X["deltaR_tau_lep"] = np.sqrt(X["abs_eta_diff_tau_lep"] ** 2 + abs_phi_diff_tau_lep ** 2)
    X["jets_pt_sum"] = jets_pt_sum
    X["jet_lepton_balance"] = safe_divide(jets_pt_sum, X["PRI_lep_pt"], epsilon=epsilon)
    X["tau_lep_interaction"] = X["PRI_tau_pt"] * X["PRI_lep_pt"]
    X["tau_met_interaction"] = X["PRI_tau_pt"] * X["PRI_met"]
    X["has_jet"] = has_jet
    X["ht_if_jets"] = ht * has_jet

    for column in squared_columns:
        if column in X.columns:
            X[f"{column}_sq"] = X[column] ** 2

    return X


class I3TreeSafeMixPreprocessor(BaseEstimator, TransformerMixin):
    """Hybrid preprocessor: i3-style physical features plus tree_safe methodology."""

    def __init__(
        self,
        jet_num_column: str = JET_NUM_COLUMN,
        squared_columns: list[str] | None = None,
        rank_columns: list[str] | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        self.jet_num_column = jet_num_column
        self.squared_columns = squared_columns
        self.rank_columns = rank_columns
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "I3TreeSafeMixPreprocessor":
        frame = self._validate_frame(X)
        self.feature_columns_ = list(frame.columns)
        self.base_engineered_columns_ = None
        self.rank_columns_ = []
        expanded = self._expand_with_rank_features(frame, fitting=True)
        self.numeric_columns_ = [column for column in expanded.columns if column != self.jet_num_column]

        self.missing_indicator_adder_ = MissingIndicatorAdder(exclude_columns=[self.jet_num_column])
        expanded_with_flags = self.missing_indicator_adder_.fit_transform(expanded)
        self.missing_indicator_columns_ = [
            column for column in expanded_with_flags.columns
            if column.startswith("is_missing_")
        ]

        self.jet_encoder_ = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )
        self.jet_encoder_.fit(frame[[self.jet_num_column]].fillna("__missing__").astype(str))
        self.jet_encoded_columns_ = self.jet_encoder_.get_feature_names_out([self.jet_num_column]).tolist()

        self.processed_feature_names_ = (
            self.numeric_columns_
            + self.missing_indicator_columns_
            + self.jet_encoded_columns_
        )
        self.engineered_feature_columns_ = [
            column for column in expanded.columns
            if column not in self.feature_columns_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_frame(X)
        expanded = self._expand_with_rank_features(frame, fitting=False)
        expanded_with_flags = self.missing_indicator_adder_.transform(expanded)

        numeric_frame = expanded_with_flags[self.numeric_columns_].copy()
        missing_frame = expanded_with_flags[self.missing_indicator_columns_].copy()
        jet_encoded = self.jet_encoder_.transform(frame[[self.jet_num_column]].fillna("__missing__").astype(str))
        jet_frame = pd.DataFrame(
            jet_encoded,
            columns=self.jet_encoded_columns_,
            index=frame.index,
        )

        transformed = pd.concat([numeric_frame, missing_frame, jet_frame], axis=1)
        return transformed.astype(np.float32)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.processed_feature_names_, dtype=object)

    def _expand_with_rank_features(self, X: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        expanded = add_i3_tree_safe_core_features(
            X,
            jet_num_column=self.jet_num_column,
            epsilon=self.epsilon,
            squared_columns=self.squared_columns or MIX_SQUARED_COLUMNS,
        )

        if self.base_engineered_columns_ is None:
            self.base_engineered_columns_ = list(expanded.columns)
        else:
            expanded = expanded[self.base_engineered_columns_].copy()

        rank_source_columns = self.rank_columns or MIX_RANK_COLUMNS
        usable_rank_columns = [column for column in rank_source_columns if column in expanded.columns]

        if fitting:
            self.rank_columns_ = usable_rank_columns
            self.rank_reference_ = {}
            for column in self.rank_columns_:
                values = expanded[column].dropna().to_numpy(dtype=np.float64)
                self.rank_reference_[column] = np.sort(values)

        if not hasattr(self, "rank_reference_"):
            raise RuntimeError("El preprocesador no fue ajustado antes de transformar.")

        for column in self.rank_columns_:
            sorted_values = self.rank_reference_[column]
            rank_values = np.full(len(expanded), np.nan, dtype=np.float32)
            if sorted_values.size:
                column_values = expanded[column].to_numpy(dtype=np.float64)
                valid_mask = ~np.isnan(column_values)
                if valid_mask.any():
                    positions = np.searchsorted(sorted_values, column_values[valid_mask], side="right")
                    rank_values[valid_mask] = positions / float(sorted_values.size)
            expanded[f"{column}_rank"] = rank_values

        return expanded

    def _validate_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("I3TreeSafeMixPreprocessor espera un pandas.DataFrame.")
        if self.jet_num_column not in X.columns:
            raise ValueError(f"Falta la columna requerida: {self.jet_num_column}")
        if hasattr(self, "feature_columns_"):
            missing = [column for column in self.feature_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"Faltan columnas esperadas para transformar: {missing}")
            return X[self.feature_columns_].copy()
        return X.copy()


def build_i3_tree_safe_mix_pipeline() -> Pipeline:
    """Returns the sklearn pipeline for the hybrid preprocessing."""
    return Pipeline([("preprocess", I3TreeSafeMixPreprocessor())])


def _to_processed_frame(
    pipeline: Pipeline,
    data: pd.DataFrame,
) -> pd.DataFrame:
    processed = pipeline.transform(data)
    if isinstance(processed, pd.DataFrame):
        return processed
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    return pd.DataFrame(processed, columns=feature_names, index=data.index)


def run_i3_tree_safe_mix(
    project_root: Path | None = None,
    valid_size: float = DEFAULT_VALID_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Runs the hybrid preprocessing end-to-end."""
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

    pipeline = build_i3_tree_safe_mix_pipeline()
    pipeline.fit(split["X_train_raw"])

    X_train_processed = _to_processed_frame(pipeline, split["X_train_raw"]).reset_index(drop=True)
    X_valid_processed = _to_processed_frame(pipeline, split["X_valid_raw"]).reset_index(drop=True)
    X_test_processed = _to_processed_frame(pipeline, base["X_test_raw"]).reset_index(drop=True)

    preprocess_step = pipeline.named_steps["preprocess"]
    split_summary_df = build_split_summary(
        y_train=split["y_train"],
        y_valid=split["y_valid"],
        train_weight=split["train_weight"],
        valid_weight=split["valid_weight"],
        test_rows=len(base["X_test_raw"]),
    )

    metadata = {
        "variant_key": "i3_tree_safe_mix",
        "variant_title": "Mezcla i3 + tree_safe",
        "description": (
            "Features fisicas inspiradas en i3, ranks ajustados solo con train, NaN conservados, "
            "missing flags, split temprano y one-hot de PRI_jet_num ajustado solo sobre train."
        ),
        "valid_size": float(valid_size),
        "random_state": int(random_state),
        "raw_feature_count": int(len(base["feature_columns"])),
        "engineered_feature_count": int(len(preprocess_step.engineered_feature_columns_)),
        "processed_feature_count": int(len(X_train_processed.columns)),
        "rank_feature_count": int(len(preprocess_step.rank_columns_)),
        "missing_indicator_count": int(len(preprocess_step.missing_indicator_columns_)),
        "processed_train_nan_count": int(X_train_processed.isna().sum().sum()),
        "processed_valid_nan_count": int(X_valid_processed.isna().sum().sum()),
        "processed_test_nan_count": int(X_test_processed.isna().sum().sum()),
    }

    return {
        "metadata": metadata,
        "feature_columns": base["feature_columns"],
        "engineered_feature_columns": preprocess_step.engineered_feature_columns_,
        "rank_source_columns": preprocess_step.rank_columns_,
        "processed_feature_names": X_train_processed.columns.tolist(),
        "missing_indicator_source_columns": preprocess_step.missing_indicator_adder_.source_columns_,
        "preprocessing_pipeline": pipeline,
        "split_summary_df": split_summary_df,
        "train_missing_summary_df": build_raw_missing_summary(split["X_train_raw"]),
        "valid_missing_summary_df": build_raw_missing_summary(split["X_valid_raw"]),
        "test_missing_summary_df": build_raw_missing_summary(base["X_test_raw"]),
        "X_train_processed": X_train_processed,
        "X_valid_processed": X_valid_processed,
        "X_test_processed": X_test_processed,
        "y_train": split["y_train"],
        "y_valid": split["y_valid"],
        "train_weight": split["train_weight"],
        "valid_weight": split["valid_weight"],
        "train_event_id": split["train_event_id"],
        "valid_event_id": split["valid_event_id"],
        "test_event_id": base["test_event_id"].reset_index(drop=True),
    }


def save_i3_tree_safe_mix_artifacts(
    artifacts: dict[str, object],
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    """Saves processed data, metadata and the fitted pipeline."""
    if output_dir is None:
        output_dir = get_higgs_data_dir(project_root) / "processed" / "i3_tree_safe_mix"
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_save: dict[str, pd.DataFrame] = {
        "X_train_processed.csv": artifacts["X_train_processed"],
        "X_valid_processed.csv": artifacts["X_valid_processed"],
        "X_test_processed.csv": artifacts["X_test_processed"],
        "y_train.csv": artifacts["y_train"].to_frame(name="Label"),
        "y_valid.csv": artifacts["y_valid"].to_frame(name="Label"),
        "train_weight.csv": artifacts["train_weight"].to_frame(name="Weight"),
        "valid_weight.csv": artifacts["valid_weight"].to_frame(name="Weight"),
        "train_event_id.csv": artifacts["train_event_id"].to_frame(name="EventId"),
        "valid_event_id.csv": artifacts["valid_event_id"].to_frame(name="EventId"),
        "test_event_id.csv": artifacts["test_event_id"].to_frame(name="EventId"),
        "feature_columns.csv": pd.DataFrame({"feature": artifacts["feature_columns"]}),
        "engineered_feature_columns.csv": pd.DataFrame({"feature": artifacts["engineered_feature_columns"]}),
        "rank_source_columns.csv": pd.DataFrame({"feature": artifacts["rank_source_columns"]}),
        "missing_indicator_source_columns.csv": pd.DataFrame(
            {"feature": artifacts["missing_indicator_source_columns"]}
        ),
        "processed_feature_names.csv": pd.DataFrame({"feature": artifacts["processed_feature_names"]}),
        "split_summary.csv": artifacts["split_summary_df"],
        "train_missing_summary.csv": artifacts["train_missing_summary_df"],
        "valid_missing_summary.csv": artifacts["valid_missing_summary_df"],
        "test_missing_summary.csv": artifacts["test_missing_summary_df"],
    }

    saved_paths: list[Path] = []
    for filename, dataframe in files_to_save.items():
        file_path = output_dir / filename
        dataframe.to_csv(file_path, index=False)
        saved_paths.append(file_path)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(artifacts["metadata"], indent=2, ensure_ascii=False))
    saved_paths.append(metadata_path)

    pipeline_path = output_dir / "preprocessing_pipeline.joblib"
    joblib.dump(artifacts["preprocessing_pipeline"], pipeline_path)
    saved_paths.append(pipeline_path)

    return saved_paths
