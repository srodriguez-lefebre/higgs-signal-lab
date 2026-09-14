from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

MISSING_MARKER = -999.0
JET_NUM_COLUMN = "PRI_jet_num"
DEFAULT_VALID_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42
EPSILON = 1e-6

RAW_ONE_JET_COLUMNS = [
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
]

RAW_TWO_JET_COLUMNS = [
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_lep_eta_centrality",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
]


@dataclass(frozen=True)
class VariantDefinition:
    key: str
    title: str
    description: str
    output_dir_name: str
    expects_missing_after_processing: bool


VARIANT_DEFINITIONS: dict[str, VariantDefinition] = {
    "tree_safe": VariantDefinition(
        key="tree_safe",
        title="Procesamiento V2 - Seguro para Arboles",
        description=(
            "Mantiene los NaN para modelos de arboles, agrega indicadores de missing, "
            "trata PRI_jet_num como categorica y evita imputacion o escalado numerico."
        ),
        output_dir_name="v2_tree_safe",
        expects_missing_after_processing=True,
    ),
    "nn_dense": VariantDefinition(
        key="nn_dense",
        title="Procesamiento V2 - Pipeline para Redes Neuronales",
        description=(
            "Agrega indicadores de missing, aplica imputacion jet-aware aprendida solo en train, "
            "codifica PRI_jet_num y escala features numericas con RobustScaler."
        ),
        output_dir_name="v2_nn_dense",
        expects_missing_after_processing=False,
    ),
    "physics_by_jet": VariantDefinition(
        key="physics_by_jet",
        title="Procesamiento V2 - Features Fisicas por Numero de Jets",
        description=(
            "Suma features fisicas y features separadas por regimen de jets, agrega indicadores de missing, "
            "aplica imputacion jet-aware y escala numericas con RobustScaler."
        ),
        output_dir_name="v2_physics_by_jet",
        expects_missing_after_processing=False,
    ),
}


def find_project_root(start: Path | None = None) -> Path:
    """Locates the repository root by looking for roadmap.md and the data directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "roadmap.md").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


def get_higgs_data_dir(project_root: Path | None = None) -> Path:
    """Returns the local Higgs dataset directory."""
    root = find_project_root(project_root)
    data_dir = root / "data" / "higgs-boson"
    if not data_dir.exists():
        raise FileNotFoundError(f"No se encontro el directorio esperado: {data_dir}")
    return data_dir


def replace_missing_marker_with_nan(
    data: pd.DataFrame,
    missing_marker: float = MISSING_MARKER,
) -> pd.DataFrame:
    """Replaces the challenge missing marker with NaN immediately after loading."""
    return data.replace(missing_marker, np.nan)


def load_higgs_data(
    project_root: Path | None = None,
    missing_marker: float = MISSING_MARKER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train/test CSVs and replaces the challenge missing marker with NaN."""
    data_dir = get_higgs_data_dir(project_root)
    train_path = data_dir / "training.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo esperado: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo esperado: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return (
        replace_missing_marker_with_nan(train_data, missing_marker=missing_marker),
        replace_missing_marker_with_nan(test_data, missing_marker=missing_marker),
    )


def split_higgs_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, object]:
    """Separates labels, weights and features from the raw challenge files."""
    excluded_train_columns = ["EventId", "Weight", "Label"]
    feature_columns = [
        column for column in train_data.columns
        if column not in excluded_train_columns
    ]

    expected_test_columns = [column for column in test_data.columns if column != "EventId"]
    if feature_columns != expected_test_columns:
        raise ValueError("Las columnas de entrenamiento y test no coinciden despues de excluir EventId.")

    return {
        "feature_columns": feature_columns,
        "train_event_id": train_data["EventId"].copy(),
        "train_weight": train_data["Weight"].copy(),
        "y_train_full": train_data["Label"].copy(),
        "X_train_full": train_data[feature_columns].copy(),
        "test_event_id": test_data["EventId"].copy(),
        "X_test_raw": test_data[feature_columns].copy(),
    }


def create_train_valid_split(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    train_event_id: pd.Series,
    train_weight: pd.Series,
    valid_size: float = DEFAULT_VALID_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Creates the train/validation split before fitting any transformation."""
    train_idx, valid_idx = train_test_split(
        X_full.index.to_numpy(),
        test_size=valid_size,
        random_state=random_state,
        stratify=y_full,
    )

    train_idx = pd.Index(train_idx)
    valid_idx = pd.Index(valid_idx)

    return {
        "X_train_raw": X_full.loc[train_idx].copy(),
        "X_valid_raw": X_full.loc[valid_idx].copy(),
        "y_train": y_full.loc[train_idx].reset_index(drop=True),
        "y_valid": y_full.loc[valid_idx].reset_index(drop=True),
        "train_event_id": train_event_id.loc[train_idx].reset_index(drop=True),
        "valid_event_id": train_event_id.loc[valid_idx].reset_index(drop=True),
        "train_weight": train_weight.loc[train_idx].reset_index(drop=True),
        "valid_weight": train_weight.loc[valid_idx].reset_index(drop=True),
        "train_idx": train_idx,
        "valid_idx": valid_idx,
    }


def build_raw_missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarises NaN counts and percentages for a raw feature frame."""
    summary = pd.DataFrame(
        {
            "feature": frame.columns,
            "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
        }
    )
    summary["missing_pct"] = (summary["missing_count"] / len(frame) * 100).round(2)
    return summary.sort_values(["missing_pct", "feature"], ascending=[False, True]).reset_index(drop=True)


def build_split_summary(
    y_train: pd.Series,
    y_valid: pd.Series,
    train_weight: pd.Series,
    valid_weight: pd.Series,
    test_rows: int,
) -> pd.DataFrame:
    """Builds a compact split summary table."""
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": int(len(y_train)),
                "class_b": int((y_train == "b").sum()),
                "class_s": int((y_train == "s").sum()),
                "weight_sum": float(train_weight.sum()),
            },
            {
                "split": "valid",
                "rows": int(len(y_valid)),
                "class_b": int((y_valid == "b").sum()),
                "class_s": int((y_valid == "s").sum()),
                "weight_sum": float(valid_weight.sum()),
            },
            {
                "split": "test_external",
                "rows": int(test_rows),
                "class_b": np.nan,
                "class_s": np.nan,
                "weight_sum": np.nan,
            },
        ]
    )


def wrap_delta_phi(delta_phi: pd.Series | np.ndarray) -> pd.Series:
    """Wraps angular differences to the [-pi, pi] interval."""
    values = np.asarray(delta_phi, dtype=float)
    wrapped = np.arctan2(np.sin(values), np.cos(values))
    return pd.Series(wrapped, index=getattr(delta_phi, "index", None))


def safe_divide(numerator: pd.Series, denominator: pd.Series | np.ndarray, epsilon: float = EPSILON) -> pd.Series:
    """Performs a numerically stable division."""
    return numerator / (denominator + epsilon)


def add_physics_by_jet_features(
    data: pd.DataFrame,
    jet_num_column: str = JET_NUM_COLUMN,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Adds physically motivated features, including jet-regime specific views."""
    required_columns = [
        "DER_mass_MMC",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_lep_eta_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        jet_num_column,
    ]
    missing_required = [column for column in required_columns if column not in data.columns]
    if missing_required:
        raise ValueError(f"Faltan columnas requeridas para feature engineering: {missing_required}")

    X = data.copy()

    jet_num = X[jet_num_column].fillna(-1)
    is_jet_0 = (jet_num == 0).astype(np.float32)
    is_jet_1 = (jet_num == 1).astype(np.float32)
    is_jet_2plus = (jet_num >= 2).astype(np.float32)
    has_jet = (jet_num >= 1).astype(np.float32)
    has_two_jets = (jet_num >= 2).astype(np.float32)

    leading_pt = X["PRI_jet_leading_pt"].fillna(0.0)
    subleading_pt = X["PRI_jet_subleading_pt"].fillna(0.0)
    leading_eta = X["PRI_jet_leading_eta"].fillna(0.0)
    leading_phi = X["PRI_jet_leading_phi"].fillna(0.0)
    dijet_mass = X["DER_mass_jet_jet"].fillna(0.0)
    dijet_deltaeta = X["DER_deltaeta_jet_jet"].fillna(0.0)
    dijet_prodeta = X["DER_prodeta_jet_jet"].fillna(0.0)
    lep_eta_centrality = X["DER_lep_eta_centrality"].fillna(0.0)

    visible_energy = X["PRI_tau_pt"] + X["PRI_lep_pt"]
    jets_pt_sum = leading_pt + subleading_pt
    ht = visible_energy + X["PRI_met"] + jets_pt_sum

    delta_eta_tau_lep = X["PRI_tau_eta"] - X["PRI_lep_eta"]
    delta_phi_tau_lep = wrap_delta_phi(X["PRI_tau_phi"] - X["PRI_lep_phi"])
    delta_eta_tau_jet1 = X["PRI_tau_eta"] - leading_eta
    delta_phi_tau_jet1 = wrap_delta_phi(X["PRI_tau_phi"] - leading_phi)
    delta_phi_met_leading_jet = wrap_delta_phi(X["PRI_met_phi"] - leading_phi)

    X["ratio_tau_lep_pt"] = safe_divide(X["PRI_tau_pt"], X["PRI_lep_pt"], epsilon=epsilon)
    X["ratio_tau_met"] = safe_divide(X["PRI_tau_pt"], X["PRI_met"], epsilon=epsilon)
    X["ratio_lep_met"] = safe_divide(X["PRI_lep_pt"], X["PRI_met"], epsilon=epsilon)
    X["ratio_mass_visible"] = safe_divide(X["DER_mass_MMC"], visible_energy, epsilon=epsilon)
    X["visible_energy"] = visible_energy
    X["HT"] = ht
    X["met_fraction"] = safe_divide(X["PRI_met"], ht, epsilon=epsilon)
    X["jets_pt_sum"] = jets_pt_sum
    X["jet_lepton_balance"] = safe_divide(jets_pt_sum, X["PRI_lep_pt"], epsilon=epsilon)
    X["abs_eta_diff_tau_lep"] = delta_eta_tau_lep.abs()
    X["delta_phi_tau_lep"] = delta_phi_tau_lep
    X["abs_phi_diff_tau_lep"] = delta_phi_tau_lep.abs()
    X["deltaR_tau_lep"] = np.sqrt(X["abs_eta_diff_tau_lep"] ** 2 + X["abs_phi_diff_tau_lep"] ** 2)
    X["tau_lep_interaction"] = X["PRI_tau_pt"] * X["PRI_lep_pt"]
    X["tau_met_interaction"] = X["PRI_tau_pt"] * X["PRI_met"]

    X["has_jet"] = has_jet
    X["has_two_jets"] = has_two_jets
    X["jet_regime_0"] = is_jet_0
    X["jet_regime_1"] = is_jet_1
    X["jet_regime_2plus"] = is_jet_2plus

    X["leading_pt_fraction"] = safe_divide(leading_pt, ht, epsilon=epsilon)
    X["subleading_pt_fraction"] = safe_divide(subleading_pt, ht, epsilon=epsilon)
    X["abs_delta_eta_tau_jet1"] = delta_eta_tau_jet1.abs() * has_jet
    X["abs_delta_phi_tau_jet1"] = delta_phi_tau_jet1.abs() * has_jet
    X["deltaR_tau_jet1"] = np.sqrt(
        (delta_eta_tau_jet1.abs() ** 2) + (delta_phi_tau_jet1.abs() ** 2)
    ) * has_jet
    X["abs_delta_phi_met_jet1"] = delta_phi_met_leading_jet.abs() * has_jet
    X["jet1_met_balance"] = safe_divide(leading_pt, X["PRI_met"], epsilon=epsilon) * has_jet

    X["dijet_mass_over_ht"] = safe_divide(dijet_mass, ht, epsilon=epsilon) * has_two_jets
    X["abs_dijet_deltaeta"] = np.abs(dijet_deltaeta) * has_two_jets
    X["abs_dijet_prodeta"] = np.abs(dijet_prodeta) * has_two_jets
    X["lep_eta_centrality_if_2plus"] = lep_eta_centrality * has_two_jets
    X["dijet_pt_balance"] = safe_divide(leading_pt, subleading_pt, epsilon=epsilon) * has_two_jets

    X["HT_jet0"] = ht * is_jet_0
    X["HT_jet1"] = ht * is_jet_1
    X["HT_jet2plus"] = ht * is_jet_2plus
    X["met_fraction_jet0"] = X["met_fraction"] * is_jet_0
    X["met_fraction_jet1"] = X["met_fraction"] * is_jet_1
    X["met_fraction_jet2plus"] = X["met_fraction"] * is_jet_2plus
    X["deltaR_tau_lep_jet0"] = X["deltaR_tau_lep"] * is_jet_0
    X["deltaR_tau_lep_jet1"] = X["deltaR_tau_lep"] * is_jet_1
    X["deltaR_tau_lep_jet2plus"] = X["deltaR_tau_lep"] * is_jet_2plus

    return X


class JetAwareMedianImputer(BaseEstimator, TransformerMixin):
    """Imputes numeric columns while preserving structural zeros tied to the jet count."""

    def __init__(
        self,
        group_column: str = JET_NUM_COLUMN,
        structural_zero_min_jets: dict[str, int] | None = None,
    ) -> None:
        self.group_column = group_column
        self.structural_zero_min_jets = structural_zero_min_jets

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "JetAwareMedianImputer":
        frame = self._validate_frame(X)
        prepared = self._apply_structural_zero_fill(frame)
        self.feature_columns_ = list(prepared.columns)
        self.numeric_columns_ = [
            column for column in self.feature_columns_
            if column != self.group_column and pd.api.types.is_numeric_dtype(prepared[column])
        ]
        self.group_keys_ = prepared[self.group_column].fillna("__missing__").astype(str)
        self.group_medians_ = prepared.assign(__group_key__=self.group_keys_).groupby("__group_key__")[
            self.numeric_columns_
        ].median()
        self.global_medians_ = prepared[self.numeric_columns_].median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_frame(X)
        prepared = self._apply_structural_zero_fill(frame)
        group_keys = prepared[self.group_column].fillna("__missing__").astype(str)
        group_fill = self.group_medians_.reindex(group_keys.tolist()).set_index(prepared.index)
        fill_values = group_fill.fillna(self.global_medians_)

        transformed = prepared.copy()
        transformed[self.numeric_columns_] = transformed[self.numeric_columns_].fillna(fill_values)
        return transformed

    def _validate_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("JetAwareMedianImputer espera un pandas.DataFrame.")
        if self.group_column not in X.columns:
            raise ValueError(f"Falta la columna requerida para agrupar: {self.group_column}")
        return X.copy()

    def _apply_structural_zero_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        jet_count = transformed[self.group_column].fillna(-1)
        structural_rules = self.structural_zero_min_jets or {}
        for column, min_jets in structural_rules.items():
            if column not in transformed.columns:
                continue
            impossible_mask = jet_count < min_jets
            transformed.loc[impossible_mask & transformed[column].isna(), column] = 0.0
        return transformed


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Adds `is_missing_*` columns for selected source columns."""

    def __init__(
        self,
        selected_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> None:
        self.selected_columns = selected_columns
        self.exclude_columns = exclude_columns or []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingIndicatorAdder":
        frame = self._validate_frame(X)
        excluded = set(self.exclude_columns)
        if self.selected_columns is None:
            self.source_columns_ = [
                column for column in frame.columns
                if column not in excluded and frame[column].isna().any()
            ]
        else:
            missing = [column for column in self.selected_columns if column not in frame.columns]
            if missing:
                raise ValueError(f"Las columnas para missing indicator no existen: {missing}")
            self.source_columns_ = [
                column for column in self.selected_columns
                if column not in excluded
            ]
        self.output_columns_ = list(frame.columns) + [
            f"is_missing_{column}" for column in self.source_columns_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_frame(X)
        transformed = frame.copy()
        for column in self.source_columns_:
            transformed[f"is_missing_{column}"] = transformed[column].isna().astype(np.float32)
        return transformed

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.output_columns_, dtype=object)

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingIndicatorAdder espera un pandas.DataFrame.")
        return X


class TreeSafePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor that keeps NaN values for tree-based models."""

    def __init__(self, jet_num_column: str = JET_NUM_COLUMN) -> None:
        self.jet_num_column = jet_num_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TreeSafePreprocessor":
        frame = self._validate_frame(X)
        self.feature_columns_ = list(frame.columns)
        self.numeric_columns_ = [
            column for column in self.feature_columns_
            if column != self.jet_num_column
        ]
        self.missing_indicator_adder_ = MissingIndicatorAdder(
            exclude_columns=[self.jet_num_column],
        )
        expanded = self.missing_indicator_adder_.fit_transform(frame)
        self.missing_indicator_columns_ = [
            column for column in expanded.columns
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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_frame(X)
        expanded = self.missing_indicator_adder_.transform(frame)
        numeric_frame = expanded[self.numeric_columns_].copy()
        missing_frame = expanded[self.missing_indicator_columns_].copy()
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

    def _validate_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TreeSafePreprocessor espera un pandas.DataFrame.")
        if self.jet_num_column not in X.columns:
            raise ValueError(f"Falta la columna requerida: {self.jet_num_column}")
        if hasattr(self, "feature_columns_"):
            missing = [column for column in self.feature_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"Faltan columnas esperadas para transformar: {missing}")
            return X[self.feature_columns_].copy()
        return X.copy()


class DenseModelPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for dense models, optionally adding physics-by-jet features."""

    def __init__(
        self,
        feature_mode: str = "base",
        jet_num_column: str = JET_NUM_COLUMN,
        scaler_quantile_range: tuple[float, float] = (10.0, 90.0),
    ) -> None:
        self.feature_mode = feature_mode
        self.jet_num_column = jet_num_column
        self.scaler_quantile_range = scaler_quantile_range

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DenseModelPreprocessor":
        original = self._validate_original_frame(X)
        self.original_feature_columns_ = list(original.columns)
        expanded = self._expand_features(original)
        self.expanded_feature_columns_ = list(expanded.columns)

        self.missing_indicator_adder_ = MissingIndicatorAdder(
            exclude_columns=[self.jet_num_column],
        )
        expanded_with_flags = self.missing_indicator_adder_.fit_transform(expanded)
        self.missing_indicator_columns_ = [
            column for column in expanded_with_flags.columns
            if column.startswith("is_missing_")
        ]

        self.imputer_ = JetAwareMedianImputer(
            group_column=self.jet_num_column,
            structural_zero_min_jets=self._build_structural_zero_rules(expanded),
        )
        imputed = self.imputer_.fit_transform(expanded)

        self.binary_passthrough_columns_ = [
            column for column in imputed.columns
            if column.startswith("has_") or column.startswith("jet_regime_")
        ]
        self.numeric_scaled_columns_ = [
            column for column in imputed.columns
            if column != self.jet_num_column and column not in self.binary_passthrough_columns_
        ]

        self.scaler_ = RobustScaler(quantile_range=self.scaler_quantile_range)
        self.scaler_.fit(imputed[self.numeric_scaled_columns_])

        self.jet_encoder_ = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )
        self.jet_encoder_.fit(imputed[[self.jet_num_column]].fillna("__missing__").astype(str))
        self.jet_encoded_columns_ = self.jet_encoder_.get_feature_names_out([self.jet_num_column]).tolist()

        self.processed_feature_names_ = (
            self.numeric_scaled_columns_
            + self.binary_passthrough_columns_
            + self.missing_indicator_columns_
            + self.jet_encoded_columns_
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        original = self._validate_original_frame(X)
        expanded = self._expand_features(original)
        expanded_with_flags = self.missing_indicator_adder_.transform(expanded)
        flag_frame = expanded_with_flags[self.missing_indicator_columns_].copy()

        imputed = self.imputer_.transform(expanded)
        scaled_array = self.scaler_.transform(imputed[self.numeric_scaled_columns_])
        scaled_frame = pd.DataFrame(
            scaled_array,
            columns=self.numeric_scaled_columns_,
            index=imputed.index,
        )

        binary_frame = imputed[self.binary_passthrough_columns_].copy()
        jet_encoded = self.jet_encoder_.transform(imputed[[self.jet_num_column]].fillna("__missing__").astype(str))
        jet_frame = pd.DataFrame(
            jet_encoded,
            columns=self.jet_encoded_columns_,
            index=imputed.index,
        )

        transformed = pd.concat([scaled_frame, binary_frame, flag_frame, jet_frame], axis=1)
        return transformed.astype(np.float32)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.processed_feature_names_, dtype=object)

    def _expand_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_mode == "base":
            return X.copy()
        if self.feature_mode == "physics_by_jet":
            return add_physics_by_jet_features(X.copy(), jet_num_column=self.jet_num_column)
        raise ValueError(f"feature_mode no soportado: {self.feature_mode}")

    def _build_structural_zero_rules(self, expanded: pd.DataFrame) -> dict[str, int]:
        rules: dict[str, int] = {}
        for column in RAW_ONE_JET_COLUMNS:
            if column in expanded.columns:
                rules[column] = 1
        for column in RAW_TWO_JET_COLUMNS:
            if column in expanded.columns:
                rules[column] = 2
        return rules

    def _validate_original_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DenseModelPreprocessor espera un pandas.DataFrame.")
        if self.jet_num_column not in X.columns:
            raise ValueError(f"Falta la columna requerida: {self.jet_num_column}")
        if hasattr(self, "original_feature_columns_"):
            missing = [column for column in self.original_feature_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"Faltan columnas esperadas para transformar: {missing}")
            return X[self.original_feature_columns_].copy()
        return X.copy()


def build_preprocessing_pipeline(variant_key: str) -> Pipeline:
    """Builds the sklearn pipeline for the requested preprocessing variant."""
    if variant_key == "tree_safe":
        preprocessor: BaseEstimator = TreeSafePreprocessor()
    elif variant_key == "nn_dense":
        preprocessor = DenseModelPreprocessor(feature_mode="base")
    elif variant_key == "physics_by_jet":
        preprocessor = DenseModelPreprocessor(feature_mode="physics_by_jet")
    else:
        raise ValueError(f"Variante desconocida: {variant_key}")

    return Pipeline([("preprocess", preprocessor)])


def _to_processed_frame(
    pipeline: Pipeline,
    data: pd.DataFrame,
) -> pd.DataFrame:
    processed = pipeline.transform(data)
    if isinstance(processed, pd.DataFrame):
        return processed
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    return pd.DataFrame(processed, columns=feature_names, index=data.index)


def run_preprocessing_variant(
    variant_key: str,
    project_root: Path | None = None,
    valid_size: float = DEFAULT_VALID_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Runs one preprocessing variant end-to-end and returns all artifacts."""
    if variant_key not in VARIANT_DEFINITIONS:
        raise ValueError(f"Variante desconocida: {variant_key}")

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

    pipeline = build_preprocessing_pipeline(variant_key)
    pipeline.fit(split["X_train_raw"])

    X_train_processed = _to_processed_frame(pipeline, split["X_train_raw"]).reset_index(drop=True)
    X_valid_processed = _to_processed_frame(pipeline, split["X_valid_raw"]).reset_index(drop=True)
    X_test_processed = _to_processed_frame(pipeline, base["X_test_raw"]).reset_index(drop=True)

    processed_feature_names = X_train_processed.columns.tolist()
    preprocess_step = pipeline.named_steps["preprocess"]
    missing_indicator_source_columns = getattr(
        getattr(preprocess_step, "missing_indicator_adder_", None),
        "source_columns_",
        [],
    )
    split_summary_df = build_split_summary(
        y_train=split["y_train"],
        y_valid=split["y_valid"],
        train_weight=split["train_weight"],
        valid_weight=split["valid_weight"],
        test_rows=len(base["X_test_raw"]),
    )

    train_missing_summary = build_raw_missing_summary(split["X_train_raw"])
    valid_missing_summary = build_raw_missing_summary(split["X_valid_raw"])
    test_missing_summary = build_raw_missing_summary(base["X_test_raw"])

    metadata = {
        "variant_key": variant_key,
        "variant_title": VARIANT_DEFINITIONS[variant_key].title,
        "description": VARIANT_DEFINITIONS[variant_key].description,
        "valid_size": float(valid_size),
        "random_state": int(random_state),
        "raw_feature_count": int(len(base["feature_columns"])),
        "processed_feature_count": int(len(processed_feature_names)),
        "processed_train_nan_count": int(X_train_processed.isna().sum().sum()),
        "processed_valid_nan_count": int(X_valid_processed.isna().sum().sum()),
        "processed_test_nan_count": int(X_test_processed.isna().sum().sum()),
        "expects_missing_after_processing": VARIANT_DEFINITIONS[variant_key].expects_missing_after_processing,
    }

    return {
        "variant": VARIANT_DEFINITIONS[variant_key],
        "metadata": metadata,
        "feature_columns": base["feature_columns"],
        "processed_feature_names": processed_feature_names,
        "missing_indicator_source_columns": missing_indicator_source_columns,
        "preprocessing_pipeline": pipeline,
        "split_summary_df": split_summary_df,
        "train_missing_summary_df": train_missing_summary,
        "valid_missing_summary_df": valid_missing_summary,
        "test_missing_summary_df": test_missing_summary,
        "X_train_raw": split["X_train_raw"],
        "X_valid_raw": split["X_valid_raw"],
        "X_test_raw": base["X_test_raw"],
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


def save_preprocessing_artifacts(
    artifacts: dict[str, object],
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    """Saves processed CSVs, metadata and the fitted pipeline to disk."""
    variant: VariantDefinition = artifacts["variant"]
    if output_dir is None:
        output_dir = get_higgs_data_dir(project_root) / "processed" / variant.output_dir_name
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
        "processed_feature_names.csv": pd.DataFrame({"feature": artifacts["processed_feature_names"]}),
        "missing_indicator_source_columns.csv": pd.DataFrame(
            {"feature": artifacts["missing_indicator_source_columns"]}
        ),
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
