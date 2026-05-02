from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

SPECIAL_COLUMNS = ["EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"]
MISSING_MARKER = -999.0
JET_NUM_COLUMN = "PRI_jet_num"
JET_NUM_VALUES = [0, 1, 2, 3]
SEED = 42
VALID_SIZE = 0.1
EPS = 1e-6

ALLOWED_FEATURE_COLUMNS = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
    "PRI_tau_pt",
    "PRI_tau_eta",
    "PRI_tau_phi",
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_met",
    "PRI_met_phi",
    "PRI_met_sumet",
    "PRI_jet_num",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_jet_all_pt",
]

BASE_LOG_COLUMNS = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_deltar_tau_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_tau",
    "PRI_tau_pt",
    "PRI_lep_pt",
    "PRI_met",
    "PRI_met_sumet",
    "PRI_jet_leading_pt",
    "PRI_jet_subleading_pt",
    "PRI_jet_all_pt",
]

FIXED_JET_ONE_HOT_COLUMNS = [
    "PRI_jet_num_0",
    "PRI_jet_num_1",
    "PRI_jet_num_2",
    "PRI_jet_num_3",
]

P03_G0_DROP_COLUMNS = [
    "PRI_jet_num",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_lep_eta_centrality",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
]

P03_G1_DROP_COLUMNS = [
    "PRI_jet_num",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_lep_eta_centrality",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
]

P04_ENGINEERED_LOG_COLUMNS = ["FE_tau_lep_pt_sum"]


@dataclass(slots=True)
class CommonData:
    X_train_raw: pd.DataFrame
    X_valid_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    train_weight: pd.Series
    valid_weight: pd.Series
    train_event_id: pd.Series
    valid_event_id: pd.Series
    test_event_id: pd.Series
    metadata: dict[str, object]


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "roadmap.md").exists() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


def data_dir(project_root: Path | None = None) -> Path:
    return find_project_root(project_root) / "data" / "higgs-boson"


def output_root(project_root: Path | None = None) -> Path:
    return find_project_root(project_root) / "data" / "preprocesor"


def load_raw_higgs(project_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = data_dir(project_root)
    train_path = root / "training.csv"
    test_path = root / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"No se encontro {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"No se encontro {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _validate_label(label: pd.Series) -> None:
    invalid = sorted(set(label.dropna().unique()) - {"b", "s"})
    if invalid:
        raise ValueError(f"Label contiene valores no permitidos: {invalid}")


def _validate_feature_frame(X: pd.DataFrame, original_rows: int) -> None:
    leaked = [column for column in SPECIAL_COLUMNS if column in X.columns]
    if leaked:
        raise ValueError(f"Columnas especiales dentro de X: {leaked}")
    if len(X) != original_rows:
        raise ValueError("Se eliminaron filas durante C00.")
    if JET_NUM_COLUMN not in X.columns:
        raise ValueError("PRI_jet_num no existe en X.")
    jet_values = sorted(set(X[JET_NUM_COLUMN].dropna().astype(int).unique()))
    invalid = [value for value in jet_values if value not in JET_NUM_VALUES]
    if invalid:
        raise ValueError(f"PRI_jet_num contiene valores fuera de 0,1,2,3: {invalid}")
    non_numeric = [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
    if non_numeric:
        raise ValueError(f"Features no numericas detectadas: {non_numeric}")


def build_common_data(
    project_root: Path | None = None,
    valid_size: float = VALID_SIZE,
    random_state: int = SEED,
) -> CommonData:
    train_df, test_df = load_raw_higgs(project_root)
    train_special = [column for column in SPECIAL_COLUMNS if column in train_df.columns]
    test_special = [column for column in SPECIAL_COLUMNS if column in test_df.columns]
    feature_cols = [column for column in train_df.columns if column not in SPECIAL_COLUMNS]
    test_feature_cols = [column for column in test_df.columns if column not in SPECIAL_COLUMNS]

    if feature_cols != ALLOWED_FEATURE_COLUMNS:
        raise ValueError("Las features de training.csv no coinciden con la especificacion oficial esperada.")
    if test_feature_cols != feature_cols:
        raise ValueError("Las features de training.csv y test.csv no coinciden.")

    _validate_label(train_df["Label"])
    X_full = train_df[feature_cols].replace(MISSING_MARKER, np.nan)
    X_test = test_df[feature_cols].replace(MISSING_MARKER, np.nan)
    _validate_feature_frame(X_full, len(train_df))
    _validate_feature_frame(X_test, len(test_df))

    y_full = train_df["Label"].map({"b": 0, "s": 1}).astype(np.int8)
    weight_full = train_df["Weight"].astype(np.float64)
    event_id_full = train_df["EventId"].astype(np.int64)
    test_event_id = test_df["EventId"].astype(np.int64)

    train_idx, valid_idx = train_test_split(
        X_full.index.to_numpy(),
        test_size=valid_size,
        random_state=random_state,
        stratify=y_full,
    )

    X_train_raw = X_full.loc[train_idx].reset_index(drop=True)
    X_valid_raw = X_full.loc[valid_idx].reset_index(drop=True)
    y_train = y_full.loc[train_idx].reset_index(drop=True)
    y_valid = y_full.loc[valid_idx].reset_index(drop=True)
    train_weight = weight_full.loc[train_idx].reset_index(drop=True)
    valid_weight = weight_full.loc[valid_idx].reset_index(drop=True)
    train_event_id = event_id_full.loc[train_idx].reset_index(drop=True)
    valid_event_id = event_id_full.loc[valid_idx].reset_index(drop=True)

    missing_counts = X_full.isna().sum().astype(int)
    metadata = {
        "feature_cols": feature_cols,
        "special_cols_detected_train": train_special,
        "special_cols_detected_test": test_special,
        "n_rows_train_full": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "missing_counts_by_column": missing_counts.to_dict(),
        "pri_jet_num_distribution_train_full": X_full[JET_NUM_COLUMN].value_counts().sort_index().astype(int).to_dict(),
        "valid_size": float(valid_size),
        "random_state": int(random_state),
    }

    return CommonData(
        X_train_raw=X_train_raw,
        X_valid_raw=X_valid_raw,
        X_test_raw=X_test.reset_index(drop=True),
        y_train=y_train,
        y_valid=y_valid,
        train_weight=train_weight,
        valid_weight=valid_weight,
        train_event_id=train_event_id,
        valid_event_id=valid_event_id,
        test_event_id=test_event_id.reset_index(drop=True),
        metadata=metadata,
    )


def abs_delta_phi(phi_a: pd.Series, phi_b: pd.Series) -> pd.Series:
    return ((phi_a - phi_b + np.pi) % (2 * np.pi) - np.pi).abs()


def add_p04_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    result["FE_abs_dphi_tau_lep"] = abs_delta_phi(result["PRI_tau_phi"], result["PRI_lep_phi"])
    result["FE_abs_dphi_tau_met"] = abs_delta_phi(result["PRI_tau_phi"], result["PRI_met_phi"])
    result["FE_abs_dphi_lep_met"] = abs_delta_phi(result["PRI_lep_phi"], result["PRI_met_phi"])
    result["FE_abs_deta_tau_lep"] = (result["PRI_tau_eta"] - result["PRI_lep_eta"]).abs()
    result["FE_tau_lep_pt_sum"] = result["PRI_tau_pt"] + result["PRI_lep_pt"]
    result["FE_tau_pt_fraction"] = result["PRI_tau_pt"] / (result["PRI_tau_pt"] + result["PRI_lep_pt"] + EPS)
    result["FE_met_over_tau_lep_pt"] = result["PRI_met"] / (result["PRI_tau_pt"] + result["PRI_lep_pt"] + EPS)
    result["FE_met_over_sumet"] = result["PRI_met"] / (result["PRI_met_sumet"] + EPS)
    result["FE_pt_balance"] = result["DER_pt_tot"] / (result["DER_sum_pt"] + EPS)
    result["FE_mmc_over_visible_mass"] = result["DER_mass_MMC"] / (result["DER_mass_vis"] + EPS)
    result["FE_transverse_mass_over_visible_mass"] = result["DER_mass_transverse_met_lep"] / (
        result["DER_mass_vis"] + EPS
    )
    result["FE_jet_all_pt_fraction"] = result["PRI_jet_all_pt"] / (result["DER_sum_pt"] + EPS)
    result["FE_leading_jet_pt_fraction"] = result["PRI_jet_leading_pt"] / (result["PRI_jet_all_pt"] + EPS)
    result["FE_subleading_jet_pt_fraction"] = result["PRI_jet_subleading_pt"] / (result["PRI_jet_all_pt"] + EPS)
    result["FE_jet_pt_balance"] = (result["PRI_jet_leading_pt"] - result["PRI_jet_subleading_pt"]) / (
        result["PRI_jet_leading_pt"] + result["PRI_jet_subleading_pt"] + EPS
    )
    result["FE_abs_dphi_leading_subleading_jet"] = abs_delta_phi(
        result["PRI_jet_leading_phi"],
        result["PRI_jet_subleading_phi"],
    )
    result["FE_mass_jet_over_sum_pt"] = result["DER_mass_jet_jet"] / (result["DER_sum_pt"] + EPS)
    return result


class FoldPreprocessor:
    def __init__(
        self,
        *,
        pipeline_id: str,
        one_hot_jet_num: bool,
        scaler_kind: str,
        add_features: bool = False,
        scale_jet_num_numeric: bool = True,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.one_hot_jet_num = one_hot_jet_num
        self.scaler_kind = scaler_kind
        self.add_features = add_features
        self.scale_jet_num_numeric = scale_jet_num_numeric

    def fit(self, X_train_raw: pd.DataFrame) -> "FoldPreprocessor":
        X = self._prepare_features(X_train_raw, fitting=True)
        self.missing_indicator_cols_ = [
            column for column in X.columns
            if column != JET_NUM_COLUMN and X[column].isna().any()
        ]
        if not self.one_hot_jet_num and X[JET_NUM_COLUMN].isna().any():
            self.missing_indicator_cols_.append(JET_NUM_COLUMN)

        continuous = X.copy()
        if self.one_hot_jet_num:
            continuous = continuous.drop(columns=[JET_NUM_COLUMN])
        self.continuous_cols_ = continuous.columns.tolist()
        self.log_cols_ = [column for column in self._log_columns() if column in self.continuous_cols_]

        logged = self._apply_log1p(continuous, self.log_cols_)
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(logged[self.continuous_cols_])
        self.scaler_ = self._build_scaler()
        self.scaler_.fit(imputed)
        self.one_hot_cols_ = FIXED_JET_ONE_HOT_COLUMNS if self.one_hot_jet_num else []
        self.final_feature_order_ = self.continuous_cols_ + [
            f"is_missing_{column}" for column in self.missing_indicator_cols_
        ] + self.one_hot_cols_
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_features(X_raw, fitting=False)
        indicator_frame = pd.DataFrame(index=X.index)
        for column in self.missing_indicator_cols_:
            indicator_frame[f"is_missing_{column}"] = X[column].isna().astype(np.float32)

        continuous = X.copy()
        if self.one_hot_jet_num:
            jet_one_hot = self._fixed_jet_one_hot(X[JET_NUM_COLUMN])
            continuous = continuous.drop(columns=[JET_NUM_COLUMN])
        else:
            jet_one_hot = pd.DataFrame(index=X.index)

        continuous = continuous[self.continuous_cols_]
        logged = self._apply_log1p(continuous, self.log_cols_)
        imputed = self.imputer_.transform(logged[self.continuous_cols_])
        scaled = self.scaler_.transform(imputed)
        scaled_frame = pd.DataFrame(scaled, columns=self.continuous_cols_, index=X.index)
        result = pd.concat([scaled_frame, indicator_frame, jet_one_hot], axis=1)
        return result[self.final_feature_order_].astype(np.float32)

    def metadata(self) -> dict[str, object]:
        scaler_metadata = {
            "scaler_kind": self.scaler_kind,
            "scaler_center_or_mean": getattr(self.scaler_, "mean_", getattr(self.scaler_, "center_", np.array([]))).tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
        }
        return {
            "pipeline_id": self.pipeline_id,
            "missing_indicator_cols": self.missing_indicator_cols_,
            "log_cols": self.log_cols_,
            "one_hot_cols": self.one_hot_cols_,
            "median_values": self.imputer_.statistics_.tolist(),
            "continuous_cols": self.continuous_cols_,
            "final_feature_order": self.final_feature_order_,
            **scaler_metadata,
        }

    def _prepare_features(self, X_raw: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        X = add_p04_features(X_raw) if self.add_features else X_raw.copy()
        if JET_NUM_COLUMN not in X.columns:
            raise ValueError("PRI_jet_num no existe antes del preprocesamiento.")
        return X

    def _log_columns(self) -> list[str]:
        return BASE_LOG_COLUMNS + (P04_ENGINEERED_LOG_COLUMNS if self.add_features else [])

    def _apply_log1p(self, X: pd.DataFrame, log_cols: list[str]) -> pd.DataFrame:
        result = X.copy()
        for column in log_cols:
            if column not in result.columns:
                continue
            values = result[column].dropna()
            if (values < 0).any():
                min_value = float(values.min())
                raise ValueError(f"{self.pipeline_id}: {column} tiene valores < 0 y no admite log1p. Min={min_value}")
            result[column] = np.log1p(result[column])
        return result

    def _build_scaler(self):
        if self.scaler_kind == "standard":
            return StandardScaler()
        if self.scaler_kind == "robust":
            return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        raise ValueError(f"Scaler no soportado: {self.scaler_kind}")

    def _fixed_jet_one_hot(self, jet_num: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"PRI_jet_num_{value}": (jet_num.astype(int) == value).astype(np.float32)
                for value in JET_NUM_VALUES
            },
            index=jet_num.index,
        )


class GroupPreprocessor:
    def __init__(self, *, group_id: str, drop_columns: list[str], add_jet_num_is_3: bool) -> None:
        self.group_id = group_id
        self.drop_columns = drop_columns
        self.add_jet_num_is_3 = add_jet_num_is_3

    def fit(self, X_train_raw: pd.DataFrame) -> "GroupPreprocessor":
        X = self._prepare_group(X_train_raw)
        self.missing_indicator_cols_ = [column for column in X.columns if X[column].isna().any()]
        self.passthrough_cols_ = ["PRI_jet_num_is_3"] if self.add_jet_num_is_3 else []
        self.continuous_cols_ = [column for column in X.columns if column not in self.passthrough_cols_]
        self.log_cols_ = [column for column in BASE_LOG_COLUMNS if column in self.continuous_cols_]
        logged = self._apply_log1p(X, self.log_cols_)
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(logged[self.continuous_cols_])
        self.scaler_ = StandardScaler()
        self.scaler_.fit(imputed)
        self.flag_cols_ = [f"is_missing_{column}" for column in self.missing_indicator_cols_]
        self.final_feature_order_ = self.continuous_cols_ + self.passthrough_cols_ + self.flag_cols_
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_group(X_raw)
        flags = pd.DataFrame(
            {
                f"is_missing_{column}": X[column].isna().astype(np.float32)
                for column in self.missing_indicator_cols_
            },
            index=X.index,
        )
        logged = self._apply_log1p(X[self.continuous_cols_], self.log_cols_)
        imputed = self.imputer_.transform(logged[self.continuous_cols_])
        scaled = self.scaler_.transform(imputed)
        scaled_frame = pd.DataFrame(scaled, columns=self.continuous_cols_, index=X.index)
        passthrough = X[self.passthrough_cols_].astype(np.float32) if self.passthrough_cols_ else pd.DataFrame(index=X.index)
        result = pd.concat([scaled_frame, passthrough, flags], axis=1)
        return result[self.final_feature_order_].astype(np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "drop_columns": self.drop_columns,
            "add_jet_num_is_3": self.add_jet_num_is_3,
            "missing_indicator_cols": self.missing_indicator_cols_,
            "log_cols": self.log_cols_,
            "passthrough_cols": self.passthrough_cols_,
            "median_values": self.imputer_.statistics_.tolist(),
            "standard_scaler_mean": self.scaler_.mean_.tolist(),
            "standard_scaler_scale": self.scaler_.scale_.tolist(),
            "final_feature_order": self.final_feature_order_,
        }

    def _prepare_group(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = X_raw.copy()
        if self.add_jet_num_is_3:
            X["PRI_jet_num_is_3"] = (X[JET_NUM_COLUMN].astype(int) == 3).astype(np.float32)
        existing_drop = [column for column in self.drop_columns if column in X.columns]
        return X.drop(columns=existing_drop)

    def _apply_log1p(self, X: pd.DataFrame, log_cols: list[str]) -> pd.DataFrame:
        result = X.copy()
        for column in log_cols:
            values = result[column].dropna()
            if (values < 0).any():
                min_value = float(values.min())
                raise ValueError(f"{self.group_id}: {column} tiene valores < 0 y no admite log1p. Min={min_value}")
            result[column] = np.log1p(result[column])
        return result


def build_pipelines() -> dict[str, FoldPreprocessor]:
    return {
        "P00": FoldPreprocessor(
            pipeline_id="P00",
            one_hot_jet_num=False,
            scaler_kind="standard",
        ),
        "P01": FoldPreprocessor(
            pipeline_id="P01",
            one_hot_jet_num=True,
            scaler_kind="standard",
        ),
        "P02": FoldPreprocessor(
            pipeline_id="P02",
            one_hot_jet_num=True,
            scaler_kind="robust",
        ),
        "P04": FoldPreprocessor(
            pipeline_id="P04",
            one_hot_jet_num=True,
            scaler_kind="robust",
            add_features=True,
        ),
    }


def transform_global_pipeline(pipeline_id: str, common: CommonData) -> dict[str, object]:
    pipelines = build_pipelines()
    if pipeline_id not in pipelines:
        raise ValueError(f"Pipeline global desconocido: {pipeline_id}")
    processor = pipelines[pipeline_id]
    processor.fit(common.X_train_raw)
    X_train = processor.transform(common.X_train_raw)
    X_valid = processor.transform(common.X_valid_raw)
    X_test = processor.transform(common.X_test_raw)
    return {
        "processor": processor,
        "X_train_processed": X_train,
        "X_valid_processed": X_valid,
        "X_test_processed": X_test,
        "metadata": processor.metadata(),
    }


def transform_p03(common: CommonData) -> dict[str, object]:
    groups = {
        "G0": {
            "mask_train": common.X_train_raw[JET_NUM_COLUMN] == 0,
            "mask_valid": common.X_valid_raw[JET_NUM_COLUMN] == 0,
            "mask_test": common.X_test_raw[JET_NUM_COLUMN] == 0,
            "processor": GroupPreprocessor(group_id="G0", drop_columns=P03_G0_DROP_COLUMNS, add_jet_num_is_3=False),
        },
        "G1": {
            "mask_train": common.X_train_raw[JET_NUM_COLUMN] == 1,
            "mask_valid": common.X_valid_raw[JET_NUM_COLUMN] == 1,
            "mask_test": common.X_test_raw[JET_NUM_COLUMN] == 1,
            "processor": GroupPreprocessor(group_id="G1", drop_columns=P03_G1_DROP_COLUMNS, add_jet_num_is_3=False),
        },
        "G2": {
            "mask_train": common.X_train_raw[JET_NUM_COLUMN] >= 2,
            "mask_valid": common.X_valid_raw[JET_NUM_COLUMN] >= 2,
            "mask_test": common.X_test_raw[JET_NUM_COLUMN] >= 2,
            "processor": GroupPreprocessor(group_id="G2", drop_columns=[JET_NUM_COLUMN], add_jet_num_is_3=True),
        },
    }
    outputs: dict[str, object] = {}
    summary_rows: list[dict[str, object]] = []
    for group_id, group in groups.items():
        processor = group["processor"]
        train_mask = group["mask_train"]
        valid_mask = group["mask_valid"]
        test_mask = group["mask_test"]
        X_train_raw = common.X_train_raw.loc[train_mask].reset_index(drop=True)
        X_valid_raw = common.X_valid_raw.loc[valid_mask].reset_index(drop=True)
        X_test_raw = common.X_test_raw.loc[test_mask].reset_index(drop=True)
        processor.fit(X_train_raw)
        X_train = processor.transform(X_train_raw)
        X_valid = processor.transform(X_valid_raw)
        X_test = processor.transform(X_test_raw)
        outputs[group_id] = {
            "processor": processor,
            "X_train_processed": X_train,
            "X_valid_processed": X_valid,
            "X_test_processed": X_test,
            "y_train": common.y_train.loc[train_mask].reset_index(drop=True),
            "y_valid": common.y_valid.loc[valid_mask].reset_index(drop=True),
            "train_weight": common.train_weight.loc[train_mask].reset_index(drop=True),
            "valid_weight": common.valid_weight.loc[valid_mask].reset_index(drop=True),
            "train_event_id": common.train_event_id.loc[train_mask].reset_index(drop=True),
            "valid_event_id": common.valid_event_id.loc[valid_mask].reset_index(drop=True),
            "test_event_id": common.test_event_id.loc[test_mask].reset_index(drop=True),
            "metadata": processor.metadata(),
        }
        summary_rows.append(
            {
                "group": group_id,
                "train_rows": int(len(X_train)),
                "valid_rows": int(len(X_valid)),
                "test_rows": int(len(X_test)),
                "feature_count": int(X_train.shape[1]),
                "train_nan_count": int(X_train.isna().sum().sum()),
                "valid_nan_count": int(X_valid.isna().sum().sum()),
                "test_nan_count": int(X_test.isna().sum().sum()),
            }
        )
    return {"groups": outputs, "group_summary": pd.DataFrame(summary_rows)}


def split_summary(common: CommonData) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": int(len(common.y_train)),
                "class_b": int((common.y_train == 0).sum()),
                "class_s": int((common.y_train == 1).sum()),
                "weight_sum": float(common.train_weight.sum()),
            },
            {
                "split": "valid",
                "rows": int(len(common.y_valid)),
                "class_b": int((common.y_valid == 0).sum()),
                "class_s": int((common.y_valid == 1).sum()),
                "weight_sum": float(common.valid_weight.sum()),
            },
            {
                "split": "test",
                "rows": int(len(common.X_test_raw)),
                "class_b": np.nan,
                "class_s": np.nan,
                "weight_sum": np.nan,
            },
        ]
    )


def save_common_artifacts(common: CommonData, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "X_train_raw.csv": common.X_train_raw,
        "X_valid_raw.csv": common.X_valid_raw,
        "X_test_raw.csv": common.X_test_raw,
        "y_train.csv": common.y_train.to_frame("Label"),
        "y_valid.csv": common.y_valid.to_frame("Label"),
        "train_weight.csv": common.train_weight.to_frame("Weight"),
        "valid_weight.csv": common.valid_weight.to_frame("Weight"),
        "train_event_id.csv": common.train_event_id.to_frame("EventId"),
        "valid_event_id.csv": common.valid_event_id.to_frame("EventId"),
        "test_event_id.csv": common.test_event_id.to_frame("EventId"),
        "split_summary.csv": split_summary(common),
        "missing_counts_by_column.csv": pd.DataFrame(
            {
                "feature": list(common.metadata["missing_counts_by_column"].keys()),
                "missing_count": list(common.metadata["missing_counts_by_column"].values()),
            }
        ),
    }
    saved: list[Path] = []
    for filename, frame in files.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        saved.append(path)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(common.metadata, indent=2, ensure_ascii=False))
    saved.append(metadata_path)
    return saved


def save_global_pipeline_artifacts(
    pipeline_id: str,
    common: CommonData,
    result: dict[str, object],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "X_train_processed.csv": result["X_train_processed"],
        "X_valid_processed.csv": result["X_valid_processed"],
        "X_test_processed.csv": result["X_test_processed"],
        "y_train.csv": common.y_train.to_frame("Label"),
        "y_valid.csv": common.y_valid.to_frame("Label"),
        "train_weight.csv": common.train_weight.to_frame("Weight"),
        "valid_weight.csv": common.valid_weight.to_frame("Weight"),
        "train_event_id.csv": common.train_event_id.to_frame("EventId"),
        "valid_event_id.csv": common.valid_event_id.to_frame("EventId"),
        "test_event_id.csv": common.test_event_id.to_frame("EventId"),
        "split_summary.csv": split_summary(common),
        "final_feature_order.csv": pd.DataFrame({"feature": result["metadata"]["final_feature_order"]}),
        "missing_indicator_cols.csv": pd.DataFrame({"feature": result["metadata"]["missing_indicator_cols"]}),
        "log_cols.csv": pd.DataFrame({"feature": result["metadata"]["log_cols"]}),
    }
    saved: list[Path] = []
    for filename, frame in files.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        saved.append(path)
    metadata = {
        "pipeline_id": pipeline_id,
        "common_metadata": common.metadata,
        "preprocessor_metadata": result["metadata"],
        "processed_train_shape": list(result["X_train_processed"].shape),
        "processed_valid_shape": list(result["X_valid_processed"].shape),
        "processed_test_shape": list(result["X_test_processed"].shape),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    saved.append(metadata_path)
    pipeline_path = output_dir / "preprocessor.joblib"
    joblib.dump(result["processor"], pipeline_path)
    saved.append(pipeline_path)
    return saved


def save_p03_artifacts(common: CommonData, result: dict[str, object], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    summary_path = output_dir / "group_summary.csv"
    result["group_summary"].to_csv(summary_path, index=False)
    saved.append(summary_path)
    split_path = output_dir / "split_summary.csv"
    split_summary(common).to_csv(split_path, index=False)
    saved.append(split_path)
    for group_id, group in result["groups"].items():
        group_dir = output_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "X_train_processed.csv": group["X_train_processed"],
            "X_valid_processed.csv": group["X_valid_processed"],
            "X_test_processed.csv": group["X_test_processed"],
            "y_train.csv": group["y_train"].to_frame("Label"),
            "y_valid.csv": group["y_valid"].to_frame("Label"),
            "train_weight.csv": group["train_weight"].to_frame("Weight"),
            "valid_weight.csv": group["valid_weight"].to_frame("Weight"),
            "train_event_id.csv": group["train_event_id"].to_frame("EventId"),
            "valid_event_id.csv": group["valid_event_id"].to_frame("EventId"),
            "test_event_id.csv": group["test_event_id"].to_frame("EventId"),
            "final_feature_order.csv": pd.DataFrame({"feature": group["metadata"]["final_feature_order"]}),
            "missing_indicator_cols.csv": pd.DataFrame({"feature": group["metadata"]["missing_indicator_cols"]}),
            "log_cols.csv": pd.DataFrame({"feature": group["metadata"]["log_cols"]}),
        }
        for filename, frame in files.items():
            path = group_dir / filename
            frame.to_csv(path, index=False)
            saved.append(path)
        metadata_path = group_dir / "metadata.json"
        metadata_path.write_text(json.dumps(group["metadata"], indent=2, ensure_ascii=False))
        saved.append(metadata_path)
        pipeline_path = group_dir / "preprocessor.joblib"
        joblib.dump(group["processor"], pipeline_path)
        saved.append(pipeline_path)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "pipeline_id": "P03",
                "common_metadata": common.metadata,
                "group_summary": result["group_summary"].to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    saved.append(metadata_path)
    return saved
