"""Pipeline de transformacion en 7 pasos."""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from config.settings import PHYSIO_RANGES

logger = logging.getLogger(__name__)

BIOMETRIC_COLS = [
    "hr_mean_bpm", "hr_min_bpm", "hr_max_bpm", "hr_std_bpm",
    "hrv_rmssd_ms", "spo2_mean_pct", "spo2_min_pct",
    "skin_temp_mean_c", "sleep_duration_hours", "sleep_efficiency_pct",
    "deep_sleep_pct", "stress_score", "steps",
]

ROLLING_COLS = ["hr_mean_bpm", "hrv_rmssd_ms", "stress_score", "sleep_duration_hours"]

LAG_COLS = ["hr_mean_bpm", "hrv_rmssd_ms", "stress_score", "sleep_duration_hours", "steps"]
LAG_DAYS = [1, 3, 5, 7]


# --- Paso 1 ---
def validate_physiological_ranges(bio_df: pd.DataFrame) -> pd.DataFrame:
    """Valores fuera de PHYSIO_RANGES -> NaN. Retorna df limpio."""
    df = bio_df.copy()
    invalidated = {}
    for col, (lo, hi) in PHYSIO_RANGES.items():
        if col not in df.columns:
            continue
        mask = ~df[col].between(lo, hi) & df[col].notna()
        count = int(mask.sum())
        if count:
            invalidated[col] = count
            df.loc[mask, col] = np.nan
    if invalidated:
        logger.info("Valores invalidados por rango: %s", invalidated)
    return df


# --- Paso 2 ---
def remove_artifacts(bio_df: pd.DataFrame) -> pd.DataFrame:
    """Elimina artefactos de calidad baja y HR constante."""
    df = bio_df.copy()
    bio_cols_present = [c for c in BIOMETRIC_COLS if c in df.columns]

    # Registros con data_quality_score < 0.3 -> biometricas a NaN
    if "data_quality_score" in df.columns:
        low_quality = df["data_quality_score"] < 0.3
        n_low = int(low_quality.sum())
        if n_low:
            df.loc[low_quality, bio_cols_present] = np.nan
            logger.info("Registros baja calidad (< 0.3): %d", n_low)

    # HR constante (std < 0.5) por >4 horas -> NaN en HR
    if "hr_std_bpm" in df.columns:
        hr_cols = [c for c in ["hr_mean_bpm", "hr_min_bpm", "hr_max_bpm",
                                "hr_std_bpm", "hrv_rmssd_ms"] if c in df.columns]
        constant_hr = df["hr_std_bpm"] < 0.5
        if "hours_worked" in df.columns:
            constant_hr = constant_hr & (df["hours_worked"] > 4)
        n_const = int(constant_hr.sum())
        if n_const:
            df.loc[constant_hr, hr_cols] = np.nan
            logger.info("Registros HR constante: %d", n_const)

    return df


# --- Paso 3 ---
def impute_missing_values(bio_df: pd.DataFrame) -> pd.DataFrame:
    """Imputa NaN: interpolacion lineal para gaps <=2 dias, KNN para gaps >2."""
    df = bio_df.copy()
    bio_cols_present = [c for c in BIOMETRIC_COLS if c in df.columns]
    if not bio_cols_present:
        return df

    df = df.sort_values(["employee_id", "date"]).reset_index(drop=True)

    for emp_id, group in df.groupby("employee_id"):
        idx = group.index
        for col in bio_cols_present:
            series = group[col].copy()
            if series.isna().sum() == 0:
                continue

            # Identificar gaps consecutivos
            is_null = series.isna()
            gap_groups = (~is_null).cumsum()
            gap_sizes = is_null.groupby(gap_groups).transform("sum")

            # Interpolacion lineal para gaps <= 2
            short_gap_mask = is_null & (gap_sizes <= 2)
            if short_gap_mask.any():
                interpolated = series.interpolate(method="linear")
                series[short_gap_mask] = interpolated[short_gap_mask]

            df.loc[idx, col] = series

    # KNN para los NaN restantes (gaps > 2 dias)
    remaining_nulls = df[bio_cols_present].isna().any(axis=1).sum()
    if remaining_nulls > 0:
        logger.info("Aplicando KNN Imputer para %d registros con NaN restantes", remaining_nulls)
        for emp_id, group in df.groupby("employee_id"):
            idx = group.index
            subset = group[bio_cols_present]
            if subset.isna().any().any() and subset.notna().any().any():
                imputer = KNNImputer(n_neighbors=min(5, len(subset)))
                imputed = imputer.fit_transform(subset)
                df.loc[idx, bio_cols_present] = imputed

    return df


# --- Paso 4 ---
def normalize_by_individual(bio_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score por employee_id para columnas biometricas."""
    df = bio_df.copy()
    bio_cols_present = [c for c in BIOMETRIC_COLS if c in df.columns]

    for col in bio_cols_present:
        zscore_col = f"{col}_zscore"
        df[zscore_col] = df.groupby("employee_id")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )

    return df


# --- Paso 5 ---
def create_rolling_features(bio_df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean/std 7 dias, trend, y fatiga acumulada 14 dias."""
    df = bio_df.copy()
    df = df.sort_values(["employee_id", "date"]).reset_index(drop=True)

    rolling_cols_present = [c for c in ROLLING_COLS if c in df.columns]

    for col in rolling_cols_present:
        mean_col = f"{col}_7d_mean"
        std_col = f"{col}_7d_std"
        trend_col = f"{col}_trend"

        df[mean_col] = df.groupby("employee_id")[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df[std_col] = df.groupby("employee_id")[col].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )
        # Trend: media actual 7d vs media 7d anterior
        df[trend_col] = df.groupby("employee_id")[mean_col].transform(
            lambda x: x - x.shift(7)
        )

    # Fatiga acumulada: suma stress_score ultimos 14 dias
    if "stress_score" in df.columns:
        df["fatigue_14d"] = df.groupby("employee_id")["stress_score"].transform(
            lambda x: x.rolling(14, min_periods=1).sum()
        )

    # Ventanas rolling adicionales: 3d, 14d, 30d
    extra_windows = [3, 14, 30]
    for col in rolling_cols_present:
        for w in extra_windows:
            mean_col = f"{col}_{w}d_mean"
            std_col = f"{col}_{w}d_std"
            df[mean_col] = df.groupby("employee_id")[col].transform(
                lambda x, window=w: x.rolling(window, min_periods=1).mean()
            )
            df[std_col] = df.groupby("employee_id")[col].transform(
                lambda x, window=w: x.rolling(window, min_periods=1).std()
            )

    # Fatiga 30d
    if "stress_score" in df.columns:
        df["fatigue_30d"] = df.groupby("employee_id")["stress_score"].transform(
            lambda x: x.rolling(30, min_periods=1).sum()
        )

    return df


# --- Paso 5b ---
def create_lag_features(bio_df: pd.DataFrame) -> pd.DataFrame:
    """Crea lag features de 1, 3, 5, 7 dias para variables biometricas clave."""
    df = bio_df.copy()
    df = df.sort_values(["employee_id", "date"]).reset_index(drop=True)

    lag_cols_present = [c for c in LAG_COLS if c in df.columns]

    for col in lag_cols_present:
        for lag in LAG_DAYS:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby("employee_id")[col].shift(lag)

    return df


# --- Paso 5c ---
def create_baseline_deviation_features(bio_df: pd.DataFrame) -> pd.DataFrame:
    """Desviacion de cada metrica biometrica respecto a la media individual del empleado."""
    df = bio_df.copy()
    bio_cols_present = [c for c in BIOMETRIC_COLS if c in df.columns]

    for col in bio_cols_present:
        dev_col = f"{col}_baseline_dev"
        df[dev_col] = df.groupby("employee_id")[col].transform(
            lambda x: x - x.mean()
        )

    return df


# --- Paso 6b ---
def create_temporal_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Features temporales contextuales: dia de la semana, indicadores, overtime rolling."""
    df = merged_df.copy()

    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        df["day_of_week"] = dt.dt.dayofweek
        df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)
        df["is_friday"] = (dt.dt.dayofweek == 4).astype(int)
        df["is_weekend_adjacent"] = ((dt.dt.dayofweek == 0) | (dt.dt.dayofweek == 4)).astype(int)
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    if "overtime_hours" in df.columns:
        df = df.sort_values(["employee_id", "date"]).reset_index(drop=True)
        df["overtime_7d_sum"] = df.groupby("employee_id")["overtime_hours"].transform(
            lambda x: x.rolling(7, min_periods=1).sum()
        )
        df["overtime_14d_sum"] = df.groupby("employee_id")["overtime_hours"].transform(
            lambda x: x.rolling(14, min_periods=1).sum()
        )

    return df


# --- Paso 6 ---
def merge_all_sources(
    emp_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    work_df: pd.DataFrame,
    abs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left join de todas las fuentes sobre employee_id + date."""
    # Asegurar que date sea datetime en todos
    for label, d in [("bio", bio_df), ("work", work_df), ("abs", abs_df)]:
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])

    merged = bio_df.copy()

    # Merge work records
    if not work_df.empty:
        merged = merged.merge(work_df, on=["employee_id", "date"], how="left")

    # Merge absenteeism
    if not abs_df.empty:
        merged = merged.merge(abs_df, on=["employee_id", "date"], how="left")

    # Merge employee demographics
    if not emp_df.empty:
        merged = merged.merge(emp_df, on="employee_id", how="left")

    return merged


# --- Paso 7 ---
def create_target_variable(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables objetivo: absent_next_7days y absence_hours_next_7days."""
    df = merged_df.copy()
    df = df.sort_values(["employee_id", "date"]).reset_index(drop=True)

    # Asegurar columnas de ausencia
    if "is_absent" not in df.columns:
        df["is_absent"] = 0
    if "absence_hours" not in df.columns:
        df["absence_hours"] = 0.0

    df["is_absent"] = df["is_absent"].fillna(0).astype(int)
    df["absence_hours"] = df["absence_hours"].fillna(0.0)

    # Calcular ventana futura de 7 dias por empleado
    df["absent_next_7days"] = 0
    df["absence_hours_next_7days"] = 0.0

    for emp_id, group in df.groupby("employee_id"):
        idx = group.index
        absent_vals = group["is_absent"].values
        hours_vals = group["absence_hours"].values

        n = len(group)
        next_7d_absent = np.zeros(n, dtype=int)
        next_7d_hours = np.zeros(n, dtype=float)

        for i in range(n):
            window_end = min(i + 8, n)  # i+1 a i+7 inclusive
            window = slice(i + 1, window_end)
            next_7d_absent[i] = 1 if absent_vals[window].sum() > 0 else 0
            next_7d_hours[i] = hours_vals[window].sum()

        df.loc[idx, "absent_next_7days"] = next_7d_absent
        df.loc[idx, "absence_hours_next_7days"] = next_7d_hours

    return df


# --- Orquestador de transformaciones ---
def run_transforms(
    emp_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    work_df: pd.DataFrame,
    abs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ejecuta los 7 pasos de transformacion en orden."""
    logger.info("Paso 1: Validacion de rangos fisiologicos")
    bio_df = validate_physiological_ranges(bio_df)

    logger.info("Paso 2: Remocion de artefactos")
    bio_df = remove_artifacts(bio_df)

    logger.info("Paso 3: Imputacion de valores faltantes")
    bio_df = impute_missing_values(bio_df)

    logger.info("Paso 4: Normalizacion por individuo (Z-score)")
    bio_df = normalize_by_individual(bio_df)

    logger.info("Paso 5: Features de ventana movil")
    bio_df = create_rolling_features(bio_df)

    logger.info("Paso 5b: Lag features")
    bio_df = create_lag_features(bio_df)

    logger.info("Paso 5c: Baseline deviation features")
    bio_df = create_baseline_deviation_features(bio_df)

    logger.info("Paso 6: Merge de todas las fuentes")
    merged = merge_all_sources(emp_df, bio_df, work_df, abs_df)

    logger.info("Paso 6b: Features temporales")
    merged = create_temporal_features(merged)

    logger.info("Paso 7: Creacion de variable objetivo")
    result = create_target_variable(merged)

    logger.info("Transformacion completada: %d filas, %d columnas",
                len(result), len(result.columns))
    return result
