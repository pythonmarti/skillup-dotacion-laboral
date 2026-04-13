"""Preparacion de la matriz de features para modelos predictivos."""

import numpy as np
import pandas as pd
from src.utils.database import query_to_dataframe


def temporal_train_test_split(df, date_col="date", test_ratio=0.2):
    """Divide el DataFrame por fecha para validacion temporal.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ordenado por fecha.
    date_col : str
        Columna de fecha.
    test_ratio : float
        Proporcion de datos para test (las fechas mas recientes).

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]

    train_df = df[pd.to_datetime(df[date_col]) < split_date].copy()
    test_df = df[pd.to_datetime(df[date_col]) >= split_date].copy()

    print(f"  Split temporal: train={len(train_df)} ({split_idx} dias), "
          f"test={len(test_df)} ({len(dates) - split_idx} dias)")
    print(f"  Fecha de corte: {split_date}")

    return train_df, test_df


def prepare_feature_matrix(db_path=None):
    """Lee ml_features de SQLite y prepara X, y para entrenamiento.

    Returns
    -------
    dict
        Keys: X, y_class, y_reg, feature_names, date
    """
    df = query_to_dataframe("SELECT * FROM ml_features", db_path=db_path)

    # Ordenar temporalmente
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "employee_id"]).reset_index(drop=True)

    # Columnas categoricas a one-hot
    categorical_cols = [
        "plant_area", "shift", "position", "gender",
        "shift_pattern", "area_assigned",
    ]
    if "absence_reason" in df.columns:
        categorical_cols.append("absence_reason")

    existing_cats = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=False, dtype=float)

    # Interaction terms: workload_score * variables de area de riesgo
    risk_cols = [c for c in df.columns if c.startswith("plant_area_")]
    if "workload_score" in df.columns:
        for col in risk_cols:
            df[f"workload_x_{col}"] = df["workload_score"] * df[col]

    # Ratios e interacciones biometricas
    if "hr_mean_bpm" in df.columns and "hrv_rmssd_ms" in df.columns:
        df["hr_hrv_ratio"] = df["hr_mean_bpm"] / (df["hrv_rmssd_ms"] + 1)

    if "sleep_efficiency_pct" in df.columns and "stress_score" in df.columns:
        df["sleep_stress_interaction"] = df["sleep_efficiency_pct"] * df["stress_score"]

    if "sleep_duration_hours" in df.columns and "stress_score" in df.columns:
        df["sleep_debt_x_stress"] = (7 - df["sleep_duration_hours"]).clip(lower=0) * df["stress_score"]

    if "workload_score" in df.columns and "stress_score" in df.columns:
        df["workload_x_stress"] = df["workload_score"] * df["stress_score"]

    if "consecutive_work_days" in df.columns and "fatigue_14d" in df.columns:
        df["consec_x_fatigue"] = df["consecutive_work_days"] * df["fatigue_14d"]

    # Separar targets
    target_class = "absent_next_7days"
    target_reg = "absence_hours_next_7days"

    y_class = df[target_class].astype(int) if target_class in df.columns else None
    y_reg = df[target_reg] if target_reg in df.columns else None

    # Drop columnas no-features
    drop_cols = [
        "employee_id", "date", "name", "hire_date",
        target_class, target_reg,
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Drop any remaining string/object columns not encoded
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X = X.drop(columns=obj_cols)

    # Fill any remaining NaN with 0 (rolling features at edges)
    X = X.fillna(0)

    feature_names = list(X.columns)

    result = {
        "X": X,
        "y_class": y_class,
        "y_reg": y_reg,
        "feature_names": feature_names,
    }

    # Preservar fecha para split temporal
    if "date" in df.columns:
        result["date"] = df["date"].values

    return result
