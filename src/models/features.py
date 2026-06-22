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
    train_df, test_df, split_date : (pd.DataFrame, pd.DataFrame, str)
        train_df: datos de entrenamiento (fechas anteriores al corte).
        test_df: datos de test (fechas desde el corte en adelante).
        split_date: fecha de corte en formato ISO (YYYY-MM-DD).
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = pd.Timestamp(dates[split_idx])

    train_df = df[pd.to_datetime(df[date_col]) < split_date].copy()
    test_df = df[pd.to_datetime(df[date_col]) >= split_date].copy()

    split_date_str = split_date.strftime("%Y-%m-%d")
    print(f"  Split temporal: train={len(train_df)} ({split_idx} dias), "
          f"test={len(test_df)} ({len(dates) - split_idx} dias)")
    print(f"  Fecha de corte: {split_date_str}")

    return train_df, test_df, split_date_str


def prepare_feature_matrix(db_path=None):
    """Lee ml_features de SQLite y prepara X, y para entrenamiento a nivel de dotacion.

    Returns
    -------
    dict
        Keys: X, y_class, y_reg, feature_names, date
    """
    df = query_to_dataframe("SELECT * FROM ml_features", db_path=db_path)

    # Ordenar temporalmente
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        sort_cols = ["date"]
        if "plant_area" in df.columns:
            sort_cols.append("plant_area")
        if "shift" in df.columns:
            sort_cols.append("shift")
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Columnas categoricas a one-hot
    categorical_cols = ["plant_area", "shift"]
    existing_cats = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=False, dtype=float)

    # Separar targets
    target_class = "has_deficit"
    target_reg = "actual_headcount"

    y_class = df[target_class].astype(int) if target_class in df.columns else None
    y_reg = df[target_reg] if target_reg in df.columns else None

    # Drop columnas no-features
    drop_cols = [
        "date", target_class, target_reg, "deficit_count", "absentee_rate", "scheduled_headcount"
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
