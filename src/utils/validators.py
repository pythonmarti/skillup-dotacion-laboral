"""Validacion de rangos fisiologicos y calidad de datos."""

import pandas as pd
import numpy as np
from config.settings import PHYSIO_RANGES


def validate_range(series: pd.Series, col_name: str) -> pd.Series:
    """Reemplaza valores fuera de rango fisiologico con NaN."""
    if col_name not in PHYSIO_RANGES:
        return series
    lo, hi = PHYSIO_RANGES[col_name]
    return series.where(series.between(lo, hi))


def report_data_quality(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Genera un reporte de calidad de datos."""
    total = len(df)
    report = {
        "name": name,
        "total_rows": total,
        "columns": {},
    }
    for col in df.columns:
        col_info = {
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not df[col].isna().all() else None
            col_info["max"] = float(df[col].max()) if not df[col].isna().all() else None
            if col in PHYSIO_RANGES:
                lo, hi = PHYSIO_RANGES[col]
                out_of_range = (~df[col].between(lo, hi) & df[col].notna()).sum()
                col_info["out_of_range"] = int(out_of_range)
        report["columns"][col] = col_info
    return report
