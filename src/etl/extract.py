"""Extraccion de CSVs con parseo de tipos y validacion de columnas."""

import pandas as pd
from config.settings import RAW_DIR

EXPECTED_COLUMNS = {
    "employees": [
        "employee_id", "name", "age", "gender", "bmi", "education_level",
        "plant_area", "position", "seniority_years", "shift_pattern",
        "distance_to_work_km", "children", "social_drinker", "smoker",
        "hire_date",
    ],
    "biometrics": [
        "employee_id", "date", "hr_mean_bpm", "hr_min_bpm", "hr_max_bpm",
        "hr_std_bpm", "hrv_rmssd_ms", "spo2_mean_pct", "spo2_min_pct",
        "skin_temp_mean_c", "sleep_duration_hours", "sleep_efficiency_pct",
        "deep_sleep_pct", "stress_score", "steps", "data_quality_score",
    ],
    "work_records": [
        "employee_id", "date", "shift", "area_assigned", "workload_score",
        "hours_worked", "overtime_hours", "consecutive_work_days",
        "is_holiday", "is_rest_day",
    ],
    "absenteeism": [
        "employee_id", "date", "absence_reason", "absence_hours", "is_absent",
    ],
}


def _validate_columns(df: pd.DataFrame, name: str) -> None:
    expected = set(EXPECTED_COLUMNS[name])
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise ValueError(
            f"{name}: faltan columnas esperadas: {sorted(missing)}"
        )


def extract_employees(path=None) -> pd.DataFrame:
    path = path or RAW_DIR / "employees.csv"
    df = pd.read_csv(path, parse_dates=["hire_date"])
    _validate_columns(df, "employees")
    return df


def extract_biometrics(path=None) -> pd.DataFrame:
    path = path or RAW_DIR / "biometrics.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    _validate_columns(df, "biometrics")
    return df


def extract_work_records(path=None) -> pd.DataFrame:
    path = path or RAW_DIR / "work_records.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    _validate_columns(df, "work_records")
    return df


def extract_absenteeism(path=None) -> pd.DataFrame:
    path = path or RAW_DIR / "absenteeism.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    _validate_columns(df, "absenteeism")
    return df
