"""Carga de datos a SQLite."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    name TEXT,
    age INTEGER,
    gender TEXT,
    bmi REAL,
    education_level INTEGER,
    plant_area TEXT,
    position TEXT,
    seniority_years INTEGER,
    shift_pattern TEXT,
    distance_to_work_km REAL,
    children INTEGER,
    social_drinker INTEGER,
    smoker INTEGER,
    hire_date TEXT
);

CREATE TABLE IF NOT EXISTS biometrics_raw (
    employee_id TEXT,
    date TEXT,
    hr_mean_bpm REAL,
    hr_min_bpm REAL,
    hr_max_bpm REAL,
    hr_std_bpm REAL,
    hrv_rmssd_ms REAL,
    spo2_mean_pct REAL,
    spo2_min_pct REAL,
    skin_temp_mean_c REAL,
    sleep_duration_hours REAL,
    sleep_efficiency_pct REAL,
    deep_sleep_pct REAL,
    stress_score REAL,
    steps INTEGER,
    data_quality_score REAL,
    PRIMARY KEY (employee_id, date)
);

CREATE TABLE IF NOT EXISTS biometrics_clean (
    employee_id TEXT,
    date TEXT,
    hr_mean_bpm REAL,
    hr_min_bpm REAL,
    hr_max_bpm REAL,
    hr_std_bpm REAL,
    hrv_rmssd_ms REAL,
    spo2_mean_pct REAL,
    spo2_min_pct REAL,
    skin_temp_mean_c REAL,
    sleep_duration_hours REAL,
    sleep_efficiency_pct REAL,
    deep_sleep_pct REAL,
    stress_score REAL,
    steps INTEGER,
    data_quality_score REAL,
    PRIMARY KEY (employee_id, date)
);

CREATE TABLE IF NOT EXISTS work_records (
    employee_id TEXT,
    date TEXT,
    shift TEXT,
    hours_worked REAL,
    overtime_hours REAL,
    PRIMARY KEY (employee_id, date)
);

CREATE TABLE IF NOT EXISTS ml_features (
    plant_area TEXT,
    shift TEXT,
    date TEXT,
    PRIMARY KEY (plant_area, shift, date)
);
"""


def create_schema(conn) -> None:
    """Crea las tablas en SQLite si no existen."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.info("Schema creado/verificado")


def load_dataframe(
    df: pd.DataFrame,
    table_name: str,
    conn,
    if_exists: str = "replace",
) -> None:
    """Carga un DataFrame a una tabla SQLite."""
    # Convertir columnas datetime a string para SQLite
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_datetime64_any_dtype(df_out[col]):
            df_out[col] = df_out[col].dt.strftime("%Y-%m-%d")

    df_out.to_sql(table_name, conn, if_exists=if_exists, index=False)
    logger.info("Tabla '%s' cargada: %d filas", table_name, len(df_out))
