"""Orquestador del pipeline ETL."""

import logging
from pathlib import Path

from config.settings import DB_PATH
from src.utils.database import get_connection
from src.etl.extract import (
    extract_employees,
    extract_biometrics,
    extract_work_records,
    extract_absenteeism,
)
from src.etl.transform import run_transforms
from src.etl.load import create_schema, load_dataframe

logger = logging.getLogger(__name__)


def run_pipeline(employees_path: Path | None = None) -> None:
    """Ejecuta el pipeline ETL completo: Extract -> Transform -> Load."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Extract ---
    logger.info("=== EXTRACT ===")
    emp_df = extract_employees(path=employees_path)
    logger.info("Empleados: %d filas", len(emp_df))

    bio_df = extract_biometrics()
    logger.info("Biometrics: %d filas", len(bio_df))

    work_df = extract_work_records()
    logger.info("Work records: %d filas", len(work_df))

    abs_df = extract_absenteeism()
    logger.info("Absenteeism: %d filas", len(abs_df))

    # --- Transform ---
    logger.info("=== TRANSFORM ===")
    bio_raw = bio_df.copy()
    bio_clean, ml_features = run_transforms(emp_df, bio_df, work_df, abs_df)

    # --- Load ---
    logger.info("=== LOAD ===")
    with get_connection(DB_PATH) as conn:
        create_schema(conn)
        load_dataframe(emp_df, "employees", conn)
        load_dataframe(bio_raw, "biometrics_raw", conn)
        load_dataframe(bio_clean, "biometrics_clean", conn)
        load_dataframe(work_df, "work_records", conn)
        load_dataframe(ml_features, "ml_features", conn)

    logger.info("=== PIPELINE COMPLETADO ===")
    logger.info("Base de datos: %s", DB_PATH)
    logger.info("Features ML: %d filas x %d columnas",
                len(ml_features), len(ml_features.columns))
