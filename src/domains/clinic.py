"""Pipeline del dominio clinic ambulatorio."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.clinic_settings import (
    CLINIC_DAYS_TO_SIMULATE,
    CLINIC_NUM_EMPLOYEES,
    CLINIC_PROCESSED_DIR,
    CLINIC_RANDOM_SEED,
    CLINIC_START_DATE,
)
from src.clinic.etl import run_clinic_etl
from src.clinic.generate import generate_clinic_raw_data, save_clinic_raw_data
from src.clinic.inference import run_clinic_inference
from src.clinic.reporting import generate_clinic_dashboard
from src.clinic.train import run_clinic_training
from src.domains.base import DomainPipeline

logger = logging.getLogger(__name__)


def run_generate(args: object) -> None:
    employees = int(getattr(args, "employees", None) or CLINIC_NUM_EMPLOYEES)
    days = int(getattr(args, "days", None) or CLINIC_DAYS_TO_SIMULATE)
    seed = int(getattr(args, "seed", None) or CLINIC_RANDOM_SEED)
    logger.info("[clinic] Generando datos raw")
    raw_data = generate_clinic_raw_data(employees=employees, days=days, seed=seed, start_date=CLINIC_START_DATE)
    save_clinic_raw_data(raw_data)


def run_etl(args: object) -> None:
    logger.info("[clinic] Ejecutando ETL")
    run_clinic_etl(employees_path=Path(args.employees_path) if getattr(args, "employees_path", None) else None)


def run_train(_: object) -> None:
    logger.info("[clinic] Entrenando modelos")
    run_clinic_training()


def run_infer(args: object) -> None:
    logger.info("[clinic] Ejecutando inferencia")
    output_csv = Path(getattr(args, "output_csv", None) or (CLINIC_PROCESSED_DIR / "clinic_staffing_predictions.csv"))
    output_metrics = Path(getattr(args, "output_metrics", None) or (CLINIC_PROCESSED_DIR / "clinic_staffing_metrics.json"))
    predictions, metrics, metadata = run_clinic_inference()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    output_metrics.write_text(json.dumps({"metrics": metrics, "best_classifier": metadata["best_classifier"]}, indent=2), encoding="utf-8")
    logger.info("[clinic] Predicciones guardadas en %s", output_csv)
    logger.info("[clinic] Metricas guardadas en %s", output_metrics)


def run_report(_: object) -> None:
    logger.info("[clinic] Generando dashboard ejecutivo")
    output_path = generate_clinic_dashboard()
    logger.info("[clinic] Dashboard guardado en %s", output_path)


PIPELINE = DomainPipeline(
    name="clinic",
    description="Pipeline de clinica ambulatoria con agenda, procedimientos y roles criticos",
    generate=run_generate,
    etl=run_etl,
    train=run_train,
    infer=run_infer,
    report=run_report,
)
