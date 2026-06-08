"""Pipeline del dominio restaurant casual dining."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.restaurant_settings import (
    RESTAURANT_DAYS_TO_SIMULATE,
    RESTAURANT_NUM_EMPLOYEES,
    RESTAURANT_PROCESSED_DIR,
    RESTAURANT_RANDOM_SEED,
    RESTAURANT_START_DATE,
)
from src.domains.base import DomainPipeline
from src.restaurant.etl import run_restaurant_etl
from src.restaurant.generate import generate_restaurant_raw_data, save_restaurant_raw_data
from src.restaurant.inference import run_restaurant_inference
from src.restaurant.reporting import generate_restaurant_dashboard
from src.restaurant.train import run_restaurant_training

logger = logging.getLogger(__name__)


def run_generate(args: object) -> None:
    employees = int(getattr(args, "employees", None) or RESTAURANT_NUM_EMPLOYEES)
    days = int(getattr(args, "days", None) or RESTAURANT_DAYS_TO_SIMULATE)
    seed = int(getattr(args, "seed", None) or RESTAURANT_RANDOM_SEED)
    logger.info("[restaurant] Generando datos raw")
    raw_data = generate_restaurant_raw_data(employees=employees, days=days, seed=seed, start_date=RESTAURANT_START_DATE)
    save_restaurant_raw_data(raw_data)


def run_etl(args: object) -> None:
    logger.info("[restaurant] Ejecutando ETL")
    run_restaurant_etl(employees_path=Path(args.employees_path) if getattr(args, "employees_path", None) else None)


def run_train(_: object) -> None:
    logger.info("[restaurant] Entrenando modelos")
    run_restaurant_training()


def run_infer(args: object) -> None:
    logger.info("[restaurant] Ejecutando inferencia")
    output_csv = Path(getattr(args, "output_csv", None) or (RESTAURANT_PROCESSED_DIR / "restaurant_staffing_predictions.csv"))
    output_metrics = Path(getattr(args, "output_metrics", None) or (RESTAURANT_PROCESSED_DIR / "restaurant_staffing_metrics.json"))
    predictions, metrics, metadata = run_restaurant_inference()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    output_metrics.write_text(json.dumps({"metrics": metrics, "best_classifier": metadata["best_classifier"]}, indent=2), encoding="utf-8")
    logger.info("[restaurant] Predicciones guardadas en %s", output_csv)
    logger.info("[restaurant] Metricas guardadas en %s", output_metrics)


def run_report(_: object) -> None:
    logger.info("[restaurant] Generando dashboard ejecutivo")
    output_path = generate_restaurant_dashboard()
    logger.info("[restaurant] Dashboard guardado en %s", output_path)


PIPELINE = DomainPipeline(
    name="restaurant",
    description="Pipeline de restaurant casual dining con calendario Chile",
    generate=run_generate,
    etl=run_etl,
    train=run_train,
    infer=run_infer,
    report=run_report,
)
