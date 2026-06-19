"""Pipeline del dominio industrial actual."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import PROCESSED_DIR, RAW_DIR
from scripts import _03_train_model_loader
from src.domains.base import DomainPipeline
from src.etl.pipeline import run_pipeline
from src.generators.absenteeism import generate_absenteeism
from src.generators.biometrics import generate_biometrics
from src.generators.employees import generate_employees
from src.generators.work_records import generate_work_records
from src.models.inference import run_inference

logger = logging.getLogger(__name__)


def run_generate(args: object) -> None:
    employees = int(getattr(args, "employees", 200))
    days = int(getattr(args, "days", 180))
    seed = int(getattr(args, "seed", 42))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[industrial] Generando datos sinteticos")
    employees_df = generate_employees(n=employees, seed=seed)
    employees_df.to_csv(RAW_DIR / "employees.csv", index=False)

    work_records_df = generate_work_records(employees_df, days=days, seed=seed)
    work_records_df.to_csv(RAW_DIR / "work_records.csv", index=False)

    biometrics_df = generate_biometrics(employees_df, work_records_df, days=days, seed=seed)
    biometrics_df.to_csv(RAW_DIR / "biometrics.csv", index=False)

    absenteeism_df = generate_absenteeism(employees_df, biometrics_df, work_records_df, seed=seed)
    absenteeism_df.to_csv(RAW_DIR / "absenteeism.csv", index=False)

    logger.info("[industrial] Datos generados en %s", RAW_DIR)


def run_etl(args: object) -> None:
    logger.info("[industrial] Ejecutando ETL")
    run_pipeline(employees_path=getattr(args, "employees_path", None))


def run_train(_: object) -> None:
    logger.info("[industrial] Entrenando modelos")
    _03_train_model_loader.run()


def run_infer(args: object) -> None:
    logger.info("[industrial] Ejecutando inferencia")
    output_csv_arg = getattr(args, "output_csv", None)
    output_metrics_arg = getattr(args, "output_metrics", None)
    output_csv = Path(output_csv_arg) if output_csv_arg else PROCESSED_DIR / "staffing_inference_predictions.csv"
    output_metrics = Path(output_metrics_arg) if output_metrics_arg else PROCESSED_DIR / "staffing_inference_metrics.json"

    scored_df, metrics, metadata = run_inference(create_shap_outputs=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv, index=False)
    output_metrics.write_text(
        json.dumps({"metrics": metrics, "best_classifier": metadata["best_classifier"]}, indent=2),
        encoding="utf-8",
    )
    logger.info("[industrial] Predicciones guardadas en %s", output_csv)
    logger.info("[industrial] Metricas guardadas en %s", output_metrics)


PIPELINE = DomainPipeline(
    name="industrial",
    description="Pipeline original de dotacion industrial",
    generate=run_generate,
    etl=run_etl,
    train=run_train,
    infer=run_infer,
    report=None,
)
