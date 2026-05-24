#!/usr/bin/env python3
"""Script para generar todos los datasets sintéticos de SkillUp."""

import argparse
import logging
import sys
from pathlib import Path

# Permitir imports del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR
from src.generators.employees import generate_employees
from src.generators.work_records import generate_work_records
from src.generators.biometrics import generate_biometrics
from src.generators.absenteeism import generate_absenteeism

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Genera datos sintéticos de SkillUp")
    parser.add_argument("--employees", type=int, default=200, help="Número de empleados")
    parser.add_argument("--days", type=int, default=180, help="Días a simular")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Iniciando generacion de datos sinteticos: %d empleados, %d dias, seed=%d",
        args.employees, args.days, args.seed,
    )

    # 1. Empleados
    logger.info("[1/4] Generando perfiles de empleados...")
    employees_df = generate_employees(n=args.employees, seed=args.seed)
    employees_df.to_csv(RAW_DIR / "employees.csv", index=False)
    logger.info("  Guardado: employees.csv (%d filas)", len(employees_df))

    # 2. Registros de trabajo
    logger.info("[2/4] Generando registros de trabajo...")
    work_records_df = generate_work_records(employees_df, days=args.days, seed=args.seed)
    work_records_df.to_csv(RAW_DIR / "work_records.csv", index=False)
    logger.info("  Guardado: work_records.csv (%d filas)", len(work_records_df))

    # 3. Biométricos
    logger.info("[3/4] Generando datos biometricos de wearables...")
    biometrics_df = generate_biometrics(employees_df, work_records_df, days=args.days, seed=args.seed)
    biometrics_df.to_csv(RAW_DIR / "biometrics.csv", index=False)
    logger.info("  Guardado: biometrics.csv (%d filas)", len(biometrics_df))

    # 4. Ausentismo
    logger.info("[4/4] Generando eventos de ausentismo...")
    absenteeism_df = generate_absenteeism(employees_df, biometrics_df, work_records_df, seed=args.seed)
    absenteeism_df.to_csv(RAW_DIR / "absenteeism.csv", index=False)
    logger.info("  Guardado: absenteeism.csv (%d filas)", len(absenteeism_df))

    total = len(employees_df) + len(work_records_df) + len(biometrics_df) + len(absenteeism_df)
    logger.info(
        "Generacion completada: %d filas totales en %s",
        total, RAW_DIR,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
