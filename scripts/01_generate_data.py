#!/usr/bin/env python3
"""Script para generar todos los datasets sintéticos de SkillUp."""

import argparse
import sys
from pathlib import Path

# Permitir imports del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR
from src.generators.employees import generate_employees
from src.generators.work_records import generate_work_records
from src.generators.biometrics import generate_biometrics
from src.generators.absenteeism import generate_absenteeism


def main():
    parser = argparse.ArgumentParser(description="Genera datos sintéticos de SkillUp")
    parser.add_argument("--employees", type=int, default=200, help="Número de empleados")
    parser.add_argument("--days", type=int, default=180, help="Días a simular")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generando datos: {args.employees} empleados, {args.days} días, seed={args.seed}")
    print("-" * 60)

    # 1. Empleados
    print("Generando empleados...")
    employees_df = generate_employees(n=args.employees, seed=args.seed)
    employees_df.to_csv(RAW_DIR / "employees.csv", index=False)
    print(f"  employees.csv: {len(employees_df):,} filas")

    # 2. Registros de trabajo
    print("Generando registros de trabajo...")
    work_records_df = generate_work_records(employees_df, days=args.days, seed=args.seed)
    work_records_df.to_csv(RAW_DIR / "work_records.csv", index=False)
    print(f"  work_records.csv: {len(work_records_df):,} filas")

    # 3. Biométricos
    print("Generando datos biométricos...")
    biometrics_df = generate_biometrics(employees_df, work_records_df, days=args.days, seed=args.seed)
    biometrics_df.to_csv(RAW_DIR / "biometrics.csv", index=False)
    print(f"  biometrics.csv: {len(biometrics_df):,} filas")

    # 4. Ausentismo
    print("Generando eventos de ausentismo...")
    absenteeism_df = generate_absenteeism(employees_df, biometrics_df, work_records_df, seed=args.seed)
    absenteeism_df.to_csv(RAW_DIR / "absenteeism.csv", index=False)
    print(f"  absenteeism.csv: {len(absenteeism_df):,} filas")

    print("-" * 60)
    total = len(employees_df) + len(work_records_df) + len(biometrics_df) + len(absenteeism_df)
    print(f"Total: {total:,} filas generadas en {RAW_DIR}")
    print("Listo.")


if __name__ == "__main__":
    main()
