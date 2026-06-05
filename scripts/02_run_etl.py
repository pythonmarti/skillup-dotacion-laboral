"""Script para ejecutar el pipeline ETL completo."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.etl.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline ETL de SkillUp")
    parser.add_argument(
        "--employees-path",
        type=Path,
        default=None,
        help="Ruta alternativa para la fuente de empleados, por ejemplo data/raw/employees_from_forms.csv",
    )
    args = parser.parse_args()
    run_pipeline(employees_path=args.employees_path)
