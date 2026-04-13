"""Script para ejecutar el pipeline ETL completo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.etl.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
