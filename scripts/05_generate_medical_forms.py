#!/usr/bin/env python3
"""Script para generar fichas medicas PDF desde employees.csv."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR
from src.generators.medical_forms import generate_medical_forms_from_csv

logger = logging.getLogger(__name__)


def main() -> None:
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Genera fichas medicas PDF OCR-friendly desde employees.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "employees.csv",
        help="Ruta al archivo employees.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR / "medical_forms",
        help="Directorio de salida para los PDFs",
    )
    parser.add_argument(
        "--include-structured-page",
        action="store_true",
        help="Agrega una segunda pagina con datos estructurados para digitalizacion",
    )
    args = parser.parse_args()

    logger.info("Generando fichas medicas desde %s", args.input)
    generated_files = generate_medical_forms_from_csv(
        args.input,
        args.output_dir,
        include_structured_page=args.include_structured_page,
    )
    logger.info("Proceso completado: %d archivos en %s", len(generated_files), args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
