#!/usr/bin/env python3
"""Extrae fichas medicas PDF, valida el CSV y ejecuta el ETL."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR
from src.etl.pipeline import run_pipeline
from src.extraction.medical_forms import (
    extract_medical_forms_dir_to_csv,
    validate_employees_csv_match,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae fichas medicas PDF, valida el CSV y ejecuta el ETL"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR / "medical_forms",
        help="Directorio que contiene los PDFs",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RAW_DIR / "employees_from_forms.csv",
        help="Ruta del CSV intermedio de salida",
    )
    parser.add_argument(
        "--validate-against",
        type=Path,
        default=RAW_DIR / "employees.csv",
        help="CSV de referencia para validacion exacta",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Patron de PDFs a procesar",
    )
    parser.add_argument(
        "--disable-ocr-fallback",
        action="store_true",
        help="Desactiva el intento de OCR si el PDF no tiene texto util",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Omite la validacion exacta contra el CSV de referencia",
    )
    args = parser.parse_args()

    logger.info("[1/3] Extrayendo fichas medicas desde %s", args.input_dir)
    df = extract_medical_forms_dir_to_csv(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        pattern=args.pattern,
        allow_ocr_fallback=not args.disable_ocr_fallback,
    )
    logger.info("CSV generado: %d filas en %s", len(df), args.output_csv)

    if not args.skip_validation:
        logger.info("[2/3] Validando CSV contra %s", args.validate_against)
        validate_employees_csv_match(args.validate_against, args.output_csv)
        logger.info("Validacion exacta OK")
    else:
        logger.info("[2/3] Validacion omitida")

    logger.info("[3/3] Ejecutando ETL con %s", args.output_csv)
    run_pipeline(employees_path=args.output_csv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
