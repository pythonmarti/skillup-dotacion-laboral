#!/usr/bin/env python3
"""Script para extraer fichas medicas PDF a CSV estructurado."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR
from src.extraction.medical_forms import extract_medical_forms_dir_to_csv, validate_employees_csv_match

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae fichas medicas PDF a un CSV compatible con employees.csv"
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
        help="Ruta del CSV de salida",
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
        "--validate-against",
        type=Path,
        default=None,
        help="Valida que el CSV extraido coincida exactamente con este archivo de referencia",
    )
    args = parser.parse_args()

    logger.info("Extrayendo fichas medicas desde %s", args.input_dir)
    df = extract_medical_forms_dir_to_csv(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        pattern=args.pattern,
        allow_ocr_fallback=not args.disable_ocr_fallback,
    )
    if args.validate_against is not None:
        validate_employees_csv_match(args.validate_against, args.output_csv)
        logger.info("Validacion exacta OK contra %s", args.validate_against)
    logger.info("Extraccion completada: %d filas guardadas en %s", len(df), args.output_csv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
