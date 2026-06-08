#!/usr/bin/env python3
"""Ejecuta inferencia sobre ml_features y guarda predicciones/metricas."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DIR
from src.models.inference import run_inference

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta inferencia usando los artefactos entrenados y guarda predicciones"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROCESSED_DIR / "staffing_inference_predictions.csv",
        help="Ruta de salida para las predicciones",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=PROCESSED_DIR / "staffing_inference_metrics.json",
        help="Ruta de salida para las metricas de inferencia",
    )
    args = parser.parse_args()

    scored_df, metrics, metadata = run_inference()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(args.output_csv, index=False)
    args.output_metrics.write_text(
        json.dumps({"metrics": metrics, "best_classifier": metadata["best_classifier"]}, indent=2),
        encoding="utf-8",
    )

    logger.info("Predicciones guardadas en %s", args.output_csv)
    logger.info("Metricas guardadas en %s", args.output_metrics)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
