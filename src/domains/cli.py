"""CLI comun para ejecutar pipelines por dominio."""

from __future__ import annotations

import argparse
import logging

from src.domains.registry import get_domain_pipeline, list_domains

logger = logging.getLogger(__name__)


def build_parser(default_stage: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta pipelines por dominio de negocio")
    parser.add_argument("--domain", default="industrial", help="Dominio a ejecutar: industrial, restaurant o clinic")
    parser.add_argument(
        "--stage",
        default=default_stage or "full",
        choices=["generate", "etl", "train", "infer", "report", "full"],
        help="Stage del pipeline a ejecutar",
    )
    parser.add_argument("--employees", type=int, default=None, help="Numero de empleados para stage generate")
    parser.add_argument("--days", type=int, default=None, help="Numero de dias para stage generate")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria para stage generate")
    parser.add_argument(
        "--employees-path",
        default=None,
        help="Ruta alternativa de empleados para ETL, si aplica al dominio",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Ruta de salida para predicciones de inferencia, si aplica al dominio",
    )
    parser.add_argument(
        "--output-metrics",
        default=None,
        help="Ruta de salida para metricas de inferencia, si aplica al dominio",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="Lista dominios disponibles y termina",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    if args.list_domains:
        for domain in list_domains():
            logger.info("%s: %s", domain.name, domain.description)
        return

    try:
        pipeline = get_domain_pipeline(args.domain)
        logger.info("Dominio seleccionado: %s", pipeline.name)
        logger.info("Descripcion: %s", pipeline.description)

        if args.stage == "full":
            pipeline.run_full(args)
        else:
            pipeline.run_stage(args.stage, args)
    except (NotImplementedError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
