"""Registro central de dominios disponibles."""

from __future__ import annotations

from src.domains.clinic import PIPELINE as CLINIC_PIPELINE
from src.domains.base import DomainPipeline
from src.domains.industrial import PIPELINE as INDUSTRIAL_PIPELINE
from src.domains.restaurant import PIPELINE as RESTAURANT_PIPELINE

DOMAIN_REGISTRY: dict[str, DomainPipeline] = {
    CLINIC_PIPELINE.name: CLINIC_PIPELINE,
    INDUSTRIAL_PIPELINE.name: INDUSTRIAL_PIPELINE,
    RESTAURANT_PIPELINE.name: RESTAURANT_PIPELINE,
}


def get_domain_pipeline(domain: str) -> DomainPipeline:
    try:
        return DOMAIN_REGISTRY[domain]
    except KeyError as exc:
        available = ", ".join(sorted(DOMAIN_REGISTRY))
        raise ValueError(f"Dominio desconocido '{domain}'. Disponibles: {available}") from exc


def list_domains() -> list[DomainPipeline]:
    return [DOMAIN_REGISTRY[key] for key in sorted(DOMAIN_REGISTRY)]
