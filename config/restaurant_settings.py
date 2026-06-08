"""Configuracion del dominio restaurant casual dining."""

from pathlib import Path

from config.settings import DATA_DIR

RESTAURANT_DOMAIN = "restaurant"
RESTAURANT_DATA_DIR = DATA_DIR / RESTAURANT_DOMAIN
RESTAURANT_RAW_DIR = RESTAURANT_DATA_DIR / "raw"
RESTAURANT_PROCESSED_DIR = RESTAURANT_DATA_DIR / "processed"
RESTAURANT_DB_PATH = RESTAURANT_DATA_DIR / "restaurant_skillup.db"

RESTAURANT_NUM_EMPLOYEES = 72
RESTAURANT_DAYS_TO_SIMULATE = 180
RESTAURANT_RANDOM_SEED = 42
RESTAURANT_START_DATE = "2025-07-01"

SERVICE_PERIODS = {
    "11_13": {"label": "11:00-13:00", "demand_factor": 0.72, "fatigue_factor": 0.9},
    "13_15": {"label": "13:00-15:00", "demand_factor": 1.45, "fatigue_factor": 1.08},
    "19_21": {"label": "19:00-21:00", "demand_factor": 1.35, "fatigue_factor": 1.2},
    "21_23": {"label": "21:00-23:00", "demand_factor": 0.88, "fatigue_factor": 1.25},
}

RESTAURANT_ROLES = {
    "garzon": {"weight": 0.28, "criticality": 1.0, "customer_facing": True},
    "host": {"weight": 0.08, "criticality": 0.7, "customer_facing": True},
    "cajero": {"weight": 0.08, "criticality": 0.75, "customer_facing": True},
    "cocinero_linea": {"weight": 0.18, "criticality": 1.0, "customer_facing": False},
    "ayudante_cocina": {"weight": 0.12, "criticality": 0.85, "customer_facing": False},
    "runner": {"weight": 0.12, "criticality": 0.8, "customer_facing": True},
    "copero": {"weight": 0.08, "criticality": 0.65, "customer_facing": False},
    "jefe_turno": {"weight": 0.06, "criticality": 0.95, "customer_facing": True},
}

CRITICAL_ROLE_TARGETS = ["garzon", "cocinero_linea", "jefe_turno"]

ROLE_BACKUPS = {
    "garzon": ["runner", "host"],
    "host": ["garzon", "cajero"],
    "cajero": ["host"],
    "cocinero_linea": ["ayudante_cocina"],
    "ayudante_cocina": ["cocinero_linea"],
    "runner": ["garzon"],
    "copero": ["runner"],
    "jefe_turno": ["garzon", "cajero"],
}

CHILE_HOLIDAYS_2025 = {
    "2025-07-16": "Virgen del Carmen",
    "2025-08-15": "Asuncion de la Virgen",
    "2025-09-18": "Independencia Nacional",
    "2025-09-19": "Glorias del Ejercito",
    "2025-10-13": "Encuentro de Dos Mundos",
    "2025-10-31": "Iglesias Evangelicas y Protestantes",
    "2025-11-01": "Todos los Santos",
    "2025-12-08": "Inmaculada Concepcion",
    "2025-12-25": "Navidad",
}

RESTAURANT_MODEL_CONFIG = {
    "test_ratio": 0.2,
    "role_models": CRITICAL_ROLE_TARGETS,
    "alert_probability_threshold": 0.55,
    "high_risk_threshold": 0.75,
}
