"""Configuracion del dominio clinic ambulatorio."""

from config.settings import DATA_DIR

CLINIC_DOMAIN = "clinic"
CLINIC_DATA_DIR = DATA_DIR / CLINIC_DOMAIN
CLINIC_RAW_DIR = CLINIC_DATA_DIR / "raw"
CLINIC_PROCESSED_DIR = CLINIC_DATA_DIR / "processed"
CLINIC_DB_PATH = CLINIC_DATA_DIR / "clinic_skillup.db"

CLINIC_NUM_EMPLOYEES = 96
CLINIC_DAYS_TO_SIMULATE = 180
CLINIC_RANDOM_SEED = 42
CLINIC_START_DATE = "2025-05-01"

CLINIC_SHIFTS = {
    "morning": {"label": "07:00-15:00", "fatigue_factor": 0.95},
    "evening": {"label": "15:00-22:00", "fatigue_factor": 1.08},
}

CLINICAL_UNITS = {
    "consulta_general": {"open_shifts": ["morning", "evening"], "acuity_base": 0.44},
    "especialidades": {"open_shifts": ["morning", "evening"], "acuity_base": 0.52},
    "procedimientos_ambulatorios": {"open_shifts": ["morning", "evening"], "acuity_base": 0.71},
    "imagenologia": {"open_shifts": ["morning", "evening"], "acuity_base": 0.58},
    "toma_muestras": {"open_shifts": ["morning", "evening"], "acuity_base": 0.33},
}

CLINIC_ROLES = {
    "medico": {"weight": 0.18, "criticality": 1.0},
    "enfermera": {"weight": 0.21, "criticality": 0.96},
    "tens": {"weight": 0.26, "criticality": 0.9},
    "admision": {"weight": 0.11, "criticality": 0.55},
    "tecnologo_medico": {"weight": 0.14, "criticality": 0.78},
    "coordinador_clinico": {"weight": 0.10, "criticality": 0.82},
}

CLINIC_CRITICAL_ROLE_TARGETS = ["medico", "enfermera", "tens"]

CLINIC_HOLIDAYS_2025 = {
    "2025-05-21": "Glorias Navales",
    "2025-06-20": "Dia Nacional de los Pueblos Indigenas",
    "2025-06-29": "San Pedro y San Pablo",
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

CLINIC_MODEL_CONFIG = {
    "test_ratio": 0.2,
    "role_models": CLINIC_CRITICAL_ROLE_TARGETS,
    "alert_probability_threshold": 0.58,
    "high_risk_threshold": 0.74,
}
