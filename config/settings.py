"""Configuración central del proyecto SkillUp."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "skillup.db"

# --- Simulation parameters ---
NUM_EMPLOYEES = 200
DAYS_TO_SIMULATE = 180
RANDOM_SEED = 42

# --- Physiological ranges (valid measurement bounds) ---
PHYSIO_RANGES = {
    "hr_mean_bpm": (30, 220),
    "hr_min_bpm": (30, 220),
    "hr_max_bpm": (30, 220),
    "hr_std_bpm": (0, 60),
    "hrv_rmssd_ms": (5, 200),
    "spo2_mean_pct": (70, 100),
    "spo2_min_pct": (70, 100),
    "skin_temp_mean_c": (28, 42),
    "sleep_duration_hours": (0, 24),
    "sleep_efficiency_pct": (0, 100),
    "deep_sleep_pct": (0, 100),
    "stress_score": (0, 100),
    "steps": (0, 60000),
}

# --- Plant areas with risk profiles ---
PLANT_AREAS = {
    "destilacion": {"risk_level": "alto", "heat_exposure": True, "base_risk": 0.7},
    "cracking": {"risk_level": "alto", "heat_exposure": True, "base_risk": 0.8},
    "almacenamiento": {"risk_level": "medio", "heat_exposure": False, "base_risk": 0.4},
    "mantenimiento": {"risk_level": "medio", "heat_exposure": True, "base_risk": 0.5},
    "oficinas": {"risk_level": "bajo", "heat_exposure": False, "base_risk": 0.2},
}

# --- Shifts ---
SHIFTS = {
    "diurno": {"start": 6, "end": 14, "fatigue_factor": 1.0},
    "vespertino": {"start": 14, "end": 22, "fatigue_factor": 1.1},
    "nocturno": {"start": 22, "end": 6, "fatigue_factor": 1.3},
}

# --- Staffing requirements per area and shift ---
REQUIRED_STAFF = {
    "destilacion": {"diurno": 10, "vespertino": 8, "nocturno": 6},
    "cracking": {"diurno": 12, "vespertino": 10, "nocturno": 8},
    "almacenamiento": {"diurno": 6, "vespertino": 4, "nocturno": 2},
    "mantenimiento": {"diurno": 8, "vespertino": 6, "nocturno": 3},
    "oficinas": {"diurno": 5, "vespertino": 1, "nocturno": 0},
}

# --- Positions per area ---
POSITIONS = {
    "destilacion": ["operador_campo", "operador_panel", "supervisor"],
    "cracking": ["operador_campo", "operador_panel", "supervisor"],
    "almacenamiento": ["operador_logistica", "operador_campo", "supervisor"],
    "mantenimiento": ["tecnico_mecanico", "tecnico_electrico", "tecnico_instrumentista", "supervisor"],
    "oficinas": ["analista", "ingeniero_proceso", "administrativo", "coordinador"],
}

# --- Model configuration ---
MODEL_CONFIG = {
    "n_iter_search": 10,
    "cv_splits": 3,
    "scoring": "f1",
    "test_ratio": 0.2,
    "balancing_strategy": "smote_enn",
    "models_to_train": [
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "Stacking",
    ],
    "calibration_method": "sigmoid",
}

# --- Absence reasons (ICD-10 inspired categories) ---
ABSENCE_REASONS = {
    "musculoesqueletico": {"code": "M00-M99", "weight": 0.25, "avg_days": 5},
    "respiratorio": {"code": "J00-J99", "weight": 0.20, "avg_days": 3},
    "digestivo": {"code": "K00-K93", "weight": 0.10, "avg_days": 2},
    "mental_conductual": {"code": "F00-F99", "weight": 0.12, "avg_days": 7},
    "traumatismo": {"code": "S00-T98", "weight": 0.10, "avg_days": 8},
    "cardiovascular": {"code": "I00-I99", "weight": 0.08, "avg_days": 6},
    "genitourinario": {"code": "N00-N99", "weight": 0.07, "avg_days": 3},
    "otros": {"code": "R00-R99", "weight": 0.08, "avg_days": 2},
}
