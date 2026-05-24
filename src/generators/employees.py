"""Generador de perfiles demográficos de empleados."""

import logging
import numpy as np
import pandas as pd
from faker import Faker

from config.settings import PLANT_AREAS, POSITIONS

logger = logging.getLogger(__name__)


def generate_employees(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Genera n empleados con perfiles demográficos realistas.

    Parameters
    ----------
    n : int
        Número de empleados a generar.
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Generando perfiles de %d empleados (seed=%d)...", n, seed)
    rng = np.random.default_rng(seed)
    fake = Faker("es_MX")
    Faker.seed(seed)

    areas = list(PLANT_AREAS.keys())
    # Distribución de empleados por área (producción pesada > oficinas)
    area_weights = np.array([0.25, 0.25, 0.15, 0.20, 0.15])
    area_weights = area_weights / area_weights.sum()

    records = []
    for i in range(n):
        emp_id = f"EMP_{i + 1:03d}"

        # Género
        gender = "M" if rng.random() < 0.75 else "F"
        name = fake.name_male() if gender == "M" else fake.name_female()

        # Edad: normal(38, 8), clip 22-62
        age = int(np.clip(rng.normal(38, 8), 22, 62))

        # BMI: normal(27, 4), clip 18-45
        bmi = round(float(np.clip(rng.normal(27, 4), 18, 45)), 1)

        # Educación: 1=básica, 2=media, 3=superior, 4=posgrado
        education_level = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.40, 0.10]))

        # Área y posición
        area = rng.choice(areas, p=area_weights)
        position = rng.choice(POSITIONS[area])

        # Antigüedad correlacionada con edad
        max_seniority = age - 20
        if max_seniority < 1:
            max_seniority = 1
        seniority_years = int(np.clip(rng.exponential(scale=max_seniority * 0.4), 0, max_seniority))

        # Patrón de turno
        shift_pattern = "fijo" if area == "oficinas" else rng.choice(["rotativo", "fijo"], p=[0.7, 0.3])

        # Distancia al trabajo: log-normal(2.5, 0.8)
        distance_to_work_km = round(float(np.clip(rng.lognormal(2.5, 0.8), 1, 120)), 1)

        # Hijos: Poisson(1.2)
        children = int(rng.poisson(1.2))

        # Hábitos
        social_drinker = bool(rng.random() < 0.30)
        smoker = bool(rng.random() < 0.20)

        # Fecha de contratación derivada de antigüedad
        hire_date = pd.Timestamp("2026-03-06") - pd.Timedelta(days=int(seniority_years * 365.25))
        hire_date = hire_date.strftime("%Y-%m-%d")

        records.append({
            "employee_id": emp_id,
            "name": name,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "education_level": education_level,
            "plant_area": area,
            "position": position,
            "seniority_years": seniority_years,
            "shift_pattern": shift_pattern,
            "distance_to_work_km": distance_to_work_km,
            "children": children,
            "social_drinker": social_drinker,
            "smoker": smoker,
            "hire_date": hire_date,
        })

    df = pd.DataFrame(records)
    logger.info("Empleados generados: %d filas", len(df))
    return df
