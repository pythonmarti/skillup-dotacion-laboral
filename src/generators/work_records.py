"""Generador de registros de trabajo diarios."""

import numpy as np
import pandas as pd

from config.settings import PLANT_AREAS, SHIFTS

# Días festivos oficiales mexicanos (mes, día)
_MEXICAN_HOLIDAYS = [
    (1, 1), (2, 5), (3, 21), (5, 1), (9, 16), (11, 20), (12, 25),
]


def _is_holiday(date: pd.Timestamp) -> bool:
    return (date.month, date.day) in _MEXICAN_HOLIDAYS


def generate_work_records(
    employees_df: pd.DataFrame,
    days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera registros de trabajo diarios para cada empleado.

    Parameters
    ----------
    employees_df : pd.DataFrame
        DataFrame de empleados generado por generate_employees.
    days : int
        Número de días a simular.
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    shift_names = list(SHIFTS.keys())
    start_date = pd.Timestamp("2025-09-08")  # Lunes de inicio
    dates = pd.date_range(start_date, periods=days, freq="D")

    records = []
    for _, emp in employees_df.iterrows():
        emp_id = emp["employee_id"]
        area = emp["plant_area"]
        shift_pattern = emp["shift_pattern"]
        area_info = PLANT_AREAS[area]

        # Turno inicial aleatorio para rotativos
        shift_idx = rng.integers(0, 3) if shift_pattern == "rotativo" else 0
        if area == "oficinas":
            shift_idx = 0  # Siempre diurno

        # Día de descanso base (2 de cada 7)
        rest_day_offset = rng.integers(0, 7)
        consecutive_work = 0

        for day_num, date in enumerate(dates):
            # Rotación de turno cada 7 días
            if shift_pattern == "rotativo" and day_num > 0 and day_num % 7 == 0:
                shift_idx = (shift_idx + 1) % 3

            shift = shift_names[shift_idx] if shift_pattern == "rotativo" else (
                "diurno" if area == "oficinas" else shift_names[rng.integers(0, 3)]
            )

            # Día de descanso: 2 de cada 7
            day_in_cycle = (day_num + rest_day_offset) % 7
            is_rest_day = day_in_cycle >= 5

            is_holiday = _is_holiday(date)

            if is_rest_day:
                consecutive_work = 0
            else:
                consecutive_work += 1

            # Workload score: normal(55, 15), ajustado por riesgo de área
            base_workload = rng.normal(55, 15)
            risk_adj = area_info["base_risk"] * 20  # áreas de alto riesgo +16
            workload_score = float(np.clip(base_workload + risk_adj, 0, 100))

            # Horas trabajadas (solo si no es descanso)
            if is_rest_day and not is_holiday:
                hours_worked = 0.0
                overtime_hours = 0.0
            else:
                hours_worked = round(float(rng.uniform(7, 9)), 1)
                # Overtime más frecuente en áreas de alto riesgo
                ot_prob = 0.3 if area_info["risk_level"] == "alto" else 0.15
                overtime_hours = round(float(rng.exponential(1.0)), 1) if rng.random() < ot_prob else 0.0
                overtime_hours = min(overtime_hours, 4.0)

            records.append({
                "employee_id": emp_id,
                "date": date.strftime("%Y-%m-%d"),
                "shift": shift,
                "area_assigned": area,
                "workload_score": round(workload_score, 1),
                "hours_worked": hours_worked,
                "overtime_hours": overtime_hours,
                "consecutive_work_days": consecutive_work,
                "is_holiday": is_holiday,
                "is_rest_day": is_rest_day,
            })

    return pd.DataFrame(records)
