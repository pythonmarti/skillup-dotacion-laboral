"""Generador de eventos de ausentismo causalmente ligados a biometrics y work_records."""

import numpy as np
import pandas as pd

from config.settings import ABSENCE_REASONS, PLANT_AREAS


def generate_absenteeism(
    employees_df: pd.DataFrame,
    biometrics_df: pd.DataFrame,
    work_records_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera eventos de ausencia causalmente ligados a datos biométricos y de trabajo.

    La probabilidad de ausencia se calcula con modificadores multiplicativos
    basados en condiciones del empleado, turno, biométricos y contexto laboral.

    Parameters
    ----------
    employees_df : pd.DataFrame
        DataFrame de empleados.
    biometrics_df : pd.DataFrame
        DataFrame de biométricos diarios.
    work_records_df : pd.DataFrame
        DataFrame de registros de trabajo.
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame
        Solo filas donde hubo ausencia (is_absent=1).
    """
    rng = np.random.default_rng(seed)

    # Lookup de empleados
    emp_lookup = employees_df.set_index("employee_id")

    # Indexar biometrics y work_records
    bio_idx = biometrics_df.set_index(["employee_id", "date"])
    wr_idx = work_records_df.set_index(["employee_id", "date"])

    # Pesos y razones de ausencia
    reasons = list(ABSENCE_REASONS.keys())
    reason_weights = np.array([ABSENCE_REASONS[r]["weight"] for r in reasons])
    reason_weights = reason_weights / reason_weights.sum()

    # Tracking de turnos nocturnos consecutivos por empleado
    night_streak = {}

    base_prob = 0.035
    records = []

    all_dates = sorted(work_records_df["date"].unique())

    for date in all_dates:
        date_ts = pd.Timestamp(date)
        is_monday_or_friday = date_ts.dayofweek in (0, 4)

        for emp_id in emp_lookup.index:
            # Obtener datos del empleado
            emp = emp_lookup.loc[emp_id]
            bmi = emp["bmi"]
            distance = emp["distance_to_work_km"]
            children = emp["children"]

            # Obtener work record
            try:
                wr = wr_idx.loc[(emp_id, date)]
            except KeyError:
                continue

            # No generar ausencia en días de descanso
            if wr["is_rest_day"]:
                night_streak[emp_id] = 0
                continue

            shift = wr["shift"]
            area = wr["area_assigned"]
            workload = wr["workload_score"]
            area_info = PLANT_AREAS.get(area, {})

            # Tracking de turno nocturno
            if shift == "nocturno":
                night_streak[emp_id] = night_streak.get(emp_id, 0) + 1
            else:
                night_streak[emp_id] = 0

            # Obtener biometrics
            try:
                bio = bio_idx.loc[(emp_id, date)]
                stress = bio["stress_score"]
                hrv = bio["hrv_rmssd_ms"]
                sleep = bio["sleep_duration_hours"]
            except KeyError:
                stress = 35
                hrv = 40
                sleep = 7

            # --- Determinar si hay ausencia basado en estado de salud ---
            # Si el biométrico indica que está enfermo (_is_sick == 1), es una ausencia médica.
            # También dejamos una probabilidad muy baja de ausencia casual no médica.
            is_sick = 0
            absence_reason = ""
            try:
                is_sick = int(bio.get("_is_sick", 0))
                absence_reason = bio.get("_absence_reason", "")
            except Exception:
                pass

            is_absent = False
            reason = ""

            if is_sick == 1:
                is_absent = True
                reason = absence_reason if absence_reason else "otros"
            elif rng.random() < 0.003:  # Ausencia casual no médica
                is_absent = True
                reason = "otros"

            if is_absent:
                # Horas de ausencia: se pierde el turno completo (ej. uniform 7.5 a 8.5 horas)
                absence_hours = round(float(rng.uniform(7.5, 8.5)), 1)
                records.append({
                    "employee_id": emp_id,
                    "date": date,
                    "absence_reason": reason,
                    "absence_hours": absence_hours,
                    "is_absent": 1,
                })

    return pd.DataFrame(records)
