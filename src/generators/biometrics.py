"""Generador de series temporales de datos biométricos diarios."""

import numpy as np
import pandas as pd

from config.settings import PHYSIO_RANGES, PLANT_AREAS, ABSENCE_REASONS


def _generate_health_events(employees_df: pd.DataFrame, days: int, seed: int = 42) -> dict:
    """Pre-determina eventos de enfermedad e incubación para cada empleado.
    
    Retorna un diccionario emp_id -> (states, reason_arr)
    Donde states es un array de tamaño 'days' con valores:
      - 0: normal
      - 1: incubación (2 días antes)
      - 2: incubación (1 día antes)
      - 3: enfermo/ausente
    """
    rng = np.random.default_rng(seed)
    reasons = list(ABSENCE_REASONS.keys())
    reason_weights = np.array([ABSENCE_REASONS[r]["weight"] for r in reasons])
    reason_weights = reason_weights / reason_weights.sum()
    
    events = {}
    for emp_id in employees_df["employee_id"]:
        states = np.zeros(days, dtype=int)
        reason_arr = [""] * days
        
        day = 0
        while day < days:
            # Probabilidad de enfermarse en un día cualquiera: ~0.012
            if rng.random() < 0.012:
                reason = rng.choice(reasons, p=reason_weights)
                avg_days = ABSENCE_REASONS[reason]["avg_days"]
                # Duración de la enfermedad
                dur = int(rng.poisson(avg_days - 1) + 1)
                
                # Definir fase de incubación (hasta 2 días antes de enfermarse)
                for inc_idx, inc_day in enumerate([day - 2, day - 1]):
                    if 0 <= inc_day < days:
                        if states[inc_day] != 3: # Solo si no está enfermo ya
                            states[inc_day] = inc_idx + 1 # 1 o 2
                
                # Definir fase de enfermedad
                for sick_day in range(day, min(day + dur, days)):
                    states[sick_day] = 3
                    reason_arr[sick_day] = reason
                
                # Período refractario de 5 días antes de poder enfermarse de nuevo
                day += dur + 5
            else:
                day += 1
                
        events[emp_id] = (states, reason_arr)
    return events


def _build_baselines(employees_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Crea baselines individuales por empleado basados en edad, BMI, fumador."""
    baselines = {}
    for _, emp in employees_df.iterrows():
        age = emp["age"]
        bmi = emp["bmi"]
        smoker = emp["smoker"]

        # HR base: más jóvenes tienen HR más alta en reposo relativo
        hr_base = 72 - (age - 30) * 0.15 + rng.normal(0, 3)
        if smoker:
            hr_base += 3
        if bmi > 30:
            hr_base += 4

        # HRV base: decrece con edad y BMI alto
        hrv_base = 45 - (age - 30) * 0.4 - max(0, bmi - 25) * 0.5 + rng.normal(0, 5)
        hrv_base = max(10, hrv_base)

        # SpO2 base
        spo2_base = 97.5 + rng.normal(0, 0.5)
        if smoker:
            spo2_base -= 1.0

        # Temperatura piel
        skin_temp_base = 33.5 + rng.normal(0, 0.3)

        # Sueño base
        sleep_base = 7.0 + rng.normal(0, 0.5)

        # Estrés base
        stress_base = 35 + rng.normal(0, 5)

        # Steps base
        steps_base = 8000 + rng.normal(0, 1500)

        baselines[emp["employee_id"]] = {
            "hr_base": hr_base,
            "hrv_base": hrv_base,
            "spo2_base": spo2_base,
            "skin_temp_base": skin_temp_base,
            "sleep_base": sleep_base,
            "stress_base": stress_base,
            "steps_base": steps_base,
        }
    return baselines


def _clip_physio(value: float, col: str) -> float:
    lo, hi = PHYSIO_RANGES[col]
    return float(np.clip(value, lo, hi))


def generate_biometrics(
    employees_df: pd.DataFrame,
    work_records_df: pd.DataFrame,
    days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera series temporales de datos biométricos diarios.

    Incluye autocorrelación temporal y efectos contextuales de turno/área,
    así como fases de incubación y enfermedad reales.

    Parameters
    ----------
    employees_df : pd.DataFrame
        DataFrame de empleados.
    work_records_df : pd.DataFrame
        DataFrame de registros de trabajo.
    days : int
        Número de días a simular.
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    baselines = _build_baselines(employees_df, rng)

    # Pre-generar eventos de enfermedad
    health_events = _generate_health_events(employees_df, days, seed)

    # Indexar work_records para lookup rápido
    wr_idx = work_records_df.set_index(["employee_id", "date"])

    # Construir lookup de área info por empleado
    emp_area = employees_df.set_index("employee_id")["plant_area"].to_dict()

    autocorr = 0.7
    records = []

    # Almacenar valores previos por empleado para autocorrelación
    prev = {}

    all_dates = sorted(work_records_df["date"].unique())[:days]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for date in all_dates:
        day_idx = date_to_idx[date]
        for emp_id, bl in baselines.items():
            area = emp_area[emp_id]
            area_info = PLANT_AREAS[area]

            # Obtener info de trabajo del día
            try:
                wr = wr_idx.loc[(emp_id, date)]
                shift = wr["shift"]
                is_rest = wr["is_rest_day"]
            except KeyError:
                shift = "diurno"
                is_rest = False

            # --- Efectos contextuales ---
            hr_adj = 0.0
            spo2_adj = 0.0
            temp_adj = 0.0
            stress_adj = 0.0

            # Turno nocturno
            if shift == "nocturno":
                hr_adj -= 5
                spo2_adj -= 0.3
                stress_adj += 10

            # Área con calor
            if area_info.get("heat_exposure"):
                hr_adj += 8
                temp_adj += 0.4

            # --- Ajustes por estado de salud ---
            states, reason_arr = health_events[emp_id]
            state = states[day_idx]
            reason = reason_arr[day_idx]

            sleep_hours_adj = 0.0
            steps_override = None

            if state in (1, 2):  # Incubación (1-2 días antes de ausentarse)
                hr_adj += rng.uniform(4.0, 8.0)
                spo2_adj -= rng.uniform(0.2, 0.5)
                temp_adj += rng.uniform(0.1, 0.3)
                stress_adj += rng.uniform(10.0, 20.0)
                sleep_hours_adj -= rng.uniform(1.0, 2.0)
            elif state == 3:  # Enfermedad
                hr_adj += rng.uniform(10.0, 18.0)
                spo2_adj -= rng.uniform(0.6, 1.5)
                temp_adj += rng.uniform(0.3, 0.6)
                stress_adj += rng.uniform(25.0, 40.0)
                sleep_hours_adj -= rng.uniform(2.0, 3.5)
                steps_override = int(rng.uniform(500, 2000))

            # --- Generar valores con autocorrelación ---
            prev_vals = prev.get(emp_id, None)

            # HR mean
            target_hr = bl["hr_base"] + hr_adj + rng.normal(0, 4)
            if prev_vals:
                hr_mean = autocorr * prev_vals["hr_mean"] + (1 - autocorr) * target_hr
            else:
                hr_mean = target_hr
            hr_mean = _clip_physio(hr_mean, "hr_mean_bpm")

            hr_std = max(2, rng.normal(8, 2))
            hr_min = _clip_physio(hr_mean - rng.uniform(15, 30), "hr_min_bpm")
            hr_max = _clip_physio(hr_mean + rng.uniform(20, 50), "hr_max_bpm")

            # HRV
            target_hrv = bl["hrv_base"] + rng.normal(0, 5)
            if shift == "nocturno":
                target_hrv -= 5
            if state in (1, 2):
                target_hrv -= rng.uniform(8.0, 15.0)
            elif state == 3:
                target_hrv -= rng.uniform(15.0, 25.0)

            if prev_vals:
                hrv = autocorr * prev_vals["hrv"] + (1 - autocorr) * target_hrv
            else:
                hrv = target_hrv
            hrv = _clip_physio(hrv, "hrv_rmssd_ms")

            # SpO2
            target_spo2 = bl["spo2_base"] + spo2_adj + rng.normal(0, 0.3)
            if prev_vals:
                spo2_mean = autocorr * prev_vals["spo2_mean"] + (1 - autocorr) * target_spo2
            else:
                spo2_mean = target_spo2
            spo2_mean = _clip_physio(spo2_mean, "spo2_mean_pct")
            spo2_min = _clip_physio(spo2_mean - rng.uniform(1, 4), "spo2_min_pct")

            # Skin temp
            target_temp = bl["skin_temp_base"] + temp_adj + rng.normal(0, 0.2)
            if prev_vals:
                skin_temp = autocorr * prev_vals["skin_temp"] + (1 - autocorr) * target_temp
            else:
                skin_temp = target_temp
            skin_temp = _clip_physio(skin_temp, "skin_temp_mean_c")

            # Sueño
            sleep_adj = -1.0 if shift == "nocturno" else 0.0
            sleep_adj += 1.0 if is_rest else 0.0
            sleep_adj += sleep_hours_adj
            target_sleep = bl["sleep_base"] + sleep_adj + rng.normal(0, 0.8)
            if prev_vals:
                sleep_dur = autocorr * prev_vals["sleep_dur"] + (1 - autocorr) * target_sleep
            else:
                sleep_dur = target_sleep
            sleep_dur = _clip_physio(sleep_dur, "sleep_duration_hours")

            if state in (1, 2):
                sleep_eff = float(np.clip(rng.normal(70, 8), 35, 90))
                deep_sleep = float(np.clip(rng.normal(12, 3), 3, 20))
            elif state == 3:
                sleep_eff = float(np.clip(rng.normal(60, 10), 30, 80))
                deep_sleep = float(np.clip(rng.normal(8, 3), 0, 15))
            else:
                sleep_eff = float(np.clip(rng.normal(82, 8), 40, 100))
                deep_sleep = float(np.clip(rng.normal(20, 5), 5, 40))

            # Estrés
            target_stress = bl["stress_base"] + stress_adj + rng.normal(0, 8)
            if prev_vals:
                stress = autocorr * prev_vals["stress"] + (1 - autocorr) * target_stress
            else:
                stress = target_stress
            stress = _clip_physio(stress, "stress_score")

            # Steps
            if steps_override is not None:
                steps = steps_override
            else:
                step_mult = 0.4 if is_rest else 1.0
                target_steps = bl["steps_base"] * step_mult + rng.normal(0, 1000)
                if prev_vals:
                    steps = autocorr * prev_vals["steps"] + (1 - autocorr) * target_steps
                else:
                    steps = target_steps
            steps = _clip_physio(steps, "steps")

            # Data quality: 5-10% con score bajo
            if rng.random() < 0.075:
                data_quality = round(float(rng.uniform(0.1, 0.49)), 2)
            else:
                data_quality = round(float(rng.uniform(0.7, 1.0)), 2)

            # Guardar para autocorrelación
            prev[emp_id] = {
                "hr_mean": hr_mean,
                "hrv": hrv,
                "spo2_mean": spo2_mean,
                "skin_temp": skin_temp,
                "sleep_dur": sleep_dur,
                "stress": stress,
                "steps": steps,
            }

            records.append({
                "employee_id": emp_id,
                "date": date,
                "hr_mean_bpm": round(hr_mean, 1),
                "hr_min_bpm": round(hr_min, 1),
                "hr_max_bpm": round(hr_max, 1),
                "hr_std_bpm": round(hr_std, 1),
                "hrv_rmssd_ms": round(hrv, 1),
                "spo2_mean_pct": round(spo2_mean, 1),
                "spo2_min_pct": round(spo2_min, 1),
                "skin_temp_mean_c": round(skin_temp, 2),
                "sleep_duration_hours": round(sleep_dur, 1),
                "sleep_efficiency_pct": round(sleep_eff, 1),
                "deep_sleep_pct": round(deep_sleep, 1),
                "stress_score": round(stress, 1),
                "steps": int(steps),
                "data_quality_score": data_quality,
                "_is_sick": 1 if state == 3 else 0,
                "_absence_reason": reason,
            })

    return pd.DataFrame(records)
