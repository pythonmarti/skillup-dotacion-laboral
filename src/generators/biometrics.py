"""Generador de series temporales de datos biométricos diarios."""

import numpy as np
import pandas as pd

from config.settings import PHYSIO_RANGES, PLANT_AREAS


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

    Incluye autocorrelación temporal y efectos contextuales de turno/área.

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

    # Indexar work_records para lookup rápido
    wr_idx = work_records_df.set_index(["employee_id", "date"])

    # Construir lookup de área info por empleado
    emp_area = employees_df.set_index("employee_id")["plant_area"].to_dict()

    autocorr = 0.7
    records = []

    # Almacenar valores previos por empleado para autocorrelación
    prev = {}

    all_dates = sorted(work_records_df["date"].unique())[:days]

    for date in all_dates:
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
            target_sleep = bl["sleep_base"] + sleep_adj + rng.normal(0, 0.8)
            if prev_vals:
                sleep_dur = autocorr * prev_vals["sleep_dur"] + (1 - autocorr) * target_sleep
            else:
                sleep_dur = target_sleep
            sleep_dur = _clip_physio(sleep_dur, "sleep_duration_hours")

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
            })

    return pd.DataFrame(records)
