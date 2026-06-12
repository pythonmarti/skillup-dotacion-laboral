"""Generacion sintetica del dominio clinic ambulatorio."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from config.clinic_settings import (
    CLINICAL_UNITS,
    CLINIC_HOLIDAYS_2025,
    CLINIC_RAW_DIR,
    CLINIC_ROLES,
    CLINIC_SHIFTS,
)

logger = logging.getLogger(__name__)


ROLE_UNIT_DISTRIBUTION = {
    "medico": {
        "consulta_general": 0.34,
        "especialidades": 0.34,
        "procedimientos_ambulatorios": 0.22,
        "imagenologia": 0.10,
    },
    "enfermera": {
        "consulta_general": 0.18,
        "especialidades": 0.16,
        "procedimientos_ambulatorios": 0.30,
        "imagenologia": 0.12,
        "toma_muestras": 0.24,
    },
    "tens": {
        "consulta_general": 0.20,
        "especialidades": 0.14,
        "procedimientos_ambulatorios": 0.19,
        "imagenologia": 0.16,
        "toma_muestras": 0.31,
    },
    "admision": {
        "consulta_general": 0.34,
        "especialidades": 0.30,
        "procedimientos_ambulatorios": 0.12,
        "imagenologia": 0.12,
        "toma_muestras": 0.12,
    },
    "tecnologo_medico": {
        "imagenologia": 0.76,
        "toma_muestras": 0.18,
        "procedimientos_ambulatorios": 0.06,
    },
    "coordinador_clinico": {
        "consulta_general": 0.18,
        "especialidades": 0.24,
        "procedimientos_ambulatorios": 0.26,
        "imagenologia": 0.14,
        "toma_muestras": 0.18,
    },
}

SHIFT_PREFERENCE_WEIGHTS = {
    "medico": [0.46, 0.20, 0.34],
    "enfermera": [0.40, 0.24, 0.36],
    "tens": [0.38, 0.24, 0.38],
    "admision": [0.48, 0.22, 0.30],
    "tecnologo_medico": [0.44, 0.20, 0.36],
    "coordinador_clinico": [0.42, 0.20, 0.38],
}

UNIT_SHIFT_BASELINES = {
    "consulta_general": {
        "morning": {"forecast_volume": 72, "procedures": 3, "stations": 14, "acuity": 5, "backlog": 4, "wait": 24, "no_show": 7, "respiratory": 5},
        "evening": {"forecast_volume": 48, "procedures": 2, "stations": 10, "acuity": 4, "backlog": 3, "wait": 20, "no_show": 4, "respiratory": 3},
    },
    "especialidades": {
        "morning": {"forecast_volume": 56, "procedures": 4, "stations": 12, "acuity": 6, "backlog": 7, "wait": 31, "no_show": 6, "respiratory": 2},
        "evening": {"forecast_volume": 40, "procedures": 3, "stations": 9, "acuity": 5, "backlog": 6, "wait": 27, "no_show": 4, "respiratory": 2},
    },
    "procedimientos_ambulatorios": {
        "morning": {"forecast_volume": 24, "procedures": 10, "stations": 8, "acuity": 7, "backlog": 3, "wait": 22, "no_show": 2, "respiratory": 1},
        "evening": {"forecast_volume": 18, "procedures": 7, "stations": 6, "acuity": 6, "backlog": 2, "wait": 18, "no_show": 1, "respiratory": 1},
    },
    "imagenologia": {
        "morning": {"forecast_volume": 46, "procedures": 8, "stations": 7, "acuity": 4, "backlog": 14, "wait": 29, "no_show": 4, "respiratory": 1},
        "evening": {"forecast_volume": 32, "procedures": 5, "stations": 5, "acuity": 3, "backlog": 10, "wait": 24, "no_show": 3, "respiratory": 1},
    },
    "toma_muestras": {
        "morning": {"forecast_volume": 84, "procedures": 0, "stations": 10, "acuity": 2, "backlog": 9, "wait": 19, "no_show": 10, "respiratory": 6},
        "evening": {"forecast_volume": 34, "procedures": 0, "stations": 5, "acuity": 2, "backlog": 3, "wait": 14, "no_show": 3, "respiratory": 2},
    },
}


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "summer"
    if month in {3, 4, 5}:
        return "autumn"
    if month in {6, 7, 8}:
        return "winter"
    return "spring"


def generate_clinic_calendar(start_date: str, days: int) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=days, freq="D")
    holidays = {pd.Timestamp(key): value for key, value in CLINIC_HOLIDAYS_2025.items()}
    records = []

    for date in dates:
        is_holiday = date in holidays
        is_weekend = date.dayofweek >= 5
        respiratory_alert = 2 if date.month in {6, 7, 8} else 1
        if date.month in {7, 8}:
            respiratory_alert += 1
        vaccination_campaign_flag = bool(date.month in {4, 5} and date.day <= 20)
        is_payday_window = date.day in {29, 30, 1, 2, 14, 15}
        elective_block_pressure = 1.16 if date.dayofweek in {1, 2, 3} else 0.94
        if is_holiday or is_weekend:
            elective_block_pressure *= 0.64

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "is_holiday": bool(is_holiday),
            "holiday_name": holidays.get(date, ""),
            "is_weekend": bool(is_weekend),
            "season": _season_for_month(date.month),
            "respiratory_alert_level": round(float(respiratory_alert), 2),
            "vaccination_campaign_flag": bool(vaccination_campaign_flag),
            "is_payday_window": bool(is_payday_window),
            "elective_block_pressure": round(float(elective_block_pressure), 2),
        })

    return pd.DataFrame(records)


def _sample_primary_unit(role: str, rng: np.random.Generator) -> str:
    unit_weights = ROLE_UNIT_DISTRIBUTION[role]
    units = list(unit_weights.keys())
    weights = np.array([unit_weights[unit] for unit in units], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(units, p=weights))


def generate_clinic_employees(n: int, seed: int = 42) -> pd.DataFrame:
    logger.info("[clinic] Generando empleados: %d", n)
    rng = np.random.default_rng(seed)
    role_names = list(CLINIC_ROLES.keys())
    role_weights = np.array([CLINIC_ROLES[role]["weight"] for role in role_names], dtype=float)
    role_weights = role_weights / role_weights.sum()
    records = []

    for index in range(n):
        role = str(rng.choice(role_names, p=role_weights))
        primary_unit = _sample_primary_unit(role, rng)
        eligible_backup_units = [
            unit for unit in ROLE_UNIT_DISTRIBUTION[role]
            if unit != primary_unit
        ]
        backup_unit = str(rng.choice(eligible_backup_units)) if eligible_backup_units and rng.random() < 0.64 else primary_unit
        contract_type = str(rng.choice(["full_time", "part_time", "weekend_support"], p=[0.78, 0.18, 0.04]))
        shift_preference = str(rng.choice(["morning", "evening", "flexible"], p=SHIFT_PREFERENCE_WEIGHTS[role]))
        can_float = bool(backup_unit != primary_unit and rng.random() < 0.66)
        age = int(np.clip(rng.normal(34, 7), 22, 62))
        bmi = round(float(np.clip(rng.normal(26.1, 3.8), 18, 39)), 1)
        years_experience = int(np.clip(rng.gamma(2.0, 4.2), 0, max(age - 22, 1)))
        commute_km = round(float(np.clip(rng.lognormal(2.0, 0.5), 1, 45)), 1)
        dependents = int(np.clip(rng.poisson(1.0), 0, 5))
        chronic_condition_flag = bool(rng.random() < 0.14)
        triage_certified = bool(role in {"medico", "enfermera", "tens"} and rng.random() < 0.42)
        icu_certified = bool(role in {"medico", "enfermera", "tens"} and rng.random() < 0.14)
        radiation_certified = bool(role == "tecnologo_medico" or (role == "enfermera" and rng.random() < 0.12))
        overtime_tolerance = round(float(np.clip(rng.normal(0.56, 0.18), 0.1, 1.0)), 2)
        hire_date = pd.Timestamp("2025-12-31") - pd.Timedelta(days=int(years_experience * 365.25))

        records.append({
            "employee_id": f"CLN_{index + 1:03d}",
            "role": role,
            "primary_unit": primary_unit,
            "backup_unit": backup_unit,
            "age": age,
            "bmi": bmi,
            "years_experience": years_experience,
            "contract_type": contract_type,
            "shift_preference": shift_preference,
            "can_float": can_float,
            "triage_certified": triage_certified,
            "icu_certified": icu_certified,
            "radiation_certified": radiation_certified,
            "commute_km": commute_km,
            "dependents": dependents,
            "chronic_condition_flag": chronic_condition_flag,
            "overtime_tolerance": overtime_tolerance,
            "hire_date": hire_date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(records)


def generate_clinic_patient_flow(calendar_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    logger.info("[clinic] Generando flujo asistencial")
    rng = np.random.default_rng(seed)
    records = []
    weekday_multiplier = {0: 0.96, 1: 1.04, 2: 1.08, 3: 1.06, 4: 0.98, 5: 0.52, 6: 0.28}

    for _, cal in calendar_df.iterrows():
        date = pd.Timestamp(cal["date"])
        for clinical_unit, unit_meta in CLINICAL_UNITS.items():
            for shift in unit_meta["open_shifts"]:
                base = UNIT_SHIFT_BASELINES[clinical_unit][shift]
                multiplier = weekday_multiplier[date.dayofweek]

                if bool(cal["is_holiday"]):
                    multiplier *= 0.36
                if bool(cal["vaccination_campaign_flag"]) and clinical_unit in {"consulta_general", "toma_muestras"}:
                    multiplier *= 1.08
                if bool(cal["is_payday_window"]) and clinical_unit in {"especialidades", "imagenologia", "procedimientos_ambulatorios"}:
                    multiplier *= 1.06
                if clinical_unit == "procedimientos_ambulatorios":
                    multiplier *= float(cal["elective_block_pressure"])
                if clinical_unit == "consulta_general":
                    multiplier *= 1 + ((float(cal["respiratory_alert_level"]) - 1) * 0.06)
                if clinical_unit == "toma_muestras":
                    multiplier *= 1 + ((float(cal["respiratory_alert_level"]) - 1) * 0.10)

                forecast_volume = max(0, int(round(base["forecast_volume"] * multiplier + rng.normal(0, 4))))
                no_show_count = max(0, int(round(base["no_show"] * (1.08 if shift == "evening" else 1.0) + rng.normal(0, 1.5))))
                walk_in_buffer = int(round(forecast_volume * (0.12 if clinical_unit in {"consulta_general", "toma_muestras"} else 0.05)))
                actual_volume = max(0, int(round(forecast_volume - no_show_count + walk_in_buffer + rng.normal(0, 3))))
                scheduled_procedures = max(0, int(round(base["procedures"] * float(cal["elective_block_pressure"]) + rng.normal(0, 1.1))))
                active_care_stations = max(1, int(round(base["stations"] * rng.normal(1.0, 0.08) + actual_volume * 0.04)))
                high_acuity_cases = max(0, int(round(base["acuity"] * multiplier + scheduled_procedures * 0.25 + rng.normal(0, 1.4))))
                imaging_backlog_cases = max(0, int(round(base["backlog"] * rng.normal(1.0, 0.12) + (actual_volume * 0.18 if clinical_unit == "imagenologia" else 0))))
                respiratory_alert_cases = max(0, int(round(base["respiratory"] + float(cal["respiratory_alert_level"]) * rng.normal(1.0, 0.30))))
                respiratory_case_ratio = round(float(np.clip(rng.normal(0.10 + float(cal["respiratory_alert_level"]) * 0.05, 0.04), 0.02, 0.72)), 2)
                average_wait_minutes = max(
                    5,
                    int(round(
                        base["wait"] * rng.normal(1.0, 0.16)
                        + high_acuity_cases * 1.2
                        + imaging_backlog_cases * 0.5
                        + no_show_count * 0.3
                    )),
                )

                records.append({
                    "date": cal["date"],
                    "shift": shift,
                    "clinical_unit": clinical_unit,
                    "forecast_patient_volume": forecast_volume,
                    "actual_patient_volume": actual_volume,
                    "scheduled_procedures": scheduled_procedures,
                    "active_care_stations": active_care_stations,
                    "high_acuity_cases": high_acuity_cases,
                    "imaging_backlog_cases": imaging_backlog_cases,
                    "no_show_count": no_show_count,
                    "respiratory_alert_cases": respiratory_alert_cases,
                    "respiratory_case_ratio": respiratory_case_ratio,
                    "average_wait_minutes": average_wait_minutes,
                    "is_holiday": bool(cal["is_holiday"]),
                    "is_weekend": bool(cal["is_weekend"]),
                })

    return pd.DataFrame(records)


def estimate_required_staff(
    clinical_unit: str,
    shift: str,
    patient_volume: float,
    high_acuity_cases: float,
    scheduled_procedures: float,
    active_care_stations: float,
    imaging_backlog_cases: float,
    respiratory_alert_cases: float,
    respiratory_case_ratio: float,
    is_holiday: bool,
    is_weekend: bool,
) -> dict[str, int]:
    if clinical_unit == "consulta_general":
        pressure = patient_volume + (high_acuity_cases * 1.1) + (respiratory_alert_cases * 0.8)
        role_counts = {
            "medico": max(2, math.ceil(pressure / 18)),
            "enfermera": max(1, math.ceil(pressure / 28)),
            "tens": max(2, math.ceil(pressure / 20)),
            "admision": max(1, math.ceil(patient_volume / 34)),
            "tecnologo_medico": 0,
            "coordinador_clinico": int(pressure > 54),
        }
    elif clinical_unit == "especialidades":
        pressure = patient_volume + (high_acuity_cases * 1.5) + (scheduled_procedures * 0.5)
        role_counts = {
            "medico": max(2, math.ceil(pressure / 16)),
            "enfermera": max(1, math.ceil(pressure / 26)),
            "tens": max(1, math.ceil(pressure / 22)),
            "admision": max(1, math.ceil(patient_volume / 30)),
            "tecnologo_medico": 0,
            "coordinador_clinico": int(pressure > 48),
        }
    elif clinical_unit == "procedimientos_ambulatorios":
        pressure = (scheduled_procedures * 3.0) + (high_acuity_cases * 1.4) + (active_care_stations * 0.9)
        role_counts = {
            "medico": max(2, math.ceil(pressure / 12)),
            "enfermera": max(2, math.ceil(pressure / 12)),
            "tens": max(2, math.ceil(pressure / 10)),
            "admision": int(patient_volume >= 16),
            "tecnologo_medico": int(scheduled_procedures >= 6),
            "coordinador_clinico": max(1, math.ceil(max(scheduled_procedures, 1) / 8)),
        }
    elif clinical_unit == "imagenologia":
        pressure = patient_volume + (imaging_backlog_cases * 0.7) + (scheduled_procedures * 0.8)
        role_counts = {
            "medico": 1 + int(high_acuity_cases >= 6),
            "enfermera": 1 + int(high_acuity_cases >= 5),
            "tens": 1 + int(pressure >= 42),
            "admision": 1 + int(patient_volume >= 38),
            "tecnologo_medico": max(2, math.ceil(pressure / 22)),
            "coordinador_clinico": int(imaging_backlog_cases >= 16),
        }
    else:
        pressure = patient_volume + (respiratory_alert_cases * 1.1)
        role_counts = {
            "medico": int(high_acuity_cases >= 3),
            "enfermera": max(1, math.ceil(pressure / 34)),
            "tens": max(2, math.ceil(pressure / 18)),
            "admision": max(1, math.ceil(patient_volume / 36)),
            "tecnologo_medico": 1 + int(patient_volume >= 70),
            "coordinador_clinico": int(pressure >= 76),
        }

    multiplier = 1.0 + (respiratory_case_ratio * 0.14)
    if shift == "evening":
        multiplier += 0.04
    if is_weekend:
        multiplier *= 0.72
    if is_holiday:
        multiplier *= 0.58

    return {
        role: max(0, int(math.ceil(value * multiplier))) if value > 0 else 0
        for role, value in role_counts.items()
    }


def _build_daily_availability(employees_df: pd.DataFrame, calendar_df: pd.DataFrame, seed: int) -> dict[tuple[str, str], bool]:
    rng = np.random.default_rng(seed)
    availability: dict[tuple[str, str], bool] = {}
    weekend_lookup = calendar_df.set_index("date")[["is_weekend", "is_holiday"]].to_dict("index")

    for _, employee in employees_df.iterrows():
        employee_id = str(employee["employee_id"])
        contract_type = str(employee["contract_type"])
        for date, flags in weekend_lookup.items():
            is_special_day = bool(flags["is_weekend"] or flags["is_holiday"])
            if contract_type == "full_time":
                prob = 0.86 if not is_special_day else 0.28
            elif contract_type == "part_time":
                prob = 0.58 if not is_special_day else 0.18
            else:
                prob = 0.08 if not is_special_day else 0.64
            availability[(employee_id, date)] = bool(rng.random() < prob)

    return availability


def _shift_matches(shift: str, shift_preference: str) -> bool:
    return shift_preference == "flexible" or shift_preference == shift


def generate_clinic_work_records(
    employees_df: pd.DataFrame,
    patient_flow_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    logger.info("[clinic] Generando registros de trabajo")
    rng = np.random.default_rng(seed)
    employee_lookup = employees_df.set_index("employee_id")
    availability = _build_daily_availability(employees_df, calendar_df, seed)
    streaks = {employee_id: 0 for employee_id in employees_df["employee_id"]}
    worked_previous_day = {employee_id: False for employee_id in employees_df["employee_id"]}
    records = []

    shift_order = {shift: index for index, shift in enumerate(CLINIC_SHIFTS)}
    dates = sorted(calendar_df["date"].tolist())
    flow_by_date = {date: patient_flow_df[patient_flow_df["date"] == date].to_dict("records") for date in dates}
    calendar_lookup = calendar_df.set_index("date").to_dict("index")

    for date in dates:
        assigned_today: set[str] = set()
        day_rows = []
        day_segments = sorted(flow_by_date[date], key=lambda row: (shift_order[str(row["shift"])], str(row["clinical_unit"])))
        day_meta = calendar_lookup[date]

        for segment in day_segments:
            required = estimate_required_staff(
                clinical_unit=str(segment["clinical_unit"]),
                shift=str(segment["shift"]),
                patient_volume=float(segment["forecast_patient_volume"]),
                high_acuity_cases=float(segment["high_acuity_cases"]),
                scheduled_procedures=float(segment["scheduled_procedures"]),
                active_care_stations=float(segment["active_care_stations"]),
                imaging_backlog_cases=float(segment["imaging_backlog_cases"]),
                respiratory_alert_cases=float(segment["respiratory_alert_cases"]),
                respiratory_case_ratio=float(segment["respiratory_case_ratio"]),
                is_holiday=bool(segment["is_holiday"]),
                is_weekend=bool(segment["is_weekend"]),
            )

            patient_load_score = float(np.clip(segment["forecast_patient_volume"] / max(sum(required.values()), 1), 0.4, 5.0))
            acuity_load_score = float(np.clip((segment["high_acuity_cases"] * 1.5 + segment["scheduled_procedures"] * 1.8) / max(sum(required.values()), 1), 0.1, 4.0))
            infection_exposure_score = float(np.clip(segment["respiratory_case_ratio"] + (segment["respiratory_alert_cases"] / 20), 0.0, 1.5))

            for role, needed in required.items():
                buffer = 1 if role in {"medico", "enfermera"} and segment["clinical_unit"] == "procedimientos_ambulatorios" else 0
                target_count = needed + buffer

                primary_candidates = employees_df[
                    (employees_df["role"] == role)
                    & (employees_df["primary_unit"] == segment["clinical_unit"])
                ]["employee_id"].tolist()
                float_candidates = employees_df[
                    (employees_df["role"] == role)
                    & (employees_df["can_float"])
                    & (
                        (employees_df["backup_unit"] == segment["clinical_unit"])
                        | (employees_df["primary_unit"] == segment["clinical_unit"])
                    )
                ]["employee_id"].tolist()
                fallback_candidates = employees_df[employees_df["role"] == role]["employee_id"].tolist()

                selected: list[str] = []
                for candidate_pool in [primary_candidates, float_candidates, fallback_candidates]:
                    rng.shuffle(candidate_pool)
                    for employee_id in candidate_pool:
                        if len(selected) >= target_count:
                            break
                        if employee_id in assigned_today or employee_id in selected:
                            continue
                        if not availability.get((employee_id, date), False):
                            continue
                        employee = employee_lookup.loc[employee_id]
                        if not _shift_matches(str(segment["shift"]), str(employee["shift_preference"])) and not bool(employee["can_float"]):
                            continue
                        if str(employee["contract_type"]) == "weekend_support" and not (bool(day_meta["is_weekend"]) or bool(day_meta["is_holiday"])):
                            continue
                        selected.append(employee_id)
                        assigned_today.add(employee_id)

                for employee_id in selected:
                    employee = employee_lookup.loc[employee_id]
                    overtime_hours = 0.0
                    if segment["shift"] == "evening" and rng.random() < float(employee["overtime_tolerance"]):
                        overtime_hours = round(float(rng.uniform(0.3, 1.2)), 1)
                    if patient_load_score > 2.8 and rng.random() < 0.18:
                        overtime_hours = max(overtime_hours, round(float(rng.uniform(0.4, 1.5)), 1))

                    day_rows.append({
                        "employee_id": employee_id,
                        "date": date,
                        "shift": segment["shift"],
                        "clinical_unit": segment["clinical_unit"],
                        "role_assigned": role,
                        "scheduled_hours": 8.0 if segment["shift"] == "morning" else 7.0,
                        "hours_worked": 8.0 if segment["shift"] == "morning" else 7.0,
                        "overtime_hours": overtime_hours,
                        "consecutive_work_days": 0,
                        "is_rest_day": False,
                        "is_holiday": bool(segment["is_holiday"]),
                        "is_weekend": bool(segment["is_weekend"]),
                        "extended_hours_flag": int(segment["shift"] == "evening"),
                        "patient_load_score": round(patient_load_score, 2),
                        "acuity_load_score": round(acuity_load_score, 2),
                        "infection_exposure_score": round(infection_exposure_score, 2),
                    })

        worked_today = {row["employee_id"] for row in day_rows}
        for employee_id in streaks:
            if employee_id in worked_today:
                streaks[employee_id] = streaks[employee_id] + 1 if worked_previous_day[employee_id] else 1
                worked_previous_day[employee_id] = True
            else:
                streaks[employee_id] = 0
                worked_previous_day[employee_id] = False

        for row in day_rows:
            row["consecutive_work_days"] = streaks[row["employee_id"]]
            records.append(row)

    return pd.DataFrame(records)


def _health_event_schedule(employee_ids: list[str], days: int, seed: int) -> dict[str, tuple[np.ndarray, list[str]]]:
    rng = np.random.default_rng(seed)
    reasons = ["respiratorio", "burnout", "gastrointestinal", "lumbalgia", "migraña"]
    weights = np.array([0.32, 0.24, 0.16, 0.16, 0.12], dtype=float)
    weights = weights / weights.sum()
    events: dict[str, tuple[np.ndarray, list[str]]] = {}

    for employee_id in employee_ids:
        states = np.zeros(days, dtype=int)
        reasons_by_day = [""] * days
        day = 0
        while day < days:
            if rng.random() < 0.011:
                duration = int(rng.integers(1, 4))
                reason = str(rng.choice(reasons, p=weights))
                for offset, incubation_day in enumerate([day - 2, day - 1]):
                    if 0 <= incubation_day < days and states[incubation_day] == 0:
                        states[incubation_day] = offset + 1
                for sick_day in range(day, min(day + duration, days)):
                    states[sick_day] = 3
                    reasons_by_day[sick_day] = reason
                day += duration + int(rng.integers(3, 6))
            else:
                day += 1
        events[employee_id] = (states, reasons_by_day)

    return events


def generate_clinic_biometrics(
    employees_df: pd.DataFrame,
    work_records_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    logger.info("[clinic] Generando biometria diaria")
    rng = np.random.default_rng(seed)
    date_index = sorted(calendar_df["date"].tolist())
    date_to_idx = {date: index for index, date in enumerate(date_index)}
    calendar_lookup = calendar_df.set_index("date").to_dict("index")
    employee_lookup = employees_df.set_index("employee_id")
    health_events = _health_event_schedule(employees_df["employee_id"].tolist(), len(date_index), seed)
    work_daily = work_records_df.groupby(["employee_id", "date"]).agg(
        shifts_worked=("shift", "count"),
        scheduled_hours=("scheduled_hours", "sum"),
        overtime_hours=("overtime_hours", "sum"),
        consecutive_work_days=("consecutive_work_days", "max"),
        patient_load_score=("patient_load_score", "mean"),
        acuity_load_score=("acuity_load_score", "mean"),
        infection_exposure_score=("infection_exposure_score", "mean"),
        extended_hours_flag=("extended_hours_flag", "max"),
    )
    records = []
    previous_values: dict[str, dict[str, float]] = {}

    base_steps = {
        "medico": 7600,
        "enfermera": 9900,
        "tens": 10800,
        "admision": 6200,
        "tecnologo_medico": 8400,
        "coordinador_clinico": 7600,
    }

    for date in date_index:
        respiratory_alert = float(calendar_lookup[date]["respiratory_alert_level"])
        for employee_id in employees_df["employee_id"]:
            employee = employee_lookup.loc[employee_id]
            role = str(employee["role"])
            work_info = work_daily.loc[(employee_id, date)] if (employee_id, date) in work_daily.index else None
            shifts_worked = int(work_info["shifts_worked"]) if work_info is not None else 0
            overtime = float(work_info["overtime_hours"]) if work_info is not None else 0.0
            consecutive_days = int(work_info["consecutive_work_days"]) if work_info is not None else 0
            patient_load = float(work_info["patient_load_score"]) if work_info is not None else 0.0
            acuity_load = float(work_info["acuity_load_score"]) if work_info is not None else 0.0
            infection_exposure = float(work_info["infection_exposure_score"]) if work_info is not None else 0.0
            extended_hours_flag = int(work_info["extended_hours_flag"]) if work_info is not None else 0

            state = health_events[employee_id][0][date_to_idx[date]]
            reason = health_events[employee_id][1][date_to_idx[date]]
            load_index = patient_load * 7 + acuity_load * 8 + overtime * 8 + extended_hours_flag * 5 + consecutive_days * 1.6
            load_index += infection_exposure * 11 + respiratory_alert * 1.8
            if state in {1, 2}:
                load_index += rng.uniform(6, 11)
            if state == 3:
                load_index += rng.uniform(17, 26)

            hr_base = 69 + (2 if role in {"enfermera", "tens"} else 0) + (2 if bool(employee["chronic_condition_flag"]) else 0)
            hrv_base = 44 - max(0.0, float(employee["bmi"]) - 26) * 0.6
            sleep_base = 7.2 - (0.30 if extended_hours_flag else 0.0)
            steps = base_steps[role] * (0.35 if shifts_worked == 0 else 1.0)

            prev = previous_values.get(employee_id)
            hr_mean = hr_base + load_index * 0.18 + rng.normal(0, 3)
            hrv = hrv_base - load_index * 0.21 + rng.normal(0, 4)
            sleep_duration = sleep_base - (patient_load * 0.15) - (overtime * 0.42) - rng.normal(0, 0.4)
            sleep_efficiency = 86 - load_index * 0.20 + rng.normal(0, 4)
            stress_score = np.clip(26 + load_index + rng.normal(0, 5), 8, 99)
            cognitive_load_score = np.clip(24 + load_index * 1.12 + rng.normal(0, 5), 10, 100)
            reaction_time_ms = 240 + load_index * 2.5 + rng.normal(0, 12)
            steps = steps + patient_load * 1050 + acuity_load * 620 + rng.normal(0, 800)

            if state == 3:
                hr_mean += 10
                hrv -= 11
                sleep_duration -= 1.1
                sleep_efficiency -= 8
                stress_score += 10
                cognitive_load_score += 8
                reaction_time_ms += 28
                steps *= 0.58

            if prev is not None:
                hr_mean = 0.63 * prev["hr_mean_bpm"] + 0.37 * hr_mean
                hrv = 0.60 * prev["hrv_rmssd_ms"] + 0.40 * hrv
                sleep_duration = 0.55 * prev["sleep_duration_hours"] + 0.45 * sleep_duration
                sleep_efficiency = 0.55 * prev["sleep_efficiency_pct"] + 0.45 * sleep_efficiency
                stress_score = 0.50 * prev["stress_score"] + 0.50 * stress_score
                reaction_time_ms = 0.50 * prev["reaction_time_ms"] + 0.50 * reaction_time_ms

            fatigue_proxy = np.clip((100 - sleep_efficiency) * 0.34 + stress_score * 0.42 + max(0, consecutive_days - 4) * 4.2, 0, 100)
            data_quality_score = round(float(rng.uniform(0.74, 0.99) if rng.random() > 0.07 else rng.uniform(0.20, 0.55)), 2)

            record = {
                "employee_id": employee_id,
                "date": date,
                "hr_mean_bpm": round(float(np.clip(hr_mean, 46, 138)), 1),
                "hrv_rmssd_ms": round(float(np.clip(hrv, 8, 98)), 1),
                "sleep_duration_hours": round(float(np.clip(sleep_duration, 3.0, 9.6)), 1),
                "sleep_efficiency_pct": round(float(np.clip(sleep_efficiency, 45, 98)), 1),
                "stress_score": round(float(stress_score), 1),
                "steps": int(np.clip(steps, 900, 28000)),
                "cognitive_load_score": round(float(cognitive_load_score), 1),
                "reaction_time_ms": round(float(np.clip(reaction_time_ms, 180, 480)), 1),
                "infection_exposure_score": round(float(np.clip(infection_exposure + respiratory_alert * 0.08, 0, 2.0)), 2),
                "fatigue_proxy": round(float(fatigue_proxy), 1),
                "data_quality_score": data_quality_score,
                "_is_sick": int(state == 3),
                "_absence_reason": reason,
            }
            previous_values[employee_id] = record
            records.append(record)

    return pd.DataFrame(records)


def generate_clinic_absenteeism(
    employees_df: pd.DataFrame,
    biometrics_df: pd.DataFrame,
    work_records_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    logger.info("[clinic] Generando ausentismo")
    rng = np.random.default_rng(seed)
    work_daily = work_records_df.groupby(["employee_id", "date"]).agg(
        scheduled_hours=("scheduled_hours", "sum"),
        consecutive_work_days=("consecutive_work_days", "max"),
        extended_hours_flag=("extended_hours_flag", "max"),
        infection_exposure_score=("infection_exposure_score", "mean"),
        is_weekend=("is_weekend", "max"),
        is_holiday=("is_holiday", "max"),
    )
    bio_index = biometrics_df.set_index(["employee_id", "date"])
    records = []

    for (employee_id, date), work_info in work_daily.iterrows():
        bio = bio_index.loc[(employee_id, date)]
        stress = float(bio["stress_score"])
        sleep = float(bio["sleep_duration_hours"])
        fatigue = float(bio["fatigue_proxy"])
        cognitive = float(bio["cognitive_load_score"])
        infection_exposure = float(bio["infection_exposure_score"])
        is_sick = int(bio["_is_sick"])
        absence_reason = str(bio["_absence_reason"])

        prob = 0.008
        if is_sick:
            prob += 0.56
        prob += max(0.0, stress - 72) * 0.0028
        prob += max(0.0, fatigue - 68) * 0.0023
        prob += max(0.0, cognitive - 75) * 0.0018
        prob += max(0.0, 5.7 - sleep) * 0.05
        prob += infection_exposure * 0.035
        prob += max(0, int(work_info["consecutive_work_days"]) - 5) * 0.012
        prob += int(work_info["extended_hours_flag"]) * 0.012
        if bool(work_info["is_weekend"]) or bool(work_info["is_holiday"]):
            prob += 0.010

        if rng.random() < min(prob, 0.92):
            notice_hours = max(0, round(float(rng.normal(6 if stress > 74 or is_sick else 16, 5)), 1))
            is_short_notice = notice_hours < 8
            if not absence_reason:
                if cognitive > 80 or fatigue > 78:
                    absence_reason = "fatiga_aguda"
                elif infection_exposure > 0.8:
                    absence_reason = "cuadro_respiratorio"
                else:
                    absence_reason = "contingencia_personal"
            records.append({
                "employee_id": employee_id,
                "date": date,
                "absence_reason": absence_reason,
                "absence_hours": round(float(work_info["scheduled_hours"]), 1),
                "notice_hours": notice_hours,
                "is_short_notice": bool(is_short_notice),
                "is_absent": 1,
            })

    return pd.DataFrame(
        records,
        columns=[
            "employee_id",
            "date",
            "absence_reason",
            "absence_hours",
            "notice_hours",
            "is_short_notice",
            "is_absent",
        ],
    )


def generate_clinic_raw_data(employees: int, days: int, seed: int, start_date: str) -> dict[str, pd.DataFrame]:
    calendar_df = generate_clinic_calendar(start_date=start_date, days=days)
    employees_df = generate_clinic_employees(n=employees, seed=seed)
    patient_flow_df = generate_clinic_patient_flow(calendar_df=calendar_df, seed=seed)
    work_records_df = generate_clinic_work_records(employees_df, patient_flow_df, calendar_df, seed=seed)
    biometrics_df = generate_clinic_biometrics(employees_df, work_records_df, calendar_df, seed=seed)
    absenteeism_df = generate_clinic_absenteeism(employees_df, biometrics_df, work_records_df, seed=seed)
    return {
        "clinic_employees": employees_df,
        "clinic_calendar": calendar_df,
        "clinic_patient_flow": patient_flow_df,
        "clinic_work_records": work_records_df,
        "clinic_biometrics": biometrics_df,
        "clinic_absenteeism": absenteeism_df,
    }


def save_clinic_raw_data(dataframes: dict[str, pd.DataFrame]) -> None:
    CLINIC_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in dataframes.items():
        df.to_csv(CLINIC_RAW_DIR / f"{name}.csv", index=False)
    logger.info("[clinic] Datos raw guardados en %s", CLINIC_RAW_DIR)
