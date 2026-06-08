"""Generacion sintetica del dominio restaurant casual dining."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from config.restaurant_settings import (
    CHILE_HOLIDAYS_2025,
    RESTAURANT_RAW_DIR,
    RESTAURANT_ROLES,
    ROLE_BACKUPS,
    SERVICE_PERIODS,
)

logger = logging.getLogger(__name__)


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "summer"
    if month in {3, 4, 5}:
        return "autumn"
    if month in {6, 7, 8}:
        return "winter"
    return "spring"


def generate_restaurant_calendar(start_date: str, days: int) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=days, freq="D")
    records = []
    holidays = {pd.Timestamp(key): value for key, value in CHILE_HOLIDAYS_2025.items()}

    for date in dates:
        is_holiday = date in holidays
        next_day = date + pd.Timedelta(days=1)
        prev_day = date - pd.Timedelta(days=1)
        is_holiday_eve = next_day in holidays
        is_long_weekend = is_holiday and date.dayofweek in {0, 4} or is_holiday_eve and date.dayofweek == 4
        is_school_break = (date.month == 7 and date.day <= 20) or (date.month == 12 and date.day >= 15)
        is_payday_window = date.day in {29, 30, 1, 2, 14, 15}
        tourism_pressure = 0.55 if date.month in {7, 9, 12} else 0.35
        if is_holiday:
            tourism_pressure += 0.20
        if is_school_break:
            tourism_pressure += 0.10

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "is_holiday": is_holiday,
            "holiday_name": holidays.get(date, ""),
            "is_holiday_eve": is_holiday_eve,
            "is_long_weekend": bool(is_long_weekend),
            "is_school_break": bool(is_school_break),
            "is_payday_window": bool(is_payday_window),
            "season": _season_for_month(date.month),
            "tourism_pressure_score": round(float(np.clip(tourism_pressure, 0, 1)), 2),
        })

    return pd.DataFrame(records)


def generate_restaurant_employees(n: int, seed: int = 42) -> pd.DataFrame:
    logger.info("[restaurant] Generando empleados: %d", n)
    rng = np.random.default_rng(seed)
    role_names = list(RESTAURANT_ROLES.keys())
    role_weights = np.array([RESTAURANT_ROLES[role]["weight"] for role in role_names])
    role_weights = role_weights / role_weights.sum()
    records = []

    for index in range(n):
        role = str(rng.choice(role_names, p=role_weights))
        contract_type = "full_time"
        if role in {"garzon", "runner", "host", "copero"} and rng.random() < 0.35:
            contract_type = "part_time"
        if role in {"garzon", "runner", "host"} and rng.random() < 0.10:
            contract_type = "weekend_only"

        shift_preference = str(rng.choice(["midday", "evening", "mixed"], p=[0.30, 0.35, 0.35]))
        if role in {"cocinero_linea", "ayudante_cocina", "jefe_turno"}:
            shift_preference = str(rng.choice(["midday", "evening", "mixed"], p=[0.20, 0.45, 0.35]))

        cross_trained = bool(rng.random() < 0.42)
        backup_role = ""
        if cross_trained and ROLE_BACKUPS.get(role):
            backup_role = str(rng.choice(ROLE_BACKUPS[role]))

        gender = "F" if rng.random() < 0.48 else "M"
        age = int(np.clip(rng.normal(30, 7), 18, 58))
        bmi = round(float(np.clip(rng.normal(25.8, 3.8), 18, 38)), 1)
        seniority_years = int(np.clip(rng.exponential(2.8), 0, max(age - 18, 1)))
        distance_km = round(float(np.clip(rng.lognormal(2.0, 0.55), 1, 35)), 1)
        children = int(np.clip(rng.poisson(0.8), 0, 4))
        social_drinker = bool(rng.random() < 0.32)
        smoker = bool(rng.random() < 0.18)
        hire_date = pd.Timestamp("2025-12-31") - pd.Timedelta(days=int(seniority_years * 365.25))

        records.append({
            "employee_id": f"RST_{index + 1:03d}",
            "role": role,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "seniority_years": seniority_years,
            "contract_type": contract_type,
            "shift_preference": shift_preference,
            "cross_trained": cross_trained,
            "backup_role": backup_role,
            "distance_to_work_km": distance_km,
            "children": children,
            "social_drinker": social_drinker,
            "smoker": smoker,
            "hire_date": hire_date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(records)


def estimate_required_staff(covers: float, delivery_orders: float, service_period: str, is_holiday: bool, is_weekend: bool, local_event_flag: bool) -> dict[str, int]:
    demand_pressure = covers + (delivery_orders * 0.7)
    pressure_multiplier = SERVICE_PERIODS[service_period]["demand_factor"]
    if is_holiday:
        pressure_multiplier += 0.15
    if is_weekend:
        pressure_multiplier += 0.10
    if local_event_flag:
        pressure_multiplier += 0.08

    effective_demand = demand_pressure * pressure_multiplier
    return {
        "garzon": max(2, math.ceil(effective_demand / 22)),
        "host": 1 + int(effective_demand > 75),
        "cajero": 1 + int(effective_demand > 95),
        "cocinero_linea": max(2, math.ceil((covers + delivery_orders) / 30)),
        "ayudante_cocina": max(1, math.ceil((covers + delivery_orders) / 45)),
        "runner": 1 + int(effective_demand > 60),
        "copero": 1 + int(effective_demand > 85),
        "jefe_turno": 1,
    }


def generate_restaurant_demand(calendar_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    logger.info("[restaurant] Generando demanda operativa")
    rng = np.random.default_rng(seed)
    base_covers = {"11_13": 38, "13_15": 62, "19_21": 78, "21_23": 46}
    day_multiplier = {0: 0.92, 1: 0.94, 2: 0.98, 3: 1.03, 4: 1.18, 5: 1.28, 6: 1.12}
    records = []

    for _, cal in calendar_df.iterrows():
        date = pd.Timestamp(cal["date"])
        is_weekend = date.dayofweek >= 5
        for service_period in SERVICE_PERIODS:
            promo_flag = bool(rng.random() < (0.16 if service_period in {"11_13", "13_15"} else 0.10))
            local_event_flag = bool(rng.random() < (0.18 if date.dayofweek in {4, 5} else 0.07))
            weather_impact = rng.normal(-0.03 if cal["season"] == "winter" else 0.01, 0.08)
            reservation_base = {"11_13": 10, "13_15": 22, "19_21": 34, "21_23": 18}[service_period]
            walk_in_ratio = float(np.clip(rng.normal(0.48 if service_period.startswith("19") else 0.58, 0.08), 0.20, 0.85))
            delivery_base = {"11_13": 8, "13_15": 14, "19_21": 24, "21_23": 16}[service_period]

            multiplier = day_multiplier[date.dayofweek]
            if cal["is_holiday"]:
                multiplier += 0.22
            if cal["is_holiday_eve"] and service_period in {"19_21", "21_23"}:
                multiplier += 0.14
            if cal["is_payday_window"]:
                multiplier += 0.08
            if promo_flag:
                multiplier += 0.06
            if local_event_flag:
                multiplier += 0.10
            multiplier += float(weather_impact)

            forecast_covers = max(22, int(round(base_covers[service_period] * multiplier + rng.normal(0, 5))))
            reservation_count = max(4, int(round(reservation_base * multiplier + rng.normal(0, 4))))
            delivery_orders = max(2, int(round(delivery_base * (1.0 - weather_impact) + rng.normal(0, 3))))
            actual_covers = max(18, int(round(forecast_covers * rng.normal(1.0, 0.10) + reservation_count * 0.08 + local_event_flag * 6)))
            avg_ticket = 18500 if service_period in {"19_21", "21_23"} else 14200
            forecast_sales = int(round(forecast_covers * avg_ticket + delivery_orders * 9500))
            actual_sales = int(round(actual_covers * avg_ticket * rng.normal(1.0, 0.06) + delivery_orders * 9800))

            records.append({
                "date": cal["date"],
                "service_period": service_period,
                "forecast_covers": forecast_covers,
                "actual_covers": actual_covers,
                "forecast_sales": forecast_sales,
                "actual_sales": actual_sales,
                "reservation_count": reservation_count,
                "walk_in_ratio": round(walk_in_ratio, 2),
                "delivery_order_volume": delivery_orders,
                "promo_flag": promo_flag,
                "local_event_flag": local_event_flag,
                "weather_impact_score": round(float(np.clip(weather_impact, -0.25, 0.25)), 2),
                "is_holiday": bool(cal["is_holiday"]),
                "is_weekend": is_weekend,
            })

    return pd.DataFrame(records)


def _max_periods(contract_type: str) -> int:
    if contract_type == "full_time":
        return 4
    if contract_type == "part_time":
        return 2
    return 2


def _preference_matches(service_period: str, shift_preference: str) -> bool:
    if shift_preference == "mixed":
        return True
    midday_periods = {"11_13", "13_15"}
    evening_periods = {"19_21", "21_23"}
    if shift_preference == "midday":
        return service_period in midday_periods
    if shift_preference == "evening":
        return service_period in evening_periods
    return True


def generate_restaurant_work_records(employees_df: pd.DataFrame, demand_df: pd.DataFrame, calendar_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    logger.info("[restaurant] Generando registros de trabajo")
    rng = np.random.default_rng(seed)
    employee_lookup = employees_df.set_index("employee_id")
    records = []
    streaks = {emp_id: 0 for emp_id in employees_df["employee_id"]}
    worked_previous_day = {emp_id: False for emp_id in employees_df["employee_id"]}

    dates = sorted(calendar_df["date"].tolist())
    demand_by_date = {date: demand_df[demand_df["date"] == date].to_dict("records") for date in dates}

    for date in dates:
        calendar_row = calendar_df[calendar_df["date"] == date].iloc[0]
        is_weekend = bool(pd.Timestamp(date).dayofweek >= 5)
        daily_period_count = {emp_id: 0 for emp_id in employees_df["employee_id"]}
        day_rows = []

        for demand_row in sorted(demand_by_date[date], key=lambda row: row["service_period"]):
            required = estimate_required_staff(
                covers=float(demand_row["forecast_covers"]),
                delivery_orders=float(demand_row["delivery_order_volume"]),
                service_period=str(demand_row["service_period"]),
                is_holiday=bool(demand_row["is_holiday"]),
                is_weekend=bool(demand_row["is_weekend"]),
                local_event_flag=bool(demand_row["local_event_flag"]),
            )

            for role, needed in required.items():
                buffer = 1 if role in {"garzon", "cocinero_linea"} and demand_row["service_period"] in {"13_15", "19_21"} else 0
                target_count = needed + buffer
                primary_candidates = employees_df[employees_df["role"] == role]["employee_id"].tolist()
                backup_candidates = employees_df[(employees_df["backup_role"] == role) & (employees_df["cross_trained"])]["employee_id"].tolist()

                selected: list[str] = []
                for candidate_pool, is_backup in [(primary_candidates, False), (backup_candidates, True)]:
                    rng.shuffle(candidate_pool)
                    for emp_id in candidate_pool:
                        if len(selected) >= target_count:
                            break
                        emp = employee_lookup.loc[emp_id]
                        if emp["contract_type"] == "weekend_only" and not (is_weekend or calendar_row["is_holiday"]):
                            continue
                        if daily_period_count[emp_id] >= _max_periods(str(emp["contract_type"])):
                            continue
                        if not _preference_matches(str(demand_row["service_period"]), str(emp["shift_preference"])):
                            continue
                        if emp_id in selected:
                            continue
                        if is_backup and role == "jefe_turno":
                            continue
                        selected.append(emp_id)
                        daily_period_count[emp_id] += 1

                demand_pressure = float(demand_row["forecast_covers"] / max(target_count, 1))
                for emp_id in selected:
                    emp = employee_lookup.loc[emp_id]
                    overtime_hours = 0.0
                    if demand_row["service_period"] == "21_23" and rng.random() < 0.35:
                        overtime_hours = round(float(rng.uniform(0.2, 0.8)), 1)
                    if demand_pressure > 24 and rng.random() < 0.12:
                        overtime_hours = max(overtime_hours, round(float(rng.uniform(0.3, 0.9)), 1))

                    day_rows.append({
                        "employee_id": emp_id,
                        "date": date,
                        "service_period": demand_row["service_period"],
                        "role_assigned": role,
                        "scheduled_hours": 2.0,
                        "hours_worked": 2.0,
                        "overtime_hours": overtime_hours,
                        "consecutive_work_days": 0,
                        "is_rest_day": False,
                        "is_holiday": bool(calendar_row["is_holiday"]),
                        "is_weekend": is_weekend,
                        "demand_pressure_score": round(float(np.clip(demand_pressure / 30, 0, 1.5)), 2),
                        "delivery_order_volume": int(demand_row["delivery_order_volume"]),
                    })

        worked_today = {row["employee_id"] for row in day_rows}
        for emp_id in streaks:
            if emp_id in worked_today:
                streaks[emp_id] = streaks[emp_id] + 1 if worked_previous_day[emp_id] else 1
                worked_previous_day[emp_id] = True
            else:
                streaks[emp_id] = 0
                worked_previous_day[emp_id] = False

        for row in day_rows:
            row["consecutive_work_days"] = streaks[row["employee_id"]]
            records.append(row)

    return pd.DataFrame(records)


def _health_event_schedule(employee_ids: list[str], days: int, seed: int) -> dict[str, tuple[np.ndarray, list[str]]]:
    rng = np.random.default_rng(seed)
    reasons = ["gastrointestinal", "respiratorio", "fatiga", "musculoesqueletico", "stress"]
    weights = np.array([0.22, 0.18, 0.24, 0.18, 0.18])
    weights = weights / weights.sum()
    events: dict[str, tuple[np.ndarray, list[str]]] = {}
    for employee_id in employee_ids:
        states = np.zeros(days, dtype=int)
        reasons_by_day = [""] * days
        day = 0
        while day < days:
            if rng.random() < 0.010:
                duration = int(rng.integers(1, 4))
                reason = str(rng.choice(reasons, p=weights))
                for offset, incubation_day in enumerate([day - 2, day - 1]):
                    if 0 <= incubation_day < days and states[incubation_day] == 0:
                        states[incubation_day] = offset + 1
                for sick_day in range(day, min(day + duration, days)):
                    states[sick_day] = 3
                    reasons_by_day[sick_day] = reason
                day += duration + 4
            else:
                day += 1
        events[employee_id] = (states, reasons_by_day)
    return events


def generate_restaurant_biometrics(employees_df: pd.DataFrame, work_records_df: pd.DataFrame, calendar_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    logger.info("[restaurant] Generando biometria diaria")
    rng = np.random.default_rng(seed)
    date_index = sorted(calendar_df["date"].tolist())
    date_to_idx = {date: idx for idx, date in enumerate(date_index)}
    health_events = _health_event_schedule(employees_df["employee_id"].tolist(), len(date_index), seed)
    work_daily = work_records_df.groupby(["employee_id", "date"]).agg(
        periods_worked=("service_period", "count"),
        scheduled_hours=("scheduled_hours", "sum"),
        overtime_hours=("overtime_hours", "sum"),
        consecutive_work_days=("consecutive_work_days", "max"),
        demand_pressure_score=("demand_pressure_score", "mean"),
        has_evening=("service_period", lambda s: int(any(period in {"19_21", "21_23"} for period in s))),
    )
    employee_lookup = employees_df.set_index("employee_id")
    records = []
    previous_values: dict[str, dict[str, float]] = {}

    role_step_baseline = {
        "garzon": 12500,
        "runner": 14000,
        "host": 9500,
        "cocinero_linea": 8500,
        "ayudante_cocina": 9800,
        "copero": 9000,
        "cajero": 7500,
        "jefe_turno": 8200,
    }

    for date in date_index:
        calendar_row = calendar_df[calendar_df["date"] == date].iloc[0]
        for employee_id in employees_df["employee_id"]:
            emp = employee_lookup.loc[employee_id]
            role = str(emp["role"])
            work_info = work_daily.loc[(employee_id, date)] if (employee_id, date) in work_daily.index else None
            periods_worked = int(work_info["periods_worked"]) if work_info is not None else 0
            overtime = float(work_info["overtime_hours"]) if work_info is not None else 0.0
            consecutive_days = int(work_info["consecutive_work_days"]) if work_info is not None else 0
            demand_pressure = float(work_info["demand_pressure_score"]) if work_info is not None else 0.0
            has_evening = int(work_info["has_evening"]) if work_info is not None else 0

            state, reason = health_events[employee_id][0][date_to_idx[date]], health_events[employee_id][1][date_to_idx[date]]
            stress_adj = periods_worked * 6 + overtime * 8 + consecutive_days * 1.6 + demand_pressure * 18 + has_evening * 8
            if calendar_row["is_holiday"] or pd.Timestamp(date).dayofweek >= 5:
                stress_adj += 5
            if state in {1, 2}:
                stress_adj += rng.uniform(8, 14)
            if state == 3:
                stress_adj += rng.uniform(18, 28)

            hr_base = 70 + (3 if role in {"cocinero_linea", "runner"} else 0) + (2 if bool(emp["smoker"]) else 0)
            hrv_base = 42 - max(0.0, float(emp["bmi"]) - 26) * 0.7
            sleep_base = 7.2 - (0.4 if has_evening else 0.0)
            steps_base = role_step_baseline[role] * (0.35 if periods_worked == 0 else 1.0)

            prev = previous_values.get(employee_id, None)
            hr_mean = hr_base + stress_adj * 0.18 + rng.normal(0, 3)
            hrv = hrv_base - stress_adj * 0.22 + rng.normal(0, 4)
            sleep_duration = sleep_base - periods_worked * 0.25 - overtime * 0.40 - rng.normal(0, 0.5)
            sleep_eff = 84 - stress_adj * 0.22 + rng.normal(0, 4)
            steps = steps_base + periods_worked * 1100 + rng.normal(0, 900)
            if state == 3:
                hr_mean += 9
                hrv -= 12
                sleep_duration -= 1.4
                sleep_eff -= 9
                steps *= 0.45

            if prev is not None:
                hr_mean = 0.65 * prev["hr_mean_bpm"] + 0.35 * hr_mean
                hrv = 0.65 * prev["hrv_rmssd_ms"] + 0.35 * hrv
                sleep_duration = 0.55 * prev["sleep_duration_hours"] + 0.45 * sleep_duration
                sleep_eff = 0.55 * prev["sleep_efficiency_pct"] + 0.45 * sleep_eff
                steps = 0.45 * prev["steps"] + 0.55 * steps

            stress_score = np.clip(28 + stress_adj + rng.normal(0, 6), 8, 98)
            fatigue_proxy = np.clip((100 - sleep_eff) * 0.35 + stress_score * 0.45 + max(0, consecutive_days - 3) * 4, 0, 100)
            data_quality = round(float(rng.uniform(0.72, 0.99) if rng.random() > 0.07 else rng.uniform(0.15, 0.45)), 2)

            record = {
                "employee_id": employee_id,
                "date": date,
                "hr_mean_bpm": round(float(np.clip(hr_mean, 45, 135)), 1),
                "hrv_rmssd_ms": round(float(np.clip(hrv, 8, 95)), 1),
                "sleep_duration_hours": round(float(np.clip(sleep_duration, 3.0, 9.5)), 1),
                "sleep_efficiency_pct": round(float(np.clip(sleep_eff, 45, 98)), 1),
                "stress_score": round(float(stress_score), 1),
                "steps": int(np.clip(steps, 800, 30000)),
                "data_quality_score": data_quality,
                "fatigue_proxy": round(float(fatigue_proxy), 1),
                "_is_sick": int(state == 3),
                "_absence_reason": reason,
            }
            previous_values[employee_id] = record
            records.append(record)

    return pd.DataFrame(records)


def generate_restaurant_absenteeism(employees_df: pd.DataFrame, biometrics_df: pd.DataFrame, work_records_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    logger.info("[restaurant] Generando ausentismo")
    rng = np.random.default_rng(seed)
    work_daily = work_records_df.groupby(["employee_id", "date"]).agg(
        scheduled_hours=("scheduled_hours", "sum"),
        service_periods=("service_period", lambda s: list(s)),
        consecutive_work_days=("consecutive_work_days", "max"),
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
        is_sick = int(bio["_is_sick"])
        absence_reason = str(bio["_absence_reason"])

        prob = 0.006
        if is_sick:
            prob += 0.58
        prob += max(0.0, stress - 70) * 0.003
        prob += max(0.0, fatigue - 65) * 0.0025
        prob += max(0.0, 5.5 - sleep) * 0.06
        prob += max(0, int(work_info["consecutive_work_days"]) - 5) * 0.015
        if bool(work_info["is_weekend"]) or bool(work_info["is_holiday"]):
            prob += 0.012

        if rng.random() < min(prob, 0.90):
            notice_hours = max(0, round(float(rng.normal(5 if stress > 72 else 14, 4)), 1))
            is_short_notice = notice_hours < 8
            if not absence_reason:
                absence_reason = "short_notice_personal" if is_short_notice else "personal"
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


def generate_restaurant_raw_data(employees: int, days: int, seed: int, start_date: str) -> dict[str, pd.DataFrame]:
    calendar_df = generate_restaurant_calendar(start_date=start_date, days=days)
    employees_df = generate_restaurant_employees(n=employees, seed=seed)
    demand_df = generate_restaurant_demand(calendar_df=calendar_df, seed=seed)
    work_records_df = generate_restaurant_work_records(employees_df, demand_df, calendar_df, seed=seed)
    biometrics_df = generate_restaurant_biometrics(employees_df, work_records_df, calendar_df, seed=seed)
    absenteeism_df = generate_restaurant_absenteeism(employees_df, biometrics_df, work_records_df, seed=seed)
    return {
        "restaurant_employees": employees_df,
        "restaurant_calendar_cl": calendar_df,
        "restaurant_demand": demand_df,
        "restaurant_work_records": work_records_df,
        "restaurant_biometrics": biometrics_df,
        "restaurant_absenteeism": absenteeism_df,
    }


def save_restaurant_raw_data(dataframes: dict[str, pd.DataFrame]) -> None:
    RESTAURANT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in dataframes.items():
        df.to_csv(RESTAURANT_RAW_DIR / f"{name}.csv", index=False)
    logger.info("[restaurant] Datos raw guardados en %s", RESTAURANT_RAW_DIR)
