"""ETL del dominio restaurant casual dining."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.restaurant_settings import (
    CRITICAL_ROLE_TARGETS,
    RESTAURANT_DB_PATH,
    RESTAURANT_PROCESSED_DIR,
    RESTAURANT_RAW_DIR,
)
from src.etl.load import load_dataframe
from src.restaurant.generate import estimate_required_staff
from src.utils.database import get_connection

logger = logging.getLogger(__name__)


def _read_csv(name: str, parse_dates: list[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (RESTAURANT_RAW_DIR / f"{name}.csv")
    return pd.read_csv(csv_path, parse_dates=parse_dates or [])


def _staff_requirements_from_demand(demand_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in demand_df.iterrows():
        forecast_req = estimate_required_staff(
            covers=float(row["forecast_covers"]),
            delivery_orders=float(row["delivery_order_volume"]),
            service_period=str(row["service_period"]),
            is_holiday=bool(row["is_holiday"]),
            is_weekend=bool(row["is_weekend"]),
            local_event_flag=bool(row["local_event_flag"]),
        )
        actual_req = estimate_required_staff(
            covers=float(row["actual_covers"]),
            delivery_orders=float(row["delivery_order_volume"]),
            service_period=str(row["service_period"]),
            is_holiday=bool(row["is_holiday"]),
            is_weekend=bool(row["is_weekend"]),
            local_event_flag=bool(row["local_event_flag"]),
        )
        payload = {
            "date": row["date"],
            "service_period": row["service_period"],
            "forecast_required_headcount_total": int(sum(forecast_req.values())),
            "required_headcount_total": int(sum(actual_req.values())),
        }
        for role, value in forecast_req.items():
            payload[f"forecast_required_role_{role}"] = int(value)
        for role, value in actual_req.items():
            payload[f"required_role_{role}"] = int(value)
        rows.append(payload)
    return pd.DataFrame(rows)


def _build_staff_summary(work_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    total_df = work_df.groupby(["date", "service_period"]).agg(
        headcount=("employee_id", "nunique"),
        overtime_hours=("overtime_hours", "sum"),
        avg_consecutive_work_days=("consecutive_work_days", "mean"),
        avg_demand_pressure=("demand_pressure_score", "mean"),
        delivery_order_volume=("delivery_order_volume", "mean"),
    ).reset_index()
    total_df = total_df.rename(columns={
        "headcount": f"{prefix}_headcount_total",
        "overtime_hours": f"{prefix}_overtime_hours",
        "avg_consecutive_work_days": f"{prefix}_avg_consecutive_work_days",
        "avg_demand_pressure": f"{prefix}_avg_demand_pressure",
        "delivery_order_volume": f"{prefix}_delivery_orders_seen",
    })

    role_counts = work_df.groupby(["date", "service_period", "role_assigned"])["employee_id"].nunique().reset_index()
    role_pivot = role_counts.pivot_table(
        index=["date", "service_period"],
        columns="role_assigned",
        values="employee_id",
        fill_value=0,
    ).reset_index()
    role_pivot.columns = [
        column if isinstance(column, str) else column
        for column in role_pivot.columns
    ]
    role_pivot = role_pivot.rename(columns={role: f"{prefix}_role_{role}" for role in role_pivot.columns if role not in {"date", "service_period"}})
    return total_df.merge(role_pivot, on=["date", "service_period"], how="left")


def _build_actual_work_df(work_df: pd.DataFrame, abs_df: pd.DataFrame) -> pd.DataFrame:
    absent_keys = abs_df[["employee_id", "date"]].drop_duplicates()
    absent_keys["is_absent"] = 1
    merged = work_df.merge(absent_keys, on=["employee_id", "date"], how="left")
    merged["is_absent"] = merged["is_absent"].fillna(0).astype(int)
    actual_df = merged[merged["is_absent"] == 0].copy()
    return actual_df


def _build_biometrics_period_features(work_df: pd.DataFrame, bio_df: pd.DataFrame, employees_df: pd.DataFrame) -> pd.DataFrame:
    bio_cols = [
        "hr_mean_bpm", "hrv_rmssd_ms", "sleep_duration_hours", "sleep_efficiency_pct",
        "stress_score", "steps", "fatigue_proxy",
    ]
    merged = work_df.merge(bio_df[["employee_id", "date", *bio_cols]], on=["employee_id", "date"], how="left")
    merged = merged.merge(employees_df[["employee_id", "age", "bmi", "distance_to_work_km", "cross_trained"]], on="employee_id", how="left")
    agg = merged.groupby(["date", "service_period"]).agg(
        avg_hr_mean_bpm=("hr_mean_bpm", "mean"),
        avg_hrv_rmssd_ms=("hrv_rmssd_ms", "mean"),
        avg_sleep_duration_hours=("sleep_duration_hours", "mean"),
        avg_sleep_efficiency_pct=("sleep_efficiency_pct", "mean"),
        avg_stress_score=("stress_score", "mean"),
        avg_steps=("steps", "mean"),
        avg_fatigue_proxy=("fatigue_proxy", "mean"),
        avg_age=("age", "mean"),
        avg_bmi=("bmi", "mean"),
        avg_distance_to_work_km=("distance_to_work_km", "mean"),
        cross_trained_available=("cross_trained", "sum"),
    ).reset_index()
    return agg


def _build_absence_features(work_df: pd.DataFrame, abs_df: pd.DataFrame) -> pd.DataFrame:
    absent = work_df.merge(abs_df[["employee_id", "date", "is_short_notice", "absence_reason"]], on=["employee_id", "date"], how="inner")
    if absent.empty:
        return pd.DataFrame(columns=["date", "service_period", "absent_count_total", "short_notice_absent_count", "absentee_rate", "short_notice_absentee_rate"])

    absence_summary = absent.groupby(["date", "service_period"]).agg(
        absent_count_total=("employee_id", "nunique"),
        short_notice_absent_count=("is_short_notice", "sum"),
    ).reset_index()
    scheduled = work_df.groupby(["date", "service_period"])["employee_id"].nunique().reset_index(name="scheduled_headcount_total")
    absence_summary = absence_summary.merge(scheduled, on=["date", "service_period"], how="left")
    absence_summary["absentee_rate"] = absence_summary["absent_count_total"] / absence_summary["scheduled_headcount_total"].replace(0, np.nan)
    absence_summary["short_notice_absentee_rate"] = absence_summary["short_notice_absent_count"] / absence_summary["scheduled_headcount_total"].replace(0, np.nan)
    return absence_summary.drop(columns=["scheduled_headcount_total"])


def _add_temporal_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["service_period", "date"]).reset_index(drop=True).copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["is_friday"] = (result["day_of_week"] == 4).astype(int)
    result["is_weekend_flag"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["is_peak_service"] = result["service_period"].isin(["13_15", "19_21"]).astype(int)

    for feature in [
        "forecast_covers", "forecast_sales", "reservation_count", "delivery_order_volume",
        "absentee_rate", "short_notice_absentee_rate", "actual_headcount_total", "deficit_count_total",
    ]:
        if feature in result.columns:
            result[f"{feature}_lag1"] = result.groupby("service_period")[feature].shift(1)
            result[f"{feature}_lag7"] = result.groupby("service_period")[feature].shift(7)

    return result.fillna(0)


def build_restaurant_ml_features(
    employees_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    work_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    abs_df: pd.DataFrame,
) -> pd.DataFrame:
    requirement_df = _staff_requirements_from_demand(demand_df)
    scheduled_df = _build_staff_summary(work_df, prefix="scheduled")
    actual_work_df = _build_actual_work_df(work_df, abs_df)
    actual_df = _build_staff_summary(actual_work_df, prefix="actual")
    biometrics_df = _build_biometrics_period_features(work_df, bio_df, employees_df)
    absence_df = _build_absence_features(work_df, abs_df)

    merged = demand_df.merge(calendar_df, on="date", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.merge(requirement_df, on=["date", "service_period"], how="left")
    merged = merged.merge(scheduled_df, on=["date", "service_period"], how="left")
    merged = merged.merge(actual_df, on=["date", "service_period"], how="left")
    merged = merged.merge(biometrics_df, on=["date", "service_period"], how="left")
    merged = merged.merge(absence_df, on=["date", "service_period"], how="left")
    merged = merged.fillna(0)

    merged["deficit_count_total"] = (merged["required_headcount_total"] - merged["actual_headcount_total"]).clip(lower=0)
    merged["has_deficit_total"] = (merged["deficit_count_total"] > 0).astype(int)

    for role in CRITICAL_ROLE_TARGETS:
        required_col = f"required_role_{role}"
        actual_col = f"actual_role_{role}"
        deficit_col = f"deficit_role_{role}"
        has_deficit_col = f"has_deficit_role_{role}"
        if actual_col not in merged.columns:
            merged[actual_col] = 0
        merged[deficit_col] = (merged[required_col] - merged[actual_col]).clip(lower=0)
        merged[has_deficit_col] = (merged[deficit_col] > 0).astype(int)

    processed = _add_temporal_and_lag_features(merged)
    return processed


def run_restaurant_etl(employees_path: Path | None = None) -> pd.DataFrame:
    logger.info("[restaurant] Ejecutando ETL")
    employees_df = _read_csv("restaurant_employees", parse_dates=["hire_date"], path=employees_path)
    calendar_df = _read_csv("restaurant_calendar_cl", parse_dates=["date"])
    demand_df = _read_csv("restaurant_demand", parse_dates=["date"])
    work_df = _read_csv("restaurant_work_records", parse_dates=["date"])
    bio_df = _read_csv("restaurant_biometrics", parse_dates=["date"])
    abs_df = _read_csv("restaurant_absenteeism", parse_dates=["date"])

    ml_features_df = build_restaurant_ml_features(employees_df, calendar_df, demand_df, work_df, bio_df, abs_df)
    RESTAURANT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ml_features_df.to_csv(RESTAURANT_PROCESSED_DIR / "restaurant_ml_features.csv", index=False)

    with get_connection(RESTAURANT_DB_PATH) as conn:
        load_dataframe(employees_df, "restaurant_employees", conn)
        load_dataframe(calendar_df, "restaurant_calendar_cl", conn)
        load_dataframe(demand_df, "restaurant_demand", conn)
        load_dataframe(work_df, "restaurant_work_records", conn)
        load_dataframe(bio_df, "restaurant_biometrics", conn)
        load_dataframe(abs_df, "restaurant_absenteeism", conn)
        load_dataframe(ml_features_df, "restaurant_ml_features", conn)

    logger.info("[restaurant] ETL completado: %d filas en restaurant_ml_features", len(ml_features_df))
    return ml_features_df
