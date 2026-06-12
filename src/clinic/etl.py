"""ETL del dominio clinic ambulatorio."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.clinic_settings import CLINIC_CRITICAL_ROLE_TARGETS, CLINIC_DB_PATH, CLINIC_PROCESSED_DIR, CLINIC_RAW_DIR
from src.clinic.generate import estimate_required_staff
from src.etl.load import load_dataframe
from src.utils.database import get_connection

logger = logging.getLogger(__name__)


def _read_csv(name: str, parse_dates: list[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (CLINIC_RAW_DIR / f"{name}.csv")
    return pd.read_csv(csv_path, parse_dates=parse_dates or [])


def _staff_requirements_from_flow(flow_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in flow_df.iterrows():
        forecast_req = estimate_required_staff(
            clinical_unit=str(row["clinical_unit"]),
            shift=str(row["shift"]),
            patient_volume=float(row["forecast_patient_volume"]),
            high_acuity_cases=float(row["high_acuity_cases"]),
            scheduled_procedures=float(row["scheduled_procedures"]),
            active_care_stations=float(row["active_care_stations"]),
            imaging_backlog_cases=float(row["imaging_backlog_cases"]),
            respiratory_alert_cases=float(row["respiratory_alert_cases"]),
            respiratory_case_ratio=float(row["respiratory_case_ratio"]),
            is_holiday=bool(row["is_holiday"]),
            is_weekend=bool(row["is_weekend"]),
        )
        actual_req = estimate_required_staff(
            clinical_unit=str(row["clinical_unit"]),
            shift=str(row["shift"]),
            patient_volume=float(row["actual_patient_volume"]),
            high_acuity_cases=float(row["high_acuity_cases"]),
            scheduled_procedures=float(row["scheduled_procedures"]),
            active_care_stations=float(row["active_care_stations"]),
            imaging_backlog_cases=float(row["imaging_backlog_cases"]),
            respiratory_alert_cases=float(row["respiratory_alert_cases"]),
            respiratory_case_ratio=float(row["respiratory_case_ratio"]),
            is_holiday=bool(row["is_holiday"]),
            is_weekend=bool(row["is_weekend"]),
        )
        payload = {
            "date": row["date"],
            "shift": row["shift"],
            "clinical_unit": row["clinical_unit"],
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
    total_df = work_df.groupby(["date", "shift", "clinical_unit"]).agg(
        headcount=("employee_id", "nunique"),
        overtime_hours=("overtime_hours", "sum"),
        avg_consecutive_work_days=("consecutive_work_days", "mean"),
        avg_patient_load_score=("patient_load_score", "mean"),
        avg_acuity_load_score=("acuity_load_score", "mean"),
        avg_infection_exposure_score=("infection_exposure_score", "mean"),
    ).reset_index()
    total_df = total_df.rename(columns={
        "headcount": f"{prefix}_headcount_total",
        "overtime_hours": f"{prefix}_overtime_hours",
        "avg_consecutive_work_days": f"{prefix}_avg_consecutive_work_days",
        "avg_patient_load_score": f"{prefix}_avg_patient_load_score",
        "avg_acuity_load_score": f"{prefix}_avg_acuity_load_score",
        "avg_infection_exposure_score": f"{prefix}_avg_infection_exposure_score",
    })

    role_counts = work_df.groupby(["date", "shift", "clinical_unit", "role_assigned"])["employee_id"].nunique().reset_index()
    role_pivot = role_counts.pivot_table(
        index=["date", "shift", "clinical_unit"],
        columns="role_assigned",
        values="employee_id",
        fill_value=0,
    ).reset_index()
    role_pivot.columns = [column if isinstance(column, str) else column for column in role_pivot.columns]
    role_pivot = role_pivot.rename(columns={role: f"{prefix}_role_{role}" for role in role_pivot.columns if role not in {"date", "shift", "clinical_unit"}})
    return total_df.merge(role_pivot, on=["date", "shift", "clinical_unit"], how="left")


def _build_actual_work_df(work_df: pd.DataFrame, abs_df: pd.DataFrame) -> pd.DataFrame:
    absent_keys = abs_df[["employee_id", "date"]].drop_duplicates()
    absent_keys["is_absent"] = 1
    merged = work_df.merge(absent_keys, on=["employee_id", "date"], how="left")
    merged["is_absent"] = merged["is_absent"].fillna(0).astype(int)
    return merged[merged["is_absent"] == 0].copy()


def _build_biometrics_segment_features(work_df: pd.DataFrame, bio_df: pd.DataFrame, employees_df: pd.DataFrame) -> pd.DataFrame:
    bio_cols = [
        "hr_mean_bpm",
        "hrv_rmssd_ms",
        "sleep_duration_hours",
        "sleep_efficiency_pct",
        "stress_score",
        "steps",
        "cognitive_load_score",
        "reaction_time_ms",
        "infection_exposure_score",
        "fatigue_proxy",
    ]
    merged = work_df.merge(bio_df[["employee_id", "date", *bio_cols]], on=["employee_id", "date"], how="left")
    merged = merged.merge(
        employees_df[["employee_id", "age", "bmi", "years_experience", "commute_km", "can_float"]],
        on="employee_id",
        how="left",
    )
    return merged.groupby(["date", "shift", "clinical_unit"]).agg(
        avg_hr_mean_bpm=("hr_mean_bpm", "mean"),
        avg_hrv_rmssd_ms=("hrv_rmssd_ms", "mean"),
        avg_sleep_duration_hours=("sleep_duration_hours", "mean"),
        avg_sleep_efficiency_pct=("sleep_efficiency_pct", "mean"),
        avg_stress_score=("stress_score", "mean"),
        avg_steps=("steps", "mean"),
        avg_cognitive_load_score=("cognitive_load_score", "mean"),
        avg_reaction_time_ms=("reaction_time_ms", "mean"),
        avg_fatigue_proxy=("fatigue_proxy", "mean"),
        avg_age=("age", "mean"),
        avg_bmi=("bmi", "mean"),
        avg_years_experience=("years_experience", "mean"),
        avg_commute_km=("commute_km", "mean"),
        floating_staff_available=("can_float", "sum"),
    ).reset_index()


def _build_absence_features(work_df: pd.DataFrame, abs_df: pd.DataFrame) -> pd.DataFrame:
    absent = work_df.merge(abs_df[["employee_id", "date", "is_short_notice", "absence_reason"]], on=["employee_id", "date"], how="inner")
    if absent.empty:
        return pd.DataFrame(columns=["date", "shift", "clinical_unit", "absent_count_total", "short_notice_absent_count", "absentee_rate", "short_notice_absentee_rate"])

    absence_summary = absent.groupby(["date", "shift", "clinical_unit"]).agg(
        absent_count_total=("employee_id", "nunique"),
        short_notice_absent_count=("is_short_notice", "sum"),
    ).reset_index()
    scheduled = work_df.groupby(["date", "shift", "clinical_unit"])["employee_id"].nunique().reset_index(name="scheduled_headcount_total")
    absence_summary = absence_summary.merge(scheduled, on=["date", "shift", "clinical_unit"], how="left")
    absence_summary["absentee_rate"] = absence_summary["absent_count_total"] / absence_summary["scheduled_headcount_total"].replace(0, np.nan)
    absence_summary["short_notice_absentee_rate"] = absence_summary["short_notice_absent_count"] / absence_summary["scheduled_headcount_total"].replace(0, np.nan)
    return absence_summary.drop(columns=["scheduled_headcount_total"])


def _add_temporal_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["clinical_unit", "shift", "date"]).reset_index(drop=True).copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["is_monday"] = (result["day_of_week"] == 0).astype(int)
    result["is_weekend_flag"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["is_evening_shift"] = (result["shift"] == "evening").astype(int)

    for feature in [
        "forecast_patient_volume",
        "high_acuity_cases",
        "scheduled_procedures",
        "active_care_stations",
        "average_wait_minutes",
        "respiratory_case_ratio",
        "absentee_rate",
        "short_notice_absentee_rate",
        "actual_headcount_total",
        "deficit_count_total",
    ]:
        if feature in result.columns:
            result[f"{feature}_lag1"] = result.groupby(["clinical_unit", "shift"])[feature].shift(1)
            result[f"{feature}_lag7"] = result.groupby(["clinical_unit", "shift"])[feature].shift(7)

    return result.fillna(0)


def build_clinic_ml_features(
    employees_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    patient_flow_df: pd.DataFrame,
    work_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    abs_df: pd.DataFrame,
) -> pd.DataFrame:
    calendar_df = calendar_df.copy()
    patient_flow_df = patient_flow_df.copy()
    work_df = work_df.copy()
    bio_df = bio_df.copy()
    abs_df = abs_df.copy()

    for df in [calendar_df, patient_flow_df, work_df, bio_df, abs_df]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    requirement_df = _staff_requirements_from_flow(patient_flow_df)
    scheduled_df = _build_staff_summary(work_df, prefix="scheduled")
    actual_df = _build_staff_summary(_build_actual_work_df(work_df, abs_df), prefix="actual")
    biometrics_df = _build_biometrics_segment_features(work_df, bio_df, employees_df)
    absence_df = _build_absence_features(work_df, abs_df)

    merged = patient_flow_df.merge(calendar_df, on=["date", "is_holiday", "is_weekend"], how="left")
    merged = merged.merge(requirement_df, on=["date", "shift", "clinical_unit"], how="left")
    merged = merged.merge(scheduled_df, on=["date", "shift", "clinical_unit"], how="left")
    merged = merged.merge(actual_df, on=["date", "shift", "clinical_unit"], how="left")
    merged = merged.merge(biometrics_df, on=["date", "shift", "clinical_unit"], how="left")
    merged = merged.merge(absence_df, on=["date", "shift", "clinical_unit"], how="left")
    merged = merged.fillna(0)

    merged["deficit_count_total"] = (merged["required_headcount_total"] - merged["actual_headcount_total"]).clip(lower=0)
    merged["has_deficit_total"] = (merged["deficit_count_total"] > 0).astype(int)
    merged["station_pressure_index"] = merged["active_care_stations"] / np.maximum(merged["forecast_required_headcount_total"], 1)
    merged["procedural_pressure_index"] = merged["scheduled_procedures"] / np.maximum(merged["forecast_required_role_medico"], 1)
    merged["respiratory_pressure_index"] = merged["respiratory_case_ratio"] * (1 + merged["respiratory_alert_level"])

    for role in CLINIC_CRITICAL_ROLE_TARGETS:
        required_col = f"required_role_{role}"
        actual_col = f"actual_role_{role}"
        deficit_col = f"deficit_role_{role}"
        has_deficit_col = f"has_deficit_role_{role}"
        if actual_col not in merged.columns:
            merged[actual_col] = 0
        merged[deficit_col] = (merged[required_col] - merged[actual_col]).clip(lower=0)
        merged[has_deficit_col] = (merged[deficit_col] > 0).astype(int)

    return _add_temporal_and_lag_features(merged)


def run_clinic_etl(employees_path: Path | None = None) -> pd.DataFrame:
    logger.info("[clinic] Ejecutando ETL")
    employees_df = _read_csv("clinic_employees", parse_dates=["hire_date"], path=employees_path)
    calendar_df = _read_csv("clinic_calendar", parse_dates=["date"])
    patient_flow_df = _read_csv("clinic_patient_flow", parse_dates=["date"])
    work_df = _read_csv("clinic_work_records", parse_dates=["date"])
    bio_df = _read_csv("clinic_biometrics", parse_dates=["date"])
    abs_df = _read_csv("clinic_absenteeism", parse_dates=["date"])

    ml_features_df = build_clinic_ml_features(employees_df, calendar_df, patient_flow_df, work_df, bio_df, abs_df)
    CLINIC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ml_features_df.to_csv(CLINIC_PROCESSED_DIR / "clinic_ml_features.csv", index=False)

    with get_connection(CLINIC_DB_PATH) as conn:
        load_dataframe(employees_df, "clinic_employees", conn)
        load_dataframe(calendar_df, "clinic_calendar", conn)
        load_dataframe(patient_flow_df, "clinic_patient_flow", conn)
        load_dataframe(work_df, "clinic_work_records", conn)
        load_dataframe(bio_df, "clinic_biometrics", conn)
        load_dataframe(abs_df, "clinic_absenteeism", conn)
        load_dataframe(ml_features_df, "clinic_ml_features", conn)

    logger.info("[clinic] ETL completado: %d filas en clinic_ml_features", len(ml_features_df))
    return ml_features_df
