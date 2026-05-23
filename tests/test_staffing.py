"""Tests unitarios para la funcionalidad de predicción de dotación laboral y realismo de simulación."""

import numpy as np
import pandas as pd
import pytest
from src.generators.employees import generate_employees
from src.generators.work_records import generate_work_records
from src.generators.biometrics import generate_biometrics, _generate_health_events
from src.generators.absenteeism import generate_absenteeism
from src.etl.transform import aggregate_to_staffing_level
from src.models.staffing_models import (
    train_headcount_regressor,
    train_deficit_classifier,
    train_calibrated_ensemble,
    evaluate_regression,
    evaluate_classification
)


def test_health_events_generation():
    """Verifica que _generate_health_events predetermina de forma consistente los estados de salud."""
    employees_df = pd.DataFrame({"employee_id": ["EMP001", "EMP002"]})
    days = 100
    events = _generate_health_events(employees_df, days, seed=42)
    
    assert "EMP001" in events
    assert "EMP002" in events
    
    states, reasons = events["EMP001"]
    assert len(states) == days
    assert len(reasons) == days
    # Deberían existir días normales (0) e incubation (1,2) o sick (3)
    assert set(states).issubset({0, 1, 2, 3})


def test_biometrics_degradation_sickness():
    """Verifica que los biométricos de los días enfermos e incubación muestran degradación fisiológica."""
    # Generar un set de datos mínimo
    emp_df = generate_employees(n=10, seed=42)
    work_df = generate_work_records(emp_df, days=30, seed=42)
    bio_df = generate_biometrics(emp_df, work_df, days=30, seed=42)
    
    assert "_is_sick" in bio_df.columns
    assert "_absence_reason" in bio_df.columns
    
    # Filtrar días normales vs días enfermos
    normal_days = bio_df[bio_df["_is_sick"] == 0]
    sick_days = bio_df[bio_df["_is_sick"] == 1]
    
    if len(sick_days) > 0:
        # Frecuencia cardíaca media debería ser significativamente mayor en días de enfermedad
        assert sick_days["hr_mean_bpm"].mean() > normal_days["hr_mean_bpm"].mean()
        # Sueño debería ser menor en días de enfermedad
        assert sick_days["sleep_duration_hours"].mean() < normal_days["sleep_duration_hours"].mean()
        # Stress score debería ser mayor en días de enfermedad o incubación
        assert sick_days["stress_score"].mean() > normal_days["stress_score"].mean()


def test_absenteeism_matches_sickness():
    """Verifica que el ausentismo se registre correctamente en los días enfermos prediseñados."""
    emp_df = generate_employees(n=10, seed=42)
    work_df = generate_work_records(emp_df, days=30, seed=42)
    bio_df = generate_biometrics(emp_df, work_df, days=30, seed=42)
    abs_df = generate_absenteeism(emp_df, bio_df, work_df, seed=42)
    
    # Si un día está marcado como sick y está programado, debe estar en absenteeism
    sick_scheduled = bio_df[bio_df["_is_sick"] == 1].copy()
    
    # Unir con work records para saber si estaba de descanso
    sick_scheduled["date_str"] = pd.to_datetime(sick_scheduled["date"]).dt.strftime("%Y-%m-%d")
    work_df["date_str"] = pd.to_datetime(work_df["date"]).dt.strftime("%Y-%m-%d")
    
    merged = sick_scheduled.merge(work_df, on=["employee_id", "date_str"])
    scheduled_sick = merged[merged["is_rest_day"] == 0]
    
    abs_df["date_str"] = pd.to_datetime(abs_df["date"]).dt.strftime("%Y-%m-%d")
    
    for _, row in scheduled_sick.iterrows():
        emp_id = row["employee_id"]
        date_val = row["date_str"]
        
        # Debe haber un registro en abs_df para esta combinación
        match = abs_df[(abs_df["employee_id"] == emp_id) & (abs_df["date_str"] == date_val)]
        assert len(match) == 1
        assert match.iloc[0]["is_absent"] == 1


def test_staffing_aggregation_and_deficit():
    """Verifica la agregación de dotación laboral por área/turno y el cálculo de has_deficit."""
    # Simular una tabla merged individual
    data = {
        "employee_id": ["EMP001", "EMP002", "EMP003", "EMP004"],
        "date": ["2026-05-23", "2026-05-23", "2026-05-23", "2026-05-23"],
        "shift": ["diurno", "diurno", "diurno", "diurno"],
        "plant_area": ["destilacion", "destilacion", "destilacion", "destilacion"],
        "is_rest_day": [0, 0, 0, 0],
        "is_absent": [1, 0, 0, 0],  # 1 ausente, 3 presentes
        "hr_mean_bpm": [80.0, 70.0, 75.0, 72.0],
        "hrv_rmssd_ms": [30.0, 45.0, 40.0, 50.0],
        "stress_score": [50.0, 30.0, 35.0, 25.0],
        "sleep_duration_hours": [6.0, 7.5, 8.0, 7.0]
    }
    df = pd.DataFrame(data)
    
    agg_df = aggregate_to_staffing_level(df)
    
    assert len(agg_df) == 1
    row = agg_df.iloc[0]
    
    # 4 programados, 3 reales
    assert row["scheduled_headcount"] == 4
    assert row["actual_headcount"] == 3
    # Mínimo de destilacion diurno es 10 (según config.settings.REQUIRED_STAFF)
    assert row["required_headcount"] == 10
    # Como 3 < 10, debe haber déficit
    assert row["has_deficit"] == 1
    assert row["deficit_count"] == 7
    # Debe calcular promedios correctos para biométricos
    assert row["hr_mean_bpm"] == np.mean([80.0, 70.0, 75.0, 72.0])
