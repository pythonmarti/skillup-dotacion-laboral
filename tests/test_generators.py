"""Tests para los generadores de datos sinteticos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from config.settings import NUM_EMPLOYEES, DAYS_TO_SIMULATE


class TestGenerateEmployees:
    def test_row_count(self):
        from src.generators.employees import generate_employees
        df = generate_employees(n=NUM_EMPLOYEES, seed=42)
        assert len(df) == NUM_EMPLOYEES

    def test_columns(self):
        from src.generators.employees import generate_employees
        df = generate_employees(n=NUM_EMPLOYEES, seed=42)
        expected_cols = {
            "employee_id", "name", "age", "gender", "bmi",
            "education_level", "plant_area", "position",
            "seniority_years", "shift_pattern",
            "distance_to_work_km", "children",
            "social_drinker", "smoker", "hire_date",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_age_range(self):
        from src.generators.employees import generate_employees
        df = generate_employees(n=NUM_EMPLOYEES, seed=42)
        assert df["age"].min() >= 22
        assert df["age"].max() <= 62

    def test_bmi_range(self):
        from src.generators.employees import generate_employees
        df = generate_employees(n=NUM_EMPLOYEES, seed=42)
        assert df["bmi"].min() >= 18
        assert df["bmi"].max() <= 45


class TestGenerateWorkRecords:
    def test_row_count(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        df = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        # ~200 * 180 con dias de descanso -> ~36K aprox, pero al menos 25K
        assert len(df) > 25_000
        assert len(df) < 40_000

    def test_columns(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        df = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        assert "employee_id" in df.columns
        assert "date" in df.columns


class TestGenerateBiometrics:
    def test_shapes(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        from src.generators.biometrics import generate_biometrics
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        work = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        df = generate_biometrics(emps, work, seed=42)
        assert len(df) > 0
        assert "employee_id" in df.columns

    def test_hr_range(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        from src.generators.biometrics import generate_biometrics
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        work = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        df = generate_biometrics(emps, work, seed=42)
        if "hr_mean_bpm" in df.columns:
            valid = df["hr_mean_bpm"].dropna()
            assert valid.min() >= 30
            assert valid.max() <= 220

    def test_spo2_range(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        from src.generators.biometrics import generate_biometrics
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        work = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        df = generate_biometrics(emps, work, seed=42)
        if "spo2_mean_pct" in df.columns:
            valid = df["spo2_mean_pct"].dropna()
            assert valid.min() >= 70
            assert valid.max() <= 100


class TestGenerateAbsenteeism:
    def test_events_exist(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        from src.generators.absenteeism import generate_absenteeism
        from src.generators.biometrics import generate_biometrics
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        work = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        bio = generate_biometrics(emps, work, seed=42)
        df = generate_absenteeism(emps, bio, work, seed=42)
        assert len(df) > 0

    def test_valid_employee_ids(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        from src.generators.absenteeism import generate_absenteeism
        from src.generators.biometrics import generate_biometrics
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        work = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        bio = generate_biometrics(emps, work, seed=42)
        df = generate_absenteeism(emps, bio, work, seed=42)
        valid_ids = set(emps["employee_id"])
        assert set(df["employee_id"]).issubset(valid_ids)

    def test_hours_positive(self):
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        from src.generators.absenteeism import generate_absenteeism
        from src.generators.biometrics import generate_biometrics
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        work = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        bio = generate_biometrics(emps, work, seed=42)
        df = generate_absenteeism(emps, bio, work, seed=42)
        if "absence_hours" in df.columns:
            assert (df["absence_hours"] > 0).all()


class TestCausalRelationship:
    def test_high_bmi_more_absences(self):
        """Empleados con BMI>30 deberian tener mas ausencias (relacion causal)."""
        from src.generators.employees import generate_employees
        from src.generators.work_records import generate_work_records
        from src.generators.absenteeism import generate_absenteeism
        from src.generators.biometrics import generate_biometrics
        emps = generate_employees(n=NUM_EMPLOYEES, seed=42)
        work = generate_work_records(emps, days=DAYS_TO_SIMULATE, seed=42)
        bio = generate_biometrics(emps, work, seed=42)
        absences = generate_absenteeism(emps, bio, work, seed=42)

        absence_counts = absences.groupby("employee_id").size().reset_index(name="n_absences")
        merged = emps.merge(absence_counts, on="employee_id", how="left")
        merged["n_absences"] = merged["n_absences"].fillna(0)

        high_bmi = merged[merged["bmi"] > 30]["n_absences"].mean()
        low_bmi = merged[merged["bmi"] <= 30]["n_absences"].mean()

        # Empleados con BMI alto deberian tener mas ausencias en promedio
        assert high_bmi > low_bmi, (
            f"Esperado: BMI>30 ({high_bmi:.2f}) > BMI<=30 ({low_bmi:.2f})"
        )
