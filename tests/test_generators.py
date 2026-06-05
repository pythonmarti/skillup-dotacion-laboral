"""Tests para los generadores de datos sinteticos."""

import sys
import shutil
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


class TestGenerateMedicalForms:
    def test_generates_one_pdf_per_employee(self, tmp_path):
        from src.generators.employees import generate_employees
        from src.generators.medical_forms import generate_medical_forms

        employees_df = generate_employees(n=2, seed=42)
        generated_files = generate_medical_forms(employees_df, tmp_path)

        assert len(generated_files) == 2
        assert all(path.exists() for path in generated_files)
        assert all(path.suffix == ".pdf" for path in generated_files)

    def test_pdf_has_expected_name_and_content(self, tmp_path):
        from src.generators.employees import generate_employees
        from src.generators.medical_forms import generate_medical_forms

        employees_df = generate_employees(n=1, seed=42)
        generated_files = generate_medical_forms(employees_df, tmp_path)

        expected_name = f"{employees_df.iloc[0]['employee_id']}_medical_form.pdf"
        assert generated_files[0].name == expected_name
        assert generated_files[0].stat().st_size > 1_000

    def test_structured_page_is_optional(self, tmp_path):
        from src.generators.employees import generate_employees
        from src.generators.medical_forms import generate_medical_forms

        employees_df = generate_employees(n=1, seed=42)
        plain_file = generate_medical_forms(employees_df, tmp_path / "plain", include_structured_page=False)[0]
        structured_file = generate_medical_forms(employees_df, tmp_path / "structured", include_structured_page=True)[0]

        assert structured_file.stat().st_size > plain_file.stat().st_size

    def test_raises_when_required_columns_are_missing(self, tmp_path):
        from src.generators.medical_forms import generate_medical_forms

        incomplete_df = pd.DataFrame([{"employee_id": "EMP_001", "name": "Persona"}])

        with pytest.raises(ValueError, match="Faltan columnas requeridas"):
            generate_medical_forms(incomplete_df, tmp_path)


class TestExtractMedicalForms:
    def test_extract_generated_pdfs_to_dataframe(self, tmp_path):
        if shutil.which("pdftotext") is None:
            pytest.skip("pdftotext no esta disponible")

        from src.generators.employees import generate_employees
        from src.generators.medical_forms import generate_medical_forms
        from src.extraction.medical_forms import extract_medical_forms_to_dataframe

        employees_df = generate_employees(n=3, seed=42)
        pdf_paths = generate_medical_forms(employees_df, tmp_path, include_structured_page=False)
        extracted_df = extract_medical_forms_to_dataframe(pdf_paths)

        pd.testing.assert_frame_equal(
            extracted_df.reset_index(drop=True),
            employees_df[[
                "employee_id", "name", "age", "gender", "bmi", "education_level",
                "plant_area", "position", "seniority_years", "shift_pattern",
                "distance_to_work_km", "children", "social_drinker", "smoker", "hire_date",
            ]].reset_index(drop=True),
        )

    def test_extract_generated_pdfs_to_csv(self, tmp_path):
        if shutil.which("pdftotext") is None:
            pytest.skip("pdftotext no esta disponible")

        from src.generators.employees import generate_employees
        from src.generators.medical_forms import generate_medical_forms
        from src.extraction.medical_forms import extract_medical_forms_dir_to_csv

        forms_dir = tmp_path / "forms"
        output_csv = tmp_path / "employees_from_forms.csv"
        employees_df = generate_employees(n=2, seed=123)
        generate_medical_forms(employees_df, forms_dir, include_structured_page=False)

        extracted_df = extract_medical_forms_dir_to_csv(forms_dir, output_csv)

        assert output_csv.exists()
        assert len(extracted_df) == 2

    def test_validate_employees_csv_match(self, tmp_path):
        from src.extraction.medical_forms import validate_employees_csv_match

        reference = tmp_path / "reference.csv"
        candidate = tmp_path / "candidate.csv"
        df = pd.DataFrame([
            {
                "employee_id": "EMP_001",
                "name": "Persona Uno",
                "age": 30,
                "gender": "F",
                "bmi": 25.1,
                "education_level": 3,
                "plant_area": "oficinas",
                "position": "analista",
                "seniority_years": 2,
                "shift_pattern": "fijo",
                "distance_to_work_km": 10.0,
                "children": 1,
                "social_drinker": False,
                "smoker": False,
                "hire_date": "2024-01-01",
            }
        ])
        df.to_csv(reference, index=False)
        df.to_csv(candidate, index=False)

        validate_employees_csv_match(reference, candidate)

    def test_validate_employees_csv_match_detects_difference(self, tmp_path):
        from src.extraction.medical_forms import validate_employees_csv_match

        reference = tmp_path / "reference.csv"
        candidate = tmp_path / "candidate.csv"
        reference_df = pd.DataFrame([
            {
                "employee_id": "EMP_001",
                "name": "Persona Uno",
                "age": 30,
                "gender": "F",
                "bmi": 25.1,
                "education_level": 3,
                "plant_area": "oficinas",
                "position": "analista",
                "seniority_years": 2,
                "shift_pattern": "fijo",
                "distance_to_work_km": 10.0,
                "children": 1,
                "social_drinker": False,
                "smoker": False,
                "hire_date": "2024-01-01",
            }
        ])
        candidate_df = reference_df.copy()
        candidate_df.loc[0, "age"] = 31
        reference_df.to_csv(reference, index=False)
        candidate_df.to_csv(candidate, index=False)

        with pytest.raises(ValueError, match="no coincide exactamente"):
            validate_employees_csv_match(reference, candidate)
