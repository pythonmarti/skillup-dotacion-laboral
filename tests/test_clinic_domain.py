"""Tests del dominio clinic."""

from src.clinic.etl import build_clinic_ml_features
from src.clinic.generate import generate_clinic_raw_data
from src.domains.registry import get_domain_pipeline


def test_clinic_domain_registered():
    pipeline = get_domain_pipeline("clinic")
    assert pipeline.name == "clinic"
    assert pipeline.report is not None


def test_clinic_ml_features_smoke():
    raw = generate_clinic_raw_data(employees=72, days=45, seed=42, start_date="2025-05-01")
    features = build_clinic_ml_features(
        employees_df=raw["clinic_employees"],
        calendar_df=raw["clinic_calendar"],
        patient_flow_df=raw["clinic_patient_flow"],
        work_df=raw["clinic_work_records"],
        bio_df=raw["clinic_biometrics"],
        abs_df=raw["clinic_absenteeism"],
    )

    assert not features.empty
    assert {"clinical_unit", "shift", "has_deficit_total", "deficit_count_total", "has_deficit_role_medico", "has_deficit_role_enfermera", "has_deficit_role_tens"}.issubset(features.columns)
    assert features["clinical_unit"].nunique() >= 4
    assert features["has_deficit_total"].isin([0, 1]).all()
    assert features["has_deficit_total"].sum() > 0
