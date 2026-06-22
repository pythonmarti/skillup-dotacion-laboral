"""Inferencia operativa del dominio clinic ambulatorio."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from config.clinic_settings import CLINIC_CRITICAL_ROLE_TARGETS, CLINIC_DB_PATH, CLINIC_PROCESSED_DIR
from src.models.shap_analysis import generate_shap_artifacts
from src.models.staffing_models import evaluate_classification, evaluate_regression
from src.utils.database import query_to_dataframe


def _load_artifacts():
    metadata = json.loads((CLINIC_PROCESSED_DIR / "clinic_model_artifacts.json").read_text(encoding="utf-8"))
    artifacts = {
        "metadata": metadata,
        "regressor": joblib.load(CLINIC_PROCESSED_DIR / "clinic_headcount_regressor.pkl"),
        "xgboost": joblib.load(CLINIC_PROCESSED_DIR / "clinic_deficit_classifier_xgboost.pkl"),
        "calibrated": joblib.load(CLINIC_PROCESSED_DIR / "clinic_deficit_classifier_calibrated.pkl"),
        "roles": {},
    }
    for role in CLINIC_CRITICAL_ROLE_TARGETS:
        path = CLINIC_PROCESSED_DIR / f"clinic_role_{role}_classifier.pkl"
        if path.exists():
            artifacts["roles"][role] = joblib.load(path)
    return artifacts


def _prepare_matrix(feature_names: list[str]):
    df = query_to_dataframe("SELECT * FROM clinic_ml_features", db_path=CLINIC_DB_PATH)
    df["date"] = pd.to_datetime(df["date"])
    categorical = [col for col in ["shift", "clinical_unit", "season"] if col in df.columns]
    encoded = pd.get_dummies(df.copy(), columns=categorical, drop_first=False, dtype=float)
    drop_cols = [
        col for col in encoded.columns
        if col in {
            "date",
            "actual_patient_volume",
            "required_headcount_total",
            "scheduled_headcount_total",
            "actual_headcount_total",
            "deficit_count_total",
            "has_deficit_total",
            "absent_count_total",
            "short_notice_absent_count",
            "absentee_rate",
            "short_notice_absentee_rate",
            "holiday_name",
        }
        or col.startswith("required_role_")
        or col.startswith("actual_")
        or col.startswith("deficit_role_")
        or col.startswith("has_deficit_role_")
        or col.startswith("scheduled_role_")
    ]
    X = encoded.drop(columns=drop_cols)
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        X = X.drop(columns=object_cols)
    X = X.fillna(0)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    extra_cols = [col for col in X.columns if col not in feature_names]
    if extra_cols:
        X = X.drop(columns=extra_cols)
    return df, X[feature_names]


def _build_recommendation(row: pd.Series) -> str:
    role_scores = {role: row.get(f"predicted_role_deficit_prob_{role}", 0.0) for role in CLINIC_CRITICAL_ROLE_TARGETS}
    top_role = max(role_scores, key=role_scores.get)
    prob = float(row["predicted_deficit_probability"])
    if prob >= 0.82:
        return f"Activar contingencia ambulatoria y reforzar {top_role} en {row['clinical_unit']} turno {row['shift']}"
    if prob >= 0.67:
        return f"Reasignar staff flotante y revisar cobertura critica de {top_role}"
    if prob >= 0.52:
        return f"Confirmar reemplazos preventivos y monitorear ausentismo de {top_role}"
    return "Cobertura estable; mantener monitoreo de agenda y carga asistencial"


def run_clinic_inference(create_shap_outputs: bool = False):
    artifacts = _load_artifacts()
    metadata = artifacts["metadata"]
    base_df, X = _prepare_matrix(metadata["feature_names"])

    pred_headcount = artifacts["regressor"].predict(X)
    prob_xgb = artifacts["xgboost"].predict_proba(X)[:, 1]
    prob_cal = artifacts["calibrated"].predict_proba(X)[:, 1]
    best_name = metadata["best_classifier"]["name"]
    best_prob = prob_xgb if best_name == "clinic_xgboost" else prob_cal
    best_threshold = float(metadata["best_classifier"]["threshold"])

    scored = base_df[[
        "date",
        "shift",
        "clinical_unit",
        "forecast_patient_volume",
        "scheduled_procedures",
        "active_care_stations",
        "forecast_required_headcount_total",
        "required_headcount_total",
        "actual_headcount_total",
        "has_deficit_total",
    ]].copy()
    scored["predicted_headcount_total"] = pred_headcount
    scored["predicted_deficit_prob_xgboost"] = prob_xgb
    scored["predicted_deficit_prob_calibrated"] = prob_cal
    scored["predicted_deficit_probability"] = best_prob
    scored["predicted_has_deficit"] = (best_prob >= best_threshold).astype(int)
    scored["best_classifier_model"] = best_name

    split_date = metadata.get("split_date")
    if split_date:
        test_mask = pd.to_datetime(base_df["date"]) >= pd.Timestamp(split_date)
    else:
        test_mask = slice(None)

    role_metrics = {}
    for role, model in artifacts["roles"].items():
        probs = model.predict_proba(X)[:, 1]
        scored[f"predicted_role_deficit_prob_{role}"] = probs
        threshold = float(metadata["thresholds"]["role_thresholds"][role])
        scored[f"predicted_role_has_deficit_{role}"] = (probs >= threshold).astype(int)
        if f"has_deficit_role_{role}" in base_df.columns:
            role_metrics[role] = evaluate_classification(
                base_df.loc[test_mask, f"has_deficit_role_{role}"].astype(int),
                probs[test_mask],
                threshold=threshold,
            )

    scored["recommended_action"] = scored.apply(_build_recommendation, axis=1)

    if create_shap_outputs:
        generate_shap_artifacts(
            domain="clinic",
            prefix="clinic",
            output_dir=CLINIC_PROCESSED_DIR,
            X=X,
            context_df=scored,
            risk_model=artifacts["xgboost"],
            headcount_model=artifacts["regressor"],
            risk_priority_scores=scored["predicted_deficit_probability"],
            segment_cols=["clinical_unit", "shift"],
        )

    metrics = {
        "regression": evaluate_regression(base_df.loc[test_mask, "actual_headcount_total"], pred_headcount[test_mask]),
        "classification_xgboost": evaluate_classification(base_df.loc[test_mask, "has_deficit_total"].astype(int), prob_xgb[test_mask], threshold=float(metadata["thresholds"]["clinic_xgboost"])),
        "classification_calibrated": evaluate_classification(base_df.loc[test_mask, "has_deficit_total"].astype(int), prob_cal[test_mask], threshold=float(metadata["thresholds"]["clinic_calibrated"])),
        "classification_best": evaluate_classification(base_df.loc[test_mask, "has_deficit_total"].astype(int), best_prob[test_mask], threshold=best_threshold),
        "role_models": role_metrics,
    }
    return scored, metrics, metadata
