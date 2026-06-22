"""Entrenamiento de modelos para el dominio clinic."""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from config.clinic_settings import CLINIC_CRITICAL_ROLE_TARGETS, CLINIC_DB_PATH, CLINIC_MODEL_CONFIG, CLINIC_PROCESSED_DIR
from src.models.evaluate import find_optimal_threshold
from src.models.features import temporal_train_test_split
from src.models.staffing_models import (
    evaluate_classification,
    evaluate_regression,
    train_calibrated_ensemble,
    train_deficit_classifier,
    train_headcount_regressor,
)
from src.utils.database import query_to_dataframe

logger = logging.getLogger(__name__)


ROLE_MODEL_NAME_TEMPLATE = "clinic_role_{role}_classifier.pkl"


def _prepare_clinic_feature_matrix():
    df = query_to_dataframe("SELECT * FROM clinic_ml_features", db_path=CLINIC_DB_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "shift", "clinical_unit"]).reset_index(drop=True)

    y_reg = df["actual_headcount_total"].copy()
    y_class = df["has_deficit_total"].astype(int).copy()

    categorical = [col for col in ["shift", "clinical_unit", "season"] if col in df.columns]
    encoded = pd.get_dummies(df.copy(), columns=categorical, drop_first=False, dtype=float)

    drop_cols = [
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
    ]
    prefixed_drop_cols = [
        col for col in encoded.columns
        if col.startswith("required_role_")
        or col.startswith("actual_")
        or col.startswith("deficit_role_")
        or col.startswith("has_deficit_role_")
        or col.startswith("scheduled_role_")
    ]
    drop_cols.extend(prefixed_drop_cols)
    drop_cols = [col for col in drop_cols if col in encoded.columns]

    X = encoded.drop(columns=drop_cols)
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        X = X.drop(columns=object_cols)
    X = X.fillna(0)
    return df, X, y_reg, y_class, list(X.columns)


def _select_best_classifier(class_metrics, calibrated_metrics) -> str:
    candidates = {
        "clinic_xgboost": class_metrics,
        "clinic_calibrated": calibrated_metrics,
    }
    return min(
        candidates,
        key=lambda name: (
            candidates[name]["Brier Score"],
            -candidates[name]["F1-Score"],
            -candidates[name]["AUC-ROC"],
        ),
    )


def run_clinic_training() -> dict:
    logger.info("[clinic] Entrenando modelos")
    CLINIC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base_df, X, y_reg, y_class, feature_names = _prepare_clinic_feature_matrix()

    split_df = X.copy()
    split_df["date"] = base_df["date"].values
    split_df["_target_reg"] = y_reg.values
    split_df["_target_class"] = y_class.values
    for role in CLINIC_CRITICAL_ROLE_TARGETS:
        split_df[f"_target_role_{role}"] = base_df[f"has_deficit_role_{role}"].astype(int).values

    trainval_df, test_df, split_date = temporal_train_test_split(
        split_df, date_col="date", test_ratio=CLINIC_MODEL_CONFIG["test_ratio"]
    )

    train_df, val_df, _ = temporal_train_test_split(
        trainval_df, date_col="date", test_ratio=0.2
    )

    def _Xy(sub_df):
        drop_tgts = [col for col in sub_df.columns if col.startswith("_target_")] + ["date"]
        X_out = sub_df.drop(columns=drop_tgts)
        y_reg_out = sub_df["_target_reg"]
        y_class_out = sub_df["_target_class"].astype(int)
        return X_out, y_reg_out, y_class_out

    X_train, y_train_reg, y_train_class = _Xy(train_df)
    X_val, _, y_val_class = _Xy(val_df)
    X_trainval, y_trainval_reg, y_trainval_class = _Xy(trainval_df)
    X_test, y_test_reg, y_test_class = _Xy(test_df)

    logger.info(
        "[clinic] Train: %d | Val: %d | Test: %d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )

    regressor = train_headcount_regressor(X_trainval, y_trainval_reg)
    reg_metrics = evaluate_regression(y_test_reg, regressor.predict(X_test))

    # Fase 1: train -> val para seleccion
    classifier_train = train_deficit_classifier(X_train, y_train_class)
    val_th_xgb = find_optimal_threshold(y_train_class, classifier_train.predict_proba(X_train)[:, 1])
    val_xgb_metrics = evaluate_classification(y_val_class, classifier_train.predict_proba(X_val)[:, 1], threshold=val_th_xgb)

    calibrated_train = train_calibrated_ensemble(X_train, y_train_class)
    val_th_cal = find_optimal_threshold(y_train_class, calibrated_train.predict_proba(X_train)[:, 1])
    val_cal_metrics = evaluate_classification(y_val_class, calibrated_train.predict_proba(X_val)[:, 1], threshold=val_th_cal)

    best_classifier = _select_best_classifier(val_xgb_metrics, val_cal_metrics)
    logger.info("[clinic] Mejor clasificador segun validacion: %s", best_classifier)

    # Fase 2: reentrenar en train+val y evaluar en test ciego
    classifier = train_deficit_classifier(X_trainval, y_trainval_class)
    th_xgb = find_optimal_threshold(y_trainval_class, classifier.predict_proba(X_trainval)[:, 1])
    class_metrics = evaluate_classification(y_test_class, classifier.predict_proba(X_test)[:, 1], threshold=th_xgb)

    calibrated = train_calibrated_ensemble(X_trainval, y_trainval_class)
    th_cal = find_optimal_threshold(y_trainval_class, calibrated.predict_proba(X_trainval)[:, 1])
    calibrated_metrics = evaluate_classification(y_test_class, calibrated.predict_proba(X_test)[:, 1], threshold=th_cal)

    logger.info(
        "[clinic] XGBoost  [TEST] AUC: %.4f | F1: %.4f | Brier: %.4f",
        class_metrics["AUC-ROC"], class_metrics["F1-Score"], class_metrics["Brier Score"],
    )
    logger.info(
        "[clinic] Calibrado [TEST] AUC: %.4f | F1: %.4f | Brier: %.4f",
        calibrated_metrics["AUC-ROC"], calibrated_metrics["F1-Score"], calibrated_metrics["Brier Score"],
    )

    role_models = {}
    role_thresholds = {}
    role_metrics = {}
    for role in CLINIC_CRITICAL_ROLE_TARGETS:
        y_role_train = trainval_df[f"_target_role_{role}"].astype(int)
        y_role_test = test_df[f"_target_role_{role}"].astype(int)
        if y_role_train.sum() < 8 or y_role_test.sum() < 3:
            logger.info("[clinic] Saltando modelo de rol %s por baja prevalencia", role)
            continue
        role_model = train_deficit_classifier(X_trainval, y_role_train)
        threshold = find_optimal_threshold(y_role_train, role_model.predict_proba(X_trainval)[:, 1])
        metrics = evaluate_classification(y_role_test, role_model.predict_proba(X_test)[:, 1], threshold=threshold)
        role_models[role] = role_model
        role_thresholds[role] = float(threshold)
        role_metrics[role] = {key: float(value) for key, value in metrics.items()}
        joblib.dump(role_model, CLINIC_PROCESSED_DIR / ROLE_MODEL_NAME_TEMPLATE.format(role=role))

    joblib.dump(regressor, CLINIC_PROCESSED_DIR / "clinic_headcount_regressor.pkl")
    joblib.dump(classifier, CLINIC_PROCESSED_DIR / "clinic_deficit_classifier_xgboost.pkl")
    joblib.dump(calibrated, CLINIC_PROCESSED_DIR / "clinic_deficit_classifier_calibrated.pkl")

    metadata = {
        "feature_names": feature_names,
        "split_date": split_date,
        "thresholds": {
            "clinic_xgboost": float(th_xgb),
            "clinic_calibrated": float(th_cal),
            "role_thresholds": role_thresholds,
        },
        "best_classifier": {
            "name": best_classifier,
            "path": "clinic_deficit_classifier_xgboost.pkl" if best_classifier == "clinic_xgboost" else "clinic_deficit_classifier_calibrated.pkl",
            "threshold": float(th_xgb if best_classifier == "clinic_xgboost" else th_cal),
        },
        "metrics": {
            "regression": {key: float(value) for key, value in reg_metrics.items()},
            "clinic_xgboost": {key: float(value) for key, value in class_metrics.items()},
            "clinic_calibrated": {key: float(value) for key, value in calibrated_metrics.items()},
            "role_models": role_metrics,
        },
    }
    (CLINIC_PROCESSED_DIR / "clinic_model_artifacts.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame([
        {"model": "clinic_headcount_regressor", **metadata["metrics"]["regression"]},
        {"model": "clinic_xgboost", **metadata["metrics"]["clinic_xgboost"]},
        {"model": "clinic_calibrated", **metadata["metrics"]["clinic_calibrated"]},
    ])
    for role, metrics in role_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{"model": f"clinic_role_{role}", **metrics}])], ignore_index=True)
    metrics_df.to_csv(CLINIC_PROCESSED_DIR / "clinic_model_metrics.csv", index=False)
    logger.info("[clinic] Modelos entrenados y artefactos guardados en %s", CLINIC_PROCESSED_DIR)
    return metadata
