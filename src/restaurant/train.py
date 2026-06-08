"""Entrenamiento de modelos para el dominio restaurant."""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from config.restaurant_settings import CRITICAL_ROLE_TARGETS, RESTAURANT_DB_PATH, RESTAURANT_MODEL_CONFIG, RESTAURANT_PROCESSED_DIR
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


ROLE_MODEL_NAME_TEMPLATE = "restaurant_role_{role}_classifier.pkl"


def _prepare_restaurant_feature_matrix():
    df = query_to_dataframe("SELECT * FROM restaurant_ml_features", db_path=RESTAURANT_DB_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "service_period"]).reset_index(drop=True)

    y_reg = df["actual_headcount_total"].copy()
    y_class = df["has_deficit_total"].astype(int).copy()

    categorical = [col for col in ["service_period", "season"] if col in df.columns]
    encoded = pd.get_dummies(df.copy(), columns=categorical, drop_first=False, dtype=float)

    drop_cols = [
        "date",
        "actual_covers",
        "actual_sales",
        "required_headcount_total",
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
    ]
    drop_cols.extend(prefixed_drop_cols)

    drop_cols = [col for col in drop_cols if col in encoded.columns]
    X = encoded.drop(columns=drop_cols)
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        X = X.drop(columns=object_cols)
    X = X.fillna(0)
    return df, X, y_reg, y_class, list(X.columns)


def _select_best_classifier(class2_metrics, class3_metrics) -> str:
    candidates = {
        "restaurant_xgboost": class2_metrics,
        "restaurant_calibrated": class3_metrics,
    }
    return min(
        candidates,
        key=lambda name: (
            candidates[name]["Brier Score"],
            -candidates[name]["F1-Score"],
            -candidates[name]["AUC-ROC"],
        ),
    )


def run_restaurant_training() -> dict:
    logger.info("[restaurant] Entrenando modelos")
    RESTAURANT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base_df, X, y_reg, y_class, feature_names = _prepare_restaurant_feature_matrix()

    split_df = X.copy()
    split_df["date"] = base_df["date"].values
    split_df["_target_reg"] = y_reg.values
    split_df["_target_class"] = y_class.values
    for role in CRITICAL_ROLE_TARGETS:
        split_df[f"_target_role_{role}"] = base_df[f"has_deficit_role_{role}"].astype(int).values

    train_df, test_df = temporal_train_test_split(split_df, date_col="date", test_ratio=RESTAURANT_MODEL_CONFIG["test_ratio"])
    drop_target_cols = [col for col in train_df.columns if col.startswith("_target_")] + ["date"]
    X_train = train_df.drop(columns=drop_target_cols)
    X_test = test_df.drop(columns=drop_target_cols)
    y_train_reg = train_df["_target_reg"]
    y_test_reg = test_df["_target_reg"]
    y_train_class = train_df["_target_class"].astype(int)
    y_test_class = test_df["_target_class"].astype(int)

    regressor = train_headcount_regressor(X_train, y_train_reg)
    reg_metrics = evaluate_regression(y_test_reg, regressor.predict(X_test))

    classifier = train_deficit_classifier(X_train, y_train_class)
    th_xgb = find_optimal_threshold(y_train_class, classifier.predict_proba(X_train)[:, 1])
    class_metrics = evaluate_classification(y_test_class, classifier.predict_proba(X_test)[:, 1], threshold=th_xgb)

    calibrated = train_calibrated_ensemble(X_train, y_train_class)
    th_cal = find_optimal_threshold(y_train_class, calibrated.predict_proba(X_train)[:, 1])
    calibrated_metrics = evaluate_classification(y_test_class, calibrated.predict_proba(X_test)[:, 1], threshold=th_cal)

    role_models = {}
    role_thresholds = {}
    role_metrics = {}
    for role in CRITICAL_ROLE_TARGETS:
        y_role_train = train_df[f"_target_role_{role}"].astype(int)
        y_role_test = test_df[f"_target_role_{role}"].astype(int)
        if y_role_train.sum() < 8 or y_role_test.sum() < 3:
            logger.info("[restaurant] Saltando modelo de rol %s por baja prevalencia", role)
            continue
        role_model = train_deficit_classifier(X_train, y_role_train)
        threshold = find_optimal_threshold(y_role_train, role_model.predict_proba(X_train)[:, 1])
        metrics = evaluate_classification(y_role_test, role_model.predict_proba(X_test)[:, 1], threshold=threshold)
        role_models[role] = role_model
        role_thresholds[role] = float(threshold)
        role_metrics[role] = {key: float(value) for key, value in metrics.items()}
        joblib.dump(role_model, RESTAURANT_PROCESSED_DIR / ROLE_MODEL_NAME_TEMPLATE.format(role=role))

    joblib.dump(regressor, RESTAURANT_PROCESSED_DIR / "restaurant_headcount_regressor.pkl")
    joblib.dump(classifier, RESTAURANT_PROCESSED_DIR / "restaurant_deficit_classifier_xgboost.pkl")
    joblib.dump(calibrated, RESTAURANT_PROCESSED_DIR / "restaurant_deficit_classifier_calibrated.pkl")

    best_classifier = _select_best_classifier(class_metrics, calibrated_metrics)
    metadata = {
        "feature_names": feature_names,
        "thresholds": {
            "restaurant_xgboost": float(th_xgb),
            "restaurant_calibrated": float(th_cal),
            "role_thresholds": role_thresholds,
        },
        "best_classifier": {
            "name": best_classifier,
            "path": "restaurant_deficit_classifier_xgboost.pkl" if best_classifier == "restaurant_xgboost" else "restaurant_deficit_classifier_calibrated.pkl",
            "threshold": float(th_xgb if best_classifier == "restaurant_xgboost" else th_cal),
        },
        "metrics": {
            "regression": {key: float(value) for key, value in reg_metrics.items()},
            "restaurant_xgboost": {key: float(value) for key, value in class_metrics.items()},
            "restaurant_calibrated": {key: float(value) for key, value in calibrated_metrics.items()},
            "role_models": role_metrics,
        },
    }
    (RESTAURANT_PROCESSED_DIR / "restaurant_model_artifacts.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    metrics_df = pd.DataFrame([
        {"model": "restaurant_headcount_regressor", **metadata["metrics"]["regression"]},
        {"model": "restaurant_xgboost", **metadata["metrics"]["restaurant_xgboost"]},
        {"model": "restaurant_calibrated", **metadata["metrics"]["restaurant_calibrated"]},
    ])
    for role, metrics in role_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{"model": f"restaurant_role_{role}", **metrics}])], ignore_index=True)
    metrics_df.to_csv(RESTAURANT_PROCESSED_DIR / "restaurant_model_metrics.csv", index=False)
    logger.info("[restaurant] Modelos entrenados y artefactos guardados en %s", RESTAURANT_PROCESSED_DIR)
    return metadata
