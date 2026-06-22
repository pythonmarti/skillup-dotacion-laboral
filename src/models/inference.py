"""Inferencia sobre features de dotacion usando artefactos entrenados."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from config.settings import PROCESSED_DIR
from src.models.shap_analysis import generate_shap_artifacts
from src.models.staffing_models import evaluate_classification, evaluate_regression
from src.utils.database import query_to_dataframe


def load_model_artifacts(processed_dir: Path | None = None) -> dict:
    processed_dir = processed_dir or PROCESSED_DIR
    metadata_path = processed_dir / "model_artifacts.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "regressor": joblib.load(processed_dir / "headcount_regressor.pkl"),
        "classifier_xgboost": joblib.load(processed_dir / "deficit_classifier_xgboost.pkl"),
        "classifier_calibrated": joblib.load(processed_dir / "deficit_classifier_calibrated.pkl"),
    }


def _prepare_inference_matrix(db_path=None, feature_names: list[str] | None = None):
    df = query_to_dataframe("SELECT * FROM ml_features", db_path=db_path)
    df["date"] = pd.to_datetime(df["date"])

    context_cols = [
        col for col in [
            "plant_area", "shift", "date", "required_headcount", "actual_headcount",
            "deficit_count", "has_deficit", "absentee_rate",
        ] if col in df.columns
    ]
    context_df = df[context_cols].copy()

    model_df = pd.get_dummies(df.copy(), columns=[c for c in ["plant_area", "shift"] if c in df.columns], drop_first=False, dtype=float)
    drop_cols = [c for c in ["date", "has_deficit", "actual_headcount", "deficit_count", "absentee_rate", "scheduled_headcount"] if c in model_df.columns]
    X = model_df.drop(columns=drop_cols)
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X = X.drop(columns=obj_cols)
    X = X.fillna(0)

    if feature_names is not None:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        extra_cols = [col for col in X.columns if col not in feature_names]
        if extra_cols:
            X = X.drop(columns=extra_cols)
        X = X[feature_names]

    return context_df, X, df


def run_inference(
    db_path=None,
    processed_dir: Path | None = None,
    create_shap_outputs: bool = False,
    shap_domain: str = "industrial",
    shap_prefix: str = "industrial",
    shap_segment_cols: list[str] | None = None,
):
    artifacts = load_model_artifacts(processed_dir=processed_dir)
    metadata = artifacts["metadata"]
    feature_names = metadata["feature_names"]
    context_df, X, raw_df = _prepare_inference_matrix(db_path=db_path, feature_names=feature_names)
    processed_dir = processed_dir or PROCESSED_DIR

    regressor = artifacts["regressor"]
    classifier_xgboost = artifacts["classifier_xgboost"]
    classifier_calibrated = artifacts["classifier_calibrated"]

    pred_headcount = regressor.predict(X)
    prob_xgboost = classifier_xgboost.predict_proba(X)[:, 1]
    prob_calibrated = classifier_calibrated.predict_proba(X)[:, 1]

    th_xgboost = metadata["thresholds"]["Modelo 2 (XGBoost)"]
    th_calibrated = metadata["thresholds"]["Modelo 3 (Calibrado)"]
    best_meta = metadata["best_classifier"]
    best_prob = prob_xgboost if best_meta["name"] == "Modelo 2 (XGBoost)" else prob_calibrated
    best_threshold = best_meta["threshold"]

    scored_df = context_df.copy()
    scored_df["predicted_headcount"] = pred_headcount
    scored_df["predicted_deficit_prob_xgboost"] = prob_xgboost
    scored_df["predicted_deficit_prob_calibrated"] = prob_calibrated
    scored_df["predicted_has_deficit_xgboost"] = (prob_xgboost >= th_xgboost).astype(int)
    scored_df["predicted_has_deficit_calibrated"] = (prob_calibrated >= th_calibrated).astype(int)
    scored_df["predicted_deficit_probability"] = best_prob
    scored_df["predicted_has_deficit"] = (best_prob >= best_threshold).astype(int)
    scored_df["best_classifier_model"] = best_meta["name"]

    if create_shap_outputs:
        shap_context = scored_df.copy()
        generate_shap_artifacts(
            domain=shap_domain,
            prefix=shap_prefix,
            output_dir=processed_dir,
            X=X,
            context_df=shap_context,
            risk_model=classifier_xgboost,
            headcount_model=regressor,
            risk_priority_scores=scored_df["predicted_deficit_probability"],
            segment_cols=shap_segment_cols or ["plant_area", "shift"],
        )

    metrics = {}
    split_date = metadata.get("split_date")
    if split_date:
        test_mask = pd.to_datetime(raw_df["date"]) >= pd.Timestamp(split_date)
    else:
        test_mask = slice(None)

    if "actual_headcount" in raw_df.columns:
        metrics["regression"] = evaluate_regression(
            raw_df.loc[test_mask, "actual_headcount"], pred_headcount[test_mask]
        )
    if "has_deficit" in raw_df.columns:
        metrics["classification_xgboost"] = evaluate_classification(
            raw_df.loc[test_mask, "has_deficit"], prob_xgboost[test_mask], threshold=th_xgboost
        )
        metrics["classification_calibrated"] = evaluate_classification(
            raw_df.loc[test_mask, "has_deficit"], prob_calibrated[test_mask], threshold=th_calibrated
        )
        metrics["classification_best"] = metrics[
            "classification_xgboost" if best_meta["name"] == "Modelo 2 (XGBoost)" else "classification_calibrated"
        ]

    return scored_df, metrics, metadata
