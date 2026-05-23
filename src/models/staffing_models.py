"""Modelos y funciones de entrenamiento para predicción de dotación laboral."""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_headcount_regressor(X_train, y_train_reg):
    """Modelo 1: Regresor de Headcount para predecir la dotación disponible.
    
    Usa RandomForestRegressor.
    """
    logger.info("Entrenando Modelo 1: Regresor de Headcount (RandomForest)...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regressor.fit(X_train, y_train_reg)
    return regressor


def train_deficit_classifier(X_train, y_train_class):
    """Modelo 2: Clasificador de Riesgo de Déficit.
    
    Usa XGBClassifier para estimar la probabilidad directa de déficit.
    """
    logger.info("Entrenando Modelo 2: Clasificador de Déficit (XGBoost)...")
    classifier = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    classifier.fit(X_train, y_train_class)
    return classifier


def train_calibrated_ensemble(X_train, y_train_class):
    """Modelo 3: Ensamble Calibrado.
    
    Envuelve un clasificador XGBoost en un CalibratedClassifierCV usando
    calibración isotónica para obtener estimados de probabilidad confiables.
    """
    logger.info("Entrenando Modelo 3: Ensamble Calibrado (Calibrated XGBoost)...")
    base_clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    # Calibrar usando validación cruzada interna de 3 folds
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="isotonic",
        cv=3
    )
    calibrated_clf.fit(X_train, y_train_class)
    return calibrated_clf


def evaluate_regression(y_true, y_pred):
    """Evalúa métricas de regresión para el headcount."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_classification(y_true, y_prob, threshold=0.5):
    """Evalúa métricas de clasificación para el déficit."""
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUC-ROC": auc,
        "PR-AUC": ap,
        "Brier Score": brier,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }
