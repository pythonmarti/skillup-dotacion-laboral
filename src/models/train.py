"""Entrenamiento de modelos predictivos con tecnicas avanzadas."""

import warnings
import joblib
import numpy as np
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config.settings import PROCESSED_DIR, MODEL_CONFIG
from src.models.balancing import create_balanced_pipeline

warnings.filterwarnings("ignore", category=UserWarning)


def _get_cv(seed=42, temporal=False):
    """Retorna el objeto CV segun la estrategia configurada."""
    n_splits = MODEL_CONFIG["cv_splits"]
    if temporal:
        return TimeSeriesSplit(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def train_random_forest(X, y, seed=42, temporal=False):
    """Entrena Random Forest con RandomizedSearchCV.

    Returns
    -------
    RandomForestClassifier
        Mejor modelo encontrado.
    """
    param_distributions = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "class_weight": ["balanced"],
    }

    cv = _get_cv(seed, temporal)

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=seed),
        param_distributions=param_distributions,
        n_iter=MODEL_CONFIG["n_iter_search"],
        cv=cv,
        scoring=MODEL_CONFIG["scoring"],
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)

    print(f"[Random Forest] Best F1 (CV): {search.best_score_:.4f}")
    print(f"[Random Forest] Best params: {search.best_params_}")

    return search.best_estimator_


def train_gradient_boosting(X, y, seed=42, temporal=False):
    """Entrena Gradient Boosting con RandomizedSearchCV y sample_weight balanceado.

    Returns
    -------
    GradientBoostingClassifier
        Mejor modelo encontrado.
    """
    param_distributions = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
    }

    sample_weights = compute_sample_weight("balanced", y)
    cv = _get_cv(seed, temporal)

    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=seed),
        param_distributions=param_distributions,
        n_iter=MODEL_CONFIG["n_iter_search"],
        cv=cv,
        scoring=MODEL_CONFIG["scoring"],
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y, sample_weight=sample_weights)

    print(f"[Gradient Boosting] Best F1 (CV): {search.best_score_:.4f}")
    print(f"[Gradient Boosting] Best params: {search.best_params_}")

    return search.best_estimator_


def train_xgboost(X, y, seed=42, temporal=False):
    """Entrena XGBoost con scale_pos_weight automatico.

    Returns
    -------
    XGBClassifier
        Mejor modelo encontrado.
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    cv = _get_cv(seed, temporal)

    search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        ),
        param_distributions=param_distributions,
        n_iter=MODEL_CONFIG["n_iter_search"],
        cv=cv,
        scoring=MODEL_CONFIG["scoring"],
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)

    print(f"[XGBoost] Best F1 (CV): {search.best_score_:.4f}")
    print(f"[XGBoost] scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"[XGBoost] Best params: {search.best_params_}")

    return search.best_estimator_


def train_lightgbm(X, y, seed=42, temporal=False):
    """Entrena LightGBM con is_unbalance=True.

    Returns
    -------
    LGBMClassifier
        Mejor modelo encontrado.
    """
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_samples": [5, 10, 20, 30],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 0.01, 0.1, 1.0],
    }

    cv = _get_cv(seed, temporal)

    search = RandomizedSearchCV(
        LGBMClassifier(
            is_unbalance=True,
            random_state=seed,
            verbose=-1,
        ),
        param_distributions=param_distributions,
        n_iter=MODEL_CONFIG["n_iter_search"],
        cv=cv,
        scoring=MODEL_CONFIG["scoring"],
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)

    print(f"[LightGBM] Best F1 (CV): {search.best_score_:.4f}")
    print(f"[LightGBM] Best params: {search.best_params_}")

    return search.best_estimator_


def train_stacking_ensemble(X, y, seed=42, temporal=False):
    """Stacking Ensemble: RF + XGBoost + LightGBM con meta-learner LogisticRegression.

    Returns
    -------
    StackingClassifier
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Stacking requiere StratifiedKFold (cross_val_predict necesita particiones)
    stacking_cv = StratifiedKFold(
        n_splits=MODEL_CONFIG["cv_splits"], shuffle=True, random_state=seed
    )

    base_learners = [
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=seed, eval_metric="logloss", verbosity=0,
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            is_unbalance=True, random_state=seed, verbose=-1,
        )),
    ]

    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed,
        ),
        stack_method="predict_proba",
        cv=stacking_cv,
        n_jobs=-1,
    )
    stacking.fit(X, y)
    print("[Stacking] Ensemble entrenado (RF + XGBoost + LightGBM -> LogReg)")

    return stacking


def calibrate_model(model, X, y, method="isotonic", seed=42):
    """Calibra probabilidades del modelo usando CalibratedClassifierCV.

    Parameters
    ----------
    model : sklearn estimator (ya entrenado)
    X : array-like
    y : array-like
    method : str
        "isotonic" o "sigmoid"
    seed : int

    Returns
    -------
    CalibratedClassifierCV
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    calibrated = CalibratedClassifierCV(
        model, method=method, cv=cv,
    )
    calibrated.fit(X, y)
    return calibrated


def train_and_compare(X, y, seed=42, temporal=False):
    """Entrena 5 modelos y guarda artefactos.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target variable.
    seed : int
    temporal : bool
        Si True, usa TimeSeriesSplit en vez de StratifiedKFold.

    Returns
    -------
    dict
        Keys: models (dict of name->model), probs (dict of name->probs)
    """
    models_to_train = MODEL_CONFIG["models_to_train"]

    train_funcs = {
        "Random Forest": train_random_forest,
        "Gradient Boosting": train_gradient_boosting,
        "XGBoost": train_xgboost,
        "LightGBM": train_lightgbm,
        "Stacking": train_stacking_ensemble,
    }

    models = {}
    probs = {}

    for name in models_to_train:
        if name not in train_funcs:
            print(f"  Modelo desconocido: {name}, saltando...")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Entrenando {name}...")
        print(f"{'=' * 60}")

        model = train_funcs[name](X, y, seed=seed, temporal=temporal)

        # Obtener probabilidades en train para threshold optimization
        print(f"  Generando probabilidades en train para {name}...")
        model_probs = model.predict_proba(X)[:, 1]

        models[name] = model
        probs[name] = model_probs

        # Guardar modelo
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, PROCESSED_DIR / f"{safe_name}_model.pkl")

    print(f"\nModelos guardados en {PROCESSED_DIR}")

    return {"models": models, "probs": probs}
