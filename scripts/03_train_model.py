"""Script de entrenamiento y evaluación de modelos predictivos de dotación laboral."""

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from config.settings import MODEL_CONFIG, PROCESSED_DIR
from src.models.features import prepare_feature_matrix, temporal_train_test_split
from src.models.staffing_models import (
    train_headcount_regressor,
    train_deficit_classifier,
    train_calibrated_ensemble,
    evaluate_regression,
    evaluate_classification,
)
from src.models.evaluate import (
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
)

logger = logging.getLogger(__name__)


def _select_best_classifier(class2_metrics, class3_metrics) -> str:
    candidates = {
        "Modelo 2 (XGBoost)": class2_metrics,
        "Modelo 3 (Calibrado)": class3_metrics,
    }
    return min(
        candidates,
        key=lambda name: (
            candidates[name]["Brier Score"],
            -candidates[name]["F1-Score"],
            -candidates[name]["AUC-ROC"],
        ),
    )


def _save_model_artifacts(
    feature_names,
    regressor,
    classifier_df,
    calibrated_model,
    optimal_th_c2,
    optimal_th_c3,
    reg_metrics,
    class2_metrics,
    class3_metrics,
    split_date,
    best_classifier_name,
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, PROCESSED_DIR / "headcount_regressor.pkl")
    joblib.dump(classifier_df, PROCESSED_DIR / "deficit_classifier_xgboost.pkl")
    joblib.dump(calibrated_model, PROCESSED_DIR / "deficit_classifier_calibrated.pkl")

    best_classifier_path = (
        "deficit_classifier_xgboost.pkl"
        if best_classifier_name == "Modelo 2 (XGBoost)"
        else "deficit_classifier_calibrated.pkl"
    )

    metadata = {
        "feature_names": list(feature_names),
        "split_date": split_date,
        "thresholds": {
            "Modelo 2 (XGBoost)": float(optimal_th_c2),
            "Modelo 3 (Calibrado)": float(optimal_th_c3),
        },
        "metrics": {
            "Modelo 1 (Headcount)": {k: float(v) for k, v in reg_metrics.items()},
            "Modelo 2 (XGBoost)": {k: float(v) for k, v in class2_metrics.items()},
            "Modelo 3 (Calibrado)": {k: float(v) for k, v in class3_metrics.items()},
        },
        "best_classifier": {
            "name": best_classifier_name,
            "path": best_classifier_path,
            "threshold": float(
                optimal_th_c2 if best_classifier_name == "Modelo 2 (XGBoost)" else optimal_th_c3
            ),
        },
    }

    metadata_path = PROCESSED_DIR / "model_artifacts.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Artefactos de modelo guardados en %s", PROCESSED_DIR)


def plot_regression_actual_vs_predicted(y_true, y_pred, save_path):
    """Grafica dotación real vs predicha."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="teal", edgecolors="black", linewidths=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfecto")

    ax.set_xlabel("Dotacion Disponible Real")
    ax.set_ylabel("Dotacion Disponible Predicha")
    ax.set_title("Modelo 1: Regresion de Dotacion Real vs Predicha")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico guardado: %s", save_path.name)


def plot_calibration_curve_custom(y_true, prob_dict, save_path):
    """Grafica la curva de calibración de probabilidades."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectamente calibrado")

    for name, probs in prob_dict.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=8)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)

    ax.set_ylabel("Fraccion de positivos reales")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_title("Curvas de Calibracion de Riesgo de Deficit")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico guardado: %s", save_path.name)


def main():
    logger.info("=" * 60)
    logger.info("PASO 3: Entrenamiento de Modelos de Dotacion Laboral")
    logger.info("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar features
    logger.info("[1/6] Cargando features desde SQLite...")
    data = prepare_feature_matrix()
    X_full = data["X"]
    y_full_class = data["y_class"]
    y_full_reg = data["y_reg"]
    feature_names = data["feature_names"]
    dates = data.get("date")

    logger.info(
        "  Dataset: %d registros | %d features | deficit rate: %.1f%%",
        X_full.shape[0], X_full.shape[1], y_full_class.mean() * 100,
    )

    # 2. Split temporal: train -> test
    logger.info("[2/6] Aplicando split temporal (test_ratio=%.0f%%)...", MODEL_CONFIG["test_ratio"] * 100)
    if dates is not None:
        temp_df = X_full.copy()
        temp_df["date"] = dates
        temp_df["_target_class"] = y_full_class.values
        temp_df["_target_reg"] = y_full_reg.values

        trainval_df, test_df, split_date = temporal_train_test_split(
            temp_df, date_col="date", test_ratio=MODEL_CONFIG["test_ratio"]
        )

        # Sub-split train en train + val para seleccion de modelo sin contaminar test
        train_df, val_df, _ = temporal_train_test_split(
            trainval_df, date_col="date", test_ratio=0.2
        )

        X_train = train_df.drop(columns=["date", "_target_class", "_target_reg"])
        y_train_class = train_df["_target_class"].astype(int)
        y_train_reg = train_df["_target_reg"]

        # Validation set para seleccion de mejor clasificador
        X_val = val_df.drop(columns=["date", "_target_class", "_target_reg"])
        y_val_class = val_df["_target_class"].astype(int)

        # Train+val combinado para reentrenamiento final
        X_trainval = trainval_df.drop(columns=["date", "_target_class", "_target_reg"])
        y_trainval_class = trainval_df["_target_class"].astype(int)
        y_trainval_reg = trainval_df["_target_reg"]

        X_test = test_df.drop(columns=["date", "_target_class", "_target_reg"])
        y_test_class = test_df["_target_class"].astype(int)
        y_test_reg = test_df["_target_reg"]
    else:
        raise ValueError("Se requieren fechas para realizar la validacion temporal.")

    logger.info(
        "  Train: %d reg | Val: %d reg | Test: %d reg",
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )
    logger.info(
        "  Train deficit: %.1f%% | Val deficit: %.1f%% | Test deficit: %.1f%%",
        y_train_class.mean() * 100, y_val_class.mean() * 100, y_test_class.mean() * 100,
    )

    # ==========================================
    # Modelo 1: Regresor de Headcount (train+val -> test ciego)
    # ==========================================
    logger.info("[3/6] Entrenando Modelo 1: Regresor de Headcount (RandomForest)...")
    regressor = train_headcount_regressor(X_trainval, y_trainval_reg)
    test_pred_reg = regressor.predict(X_test)
    reg_metrics = evaluate_regression(y_test_reg, test_pred_reg)

    logger.info(
        "  Modelo 1 - MAE: %.2f operadores | RMSE: %.2f | R2: %.4f",
        reg_metrics["MAE"], reg_metrics["RMSE"], reg_metrics["R2"],
    )
    plot_regression_actual_vs_predicted(
        y_test_reg, test_pred_reg,
        PROCESSED_DIR / "headcount_actual_vs_predicted.png",
    )

    # ==========================================
    # Modelo 2: Clasificador de Deficit (train -> val para seleccion, train+val -> test ciego)
    # ==========================================
    logger.info("[4/6] Entrenando Modelo 2: Clasificador de Deficit (XGBoost)...")

    # Fase 1: entrenar en train, evaluar en val para seleccion de modelo
    classifier_train = train_deficit_classifier(X_train, y_train_class)
    val_prob_c2 = classifier_train.predict_proba(X_val)[:, 1]
    val_th_c2 = find_optimal_threshold(y_train_class, classifier_train.predict_proba(X_train)[:, 1])
    val_metrics_c2 = evaluate_classification(y_val_class, val_prob_c2, threshold=val_th_c2)

    logger.info(
        "  Modelo 2 [VAL] - AUC: %.4f | F1: %.4f | Brier: %.4f",
        val_metrics_c2["AUC-ROC"], val_metrics_c2["F1-Score"], val_metrics_c2["Brier Score"],
    )

    # Fase 1: Modelo 3 tambien sobre train -> val
    logger.info("[5/6] Entrenando Modelo 3: Ensamble Calibrado (Isotonic XGBoost)...")
    calibrated_train = train_calibrated_ensemble(X_train, y_train_class)
    val_prob_c3 = calibrated_train.predict_proba(X_val)[:, 1]
    val_th_c3 = find_optimal_threshold(y_train_class, calibrated_train.predict_proba(X_train)[:, 1])
    val_metrics_c3 = evaluate_classification(y_val_class, val_prob_c3, threshold=val_th_c3)

    logger.info(
        "  Modelo 3 [VAL] - AUC: %.4f | F1: %.4f | Brier: %.4f",
        val_metrics_c3["AUC-ROC"], val_metrics_c3["F1-Score"], val_metrics_c3["Brier Score"],
    )

    # Seleccion del mejor clasificador usando solo val
    best_name = _select_best_classifier(val_metrics_c2, val_metrics_c3)
    logger.info("  Mejor clasificador segun validacion: %s", best_name)

    # Fase 2: reentrenar el mejor en train+val y evaluar en test ciego
    if best_name == "Modelo 2 (XGBoost)":
        classifier_df = train_deficit_classifier(X_trainval, y_trainval_class)
        optimal_th_c2 = find_optimal_threshold(y_trainval_class, classifier_df.predict_proba(X_trainval)[:, 1])
        test_prob_c2 = classifier_df.predict_proba(X_test)[:, 1]
        class2_metrics = evaluate_classification(y_test_class, test_prob_c2, threshold=optimal_th_c2)

        calibrated_model = train_calibrated_ensemble(X_trainval, y_trainval_class)
        optimal_th_c3 = find_optimal_threshold(y_trainval_class, calibrated_model.predict_proba(X_trainval)[:, 1])
        test_prob_c3 = calibrated_model.predict_proba(X_test)[:, 1]
        class3_metrics = evaluate_classification(y_test_class, test_prob_c3, threshold=optimal_th_c3)
    else:
        calibrated_model = train_calibrated_ensemble(X_trainval, y_trainval_class)
        optimal_th_c3 = find_optimal_threshold(y_trainval_class, calibrated_model.predict_proba(X_trainval)[:, 1])
        test_prob_c3 = calibrated_model.predict_proba(X_test)[:, 1]
        class3_metrics = evaluate_classification(y_test_class, test_prob_c3, threshold=optimal_th_c3)

        classifier_df = train_deficit_classifier(X_trainval, y_trainval_class)
        optimal_th_c2 = find_optimal_threshold(y_trainval_class, classifier_df.predict_proba(X_trainval)[:, 1])
        test_prob_c2 = classifier_df.predict_proba(X_test)[:, 1]
        class2_metrics = evaluate_classification(y_test_class, test_prob_c2, threshold=optimal_th_c2)

    test_pred_c2 = (test_prob_c2 >= optimal_th_c2).astype(int)
    test_pred_c3 = (test_prob_c3 >= optimal_th_c3).astype(int)

    logger.info(
        "  Modelo 2 [TEST] - AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Brier: %.4f",
        class2_metrics["AUC-ROC"], class2_metrics["F1-Score"],
        class2_metrics["Precision"], class2_metrics["Recall"],
        class2_metrics["Brier Score"],
    )
    logger.info(
        "  Modelo 3 [TEST] - AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Brier: %.4f",
        class3_metrics["AUC-ROC"], class3_metrics["F1-Score"],
        class3_metrics["Precision"], class3_metrics["Recall"],
        class3_metrics["Brier Score"],
    )
    plot_confusion_matrix(y_test_class, test_pred_c2, "Modelo 2 (XGBoost)")
    plot_confusion_matrix(y_test_class, test_pred_c3, "Modelo 3 (Calibrado)")

    # ==========================================
    # Graficos y reporte final
    # ==========================================
    logger.info("[6/6] Generando graficos y reporte comparativo...")

    prob_dict = {
        "Modelo 2 (XGBoost)": test_prob_c2,
        "Modelo 3 (Calibrado)": test_prob_c3,
    }
    plot_roc_curve(y_test_class, prob_dict, save_path=PROCESSED_DIR / "roc_curve_staffing.png")
    plot_precision_recall_curve(y_test_class, prob_dict, save_path=PROCESSED_DIR / "pr_curve_staffing.png")
    plot_calibration_curve_custom(y_test_class, prob_dict, save_path=PROCESSED_DIR / "calibration_curve_staffing.png")

    for name, probs in prob_dict.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_threshold_analysis(
            y_test_class, probs, name,
            save_path=PROCESSED_DIR / f"threshold_analysis_{slug}.png",
        )

    if hasattr(classifier_df, "feature_importances_"):
        plot_feature_importance(
            classifier_df, feature_names, top_n=20,
            save_path=PROCESSED_DIR / "feature_importance_staffing.png",
        )

    comp_df = pd.DataFrame({
        "Modelo 2 (XGBoost)": class2_metrics,
        "Modelo 3 (Calibrado)": class3_metrics,
    }).T
    comp_df.to_csv(PROCESSED_DIR / "model_comparison_metrics.csv")
    _save_model_artifacts(
        feature_names,
        regressor,
        classifier_df,
        calibrated_model,
        optimal_th_c2,
        optimal_th_c3,
        reg_metrics,
        class2_metrics,
        class3_metrics,
        split_date,
        best_name,
    )

    logger.info("=" * 60)
    logger.info("TABLA COMPARATIVA DE MODELOS DE DEFICIT")
    logger.info("=" * 60)
    for idx, row in comp_df.round(4).iterrows():
        logger.info(
            "  %-25s | AUC: %.4f | F1: %.4f | Brier: %.4f",
            idx, row["AUC-ROC"], row["F1-Score"], row["Brier Score"],
        )
    logger.info("=" * 60)
    logger.info(
        "Modelos 1, 2 y 3 completados. Resultados guardados en: %s", PROCESSED_DIR,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
