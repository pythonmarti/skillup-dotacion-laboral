"""Script de entrenamiento y evaluación de modelos predictivos de dotación laboral."""

import sys
import warnings
from pathlib import Path

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
    evaluate_classification
)
from src.models.evaluate import (
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
)


def plot_regression_actual_vs_predicted(y_true, y_pred, save_path):
    """Grafica dotación real vs predicha."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="teal", edgecolors="black", linewidths=0.5)
    
    # Línea de identidad
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfecto")
    
    ax.set_xlabel("Dotación Disponible Real")
    ax.set_ylabel("Dotación Disponible Predicha")
    ax.set_title("Modelo 1: Regresión de Dotación Real vs Predicha")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de regresión guardado en {save_path}")


def plot_calibration_curve_custom(y_true, prob_dict, save_path):
    """Grafica la curva de calibración de probabilidades."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectamente calibrado")
    
    for name, probs in prob_dict.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=8)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)
        
    ax.set_ylabel("Fracción de positivos reales (Frecuencia)")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_title("Curvas de Calibración de Riesgo de Déficit")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de calibración guardado en {save_path}")


def main():
    print("=" * 60)
    print("  PASO 3: Entrenamiento de Modelos de Dotación Laboral")
    print("=" * 60)

    # Asegurar que el directorio de salida existe
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar features con fechas
    print("\nCargando features desde SQLite...")
    data = prepare_feature_matrix()
    X_full = data["X"]
    y_full_class = data["y_class"]
    y_full_reg = data["y_reg"]
    feature_names = data["feature_names"]
    dates = data.get("date")

    print(f"  Samples totales: {X_full.shape[0]}")
    print(f"  Features: {X_full.shape[1]}")
    print(f"  Casos de Déficit en total: {y_full_class.sum()} ({y_full_class.mean() * 100:.1f}%)")

    # 2. Split temporal
    print("\nAplicando split temporal...")
    if dates is not None:
        temp_df = X_full.copy()
        temp_df["date"] = dates
        temp_df["_target_class"] = y_full_class.values
        temp_df["_target_reg"] = y_full_reg.values

        train_df, test_df = temporal_train_test_split(
            temp_df, date_col="date", test_ratio=MODEL_CONFIG["test_ratio"]
        )

        X_train = train_df.drop(columns=["date", "_target_class", "_target_reg"])
        y_train_class = train_df["_target_class"].astype(int)
        y_train_reg = train_df["_target_reg"]
        
        X_test = test_df.drop(columns=["date", "_target_class", "_target_reg"])
        y_test_class = test_df["_target_class"].astype(int)
        y_test_reg = test_df["_target_reg"]
    else:
        raise ValueError("Se requieren fechas para realizar la validación temporal.")

    print(f"  Train: {X_train.shape[0]} registros, Deficit: {y_train_class.sum()} ({y_train_class.mean()*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} registros, Deficit: {y_test_class.sum()} ({y_test_class.mean()*100:.1f}%)")

    # ==========================================
    # Modelo 1: Regresor de Headcount
    # ==========================================
    print("\n--- MODELO 1: Regresor de Headcount ---")
    regressor = train_headcount_regressor(X_train, y_train_reg)
    test_pred_reg = regressor.predict(X_test)
    reg_metrics = evaluate_regression(y_test_reg, test_pred_reg)
    
    print(f"  MAE (Error absoluto medio): {reg_metrics['MAE']:.2f} operadores")
    print(f"  RMSE (Error cuadrático medio): {reg_metrics['RMSE']:.2f} operadores")
    print(f"  R2 Score: {reg_metrics['R2']:.4f}")
    
    plot_regression_actual_vs_predicted(
        y_test_reg, 
        test_pred_reg, 
        PROCESSED_DIR / "headcount_actual_vs_predicted.png"
    )

    # ==========================================
    # Modelo 2: Clasificador de Riesgo de Déficit
    # ==========================================
    print("\n--- MODELO 2: Clasificador de Déficit (XGBoost) ---")
    classifier_df = train_deficit_classifier(X_train, y_train_class)
    
    # Probabilidades en entrenamiento para optimizar threshold
    train_prob_class = classifier_df.predict_proba(X_train)[:, 1]
    optimal_th_c2 = find_optimal_threshold(y_train_class, train_prob_class)
    
    # Evaluación en test
    test_prob_c2 = classifier_df.predict_proba(X_test)[:, 1]
    class2_metrics = evaluate_classification(y_test_class, test_prob_c2, threshold=optimal_th_c2)
    
    print("\nMétricas Modelo 2 (XGBoost) en Test:")
    for k, v in class2_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    test_pred_c2 = (test_prob_c2 >= optimal_th_c2).astype(int)
    plot_confusion_matrix(y_test_class, test_pred_c2, "Modelo 2 (XGBoost)")

    # ==========================================
    # Modelo 3: Ensamble Calibrado
    # ==========================================
    print("\n--- MODELO 3: Ensamble Calibrado (XGBoost Calibrado) ---")
    calibrated_model = train_calibrated_ensemble(X_train, y_train_class)
    
    # Probabilidades en entrenamiento para optimizar threshold
    train_prob_c3 = calibrated_model.predict_proba(X_train)[:, 1]
    optimal_th_c3 = find_optimal_threshold(y_train_class, train_prob_c3)
    
    # Evaluación en test
    test_prob_c3 = calibrated_model.predict_proba(X_test)[:, 1]
    class3_metrics = evaluate_classification(y_test_class, test_prob_c3, threshold=optimal_th_c3)
    
    print("\nMétricas Modelo 3 (Calibrado) en Test:")
    for k, v in class3_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    test_pred_c3 = (test_prob_c3 >= optimal_th_c3).astype(int)
    plot_confusion_matrix(y_test_class, test_pred_c3, "Modelo 3 (Calibrado)")

    # ==========================================
    # Comparaciones y Gráficos Finales
    # ==========================================
    print("\nGenerando gráficos comparativos de clasificación...")
    prob_dict = {
        "Modelo 2 (XGBoost)": test_prob_c2,
        "Modelo 3 (Calibrado)": test_prob_c3
    }
    plot_roc_curve(y_test_class, prob_dict, save_path=PROCESSED_DIR / "roc_curve_staffing.png")
    plot_precision_recall_curve(y_test_class, prob_dict, save_path=PROCESSED_DIR / "pr_curve_staffing.png")
    
    # Curva de Calibración
    plot_calibration_curve_custom(y_test_class, prob_dict, save_path=PROCESSED_DIR / "calibration_curve_staffing.png")
    
    for name, probs in prob_dict.items():
        plot_threshold_analysis(y_test_class, probs, name, save_path=PROCESSED_DIR / f"threshold_analysis_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")

    # Importancia de variables del clasificador base
    if hasattr(classifier_df, "feature_importances_"):
        plot_feature_importance(
            classifier_df, 
            feature_names, 
            top_n=20, 
            save_path=PROCESSED_DIR / "feature_importance_staffing.png"
        )

    # Imprimir resumen de comparación de clasificación
    print("\n" + "=" * 60)
    print("  Tabla Comparativa de Modelos de Déficit")
    print("=" * 60)
    comp_df = pd.DataFrame({
        "Modelo 2 (XGBoost)": class2_metrics,
        "Modelo 3 (Calibrado)": class3_metrics
    }).T
    print(comp_df.round(4).to_string())
    print("=" * 60 + "\n")
    
    # Guardar métricas en CSV para referencia
    comp_df.to_csv(PROCESSED_DIR / "model_comparison_metrics.csv")
    
    print("¡Entrenamiento y evaluación de los 3 modelos completados con éxito!")


if __name__ == "__main__":
    main()
