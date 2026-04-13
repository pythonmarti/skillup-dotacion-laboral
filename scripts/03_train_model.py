"""Script de entrenamiento y evaluacion de modelos predictivos."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from config.settings import MODEL_CONFIG
from src.models.features import prepare_feature_matrix, temporal_train_test_split
from src.models.train import train_and_compare
from src.models.evaluate import (
    compare_models,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
    print_classification_report,
)


def main():
    print("=" * 60)
    print("  PASO 3: Entrenamiento de Modelos Predictivos (v2)")
    print("=" * 60)

    # 1. Cargar features con fechas
    print("\nCargando features desde SQLite...")
    data = prepare_feature_matrix()
    X_full = data["X"]
    y_full = data["y_class"]
    feature_names = data["feature_names"]
    dates = data.get("date")

    print(f"  Samples totales: {X_full.shape[0]}")
    print(f"  Features: {X_full.shape[1]}")
    print(f"  Clase positiva: {y_full.sum()} ({y_full.mean() * 100:.1f}%)")

    # 2. Split temporal
    print("\nAplicando split temporal...")
    if dates is not None:
        temp_df = X_full.copy()
        temp_df["date"] = dates
        temp_df["_target"] = y_full.values

        train_df, test_df = temporal_train_test_split(
            temp_df, date_col="date", test_ratio=MODEL_CONFIG["test_ratio"]
        )

        X_train = train_df.drop(columns=["date", "_target"])
        y_train = train_df["_target"].astype(int)
        X_test = test_df.drop(columns=["date", "_target"])
        y_test = test_df["_target"].astype(int)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=MODEL_CONFIG["test_ratio"],
            stratify=y_full, random_state=42,
        )

    print(f"  Train: {X_train.shape[0]} samples, positivos: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples, positivos: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # 3. Entrenar todos los modelos en train set
    results = train_and_compare(X_train, y_train, temporal=True)
    models = results["models"]
    train_probs = results["probs"]

    # 4. Threshold optimization en train probs
    print("\n" + "=" * 60)
    print("  Optimizacion de Threshold")
    print("=" * 60)

    thresholds = {}
    for name, probs in train_probs.items():
        print(f"\n{name} (train set):")
        th_train = find_optimal_threshold(y_train, probs)
        thresholds[name] = th_train

    # 5. Evaluar en test set
    print("\n" + "=" * 60)
    print("  Evaluacion en Test Set (split temporal)")
    print("=" * 60)

    all_metrics = {}
    test_probs = {}

    for name, model in models.items():
        test_prob = model.predict_proba(X_test)[:, 1]
        test_probs[name] = test_prob

        # Usar threshold de train
        optimal_th = thresholds[name]
        test_pred = (test_prob >= optimal_th).astype(int)

        # Oracle threshold en test (referencia)
        oracle_th = find_optimal_threshold(y_test, test_prob)
        oracle_pred = (test_prob >= oracle_th).astype(int)
        from sklearn.metrics import f1_score
        oracle_f1 = f1_score(y_test, oracle_pred, zero_division=0)

        print(f"\n{name} (threshold={optimal_th:.4f}, oracle_th={oracle_th:.4f}, oracle_F1={oracle_f1:.4f}):")
        metrics = print_classification_report(y_test, test_pred, test_prob, name)
        metrics["threshold"] = optimal_th
        metrics["oracle_f1"] = oracle_f1
        all_metrics[name] = metrics

        plot_confusion_matrix(y_test, test_pred, name)

    # 6. Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_roc_curve(y_test, test_probs)
    plot_precision_recall_curve(y_test, test_probs)

    for name, probs in test_probs.items():
        plot_threshold_analysis(y_test, probs, name)

    # Feature importance del mejor modelo
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_model = models[best_model_name]
    model_for_importance = best_model
    if hasattr(best_model, "calibrated_classifiers_"):
        model_for_importance = best_model.calibrated_classifiers_[0].estimator
    if hasattr(model_for_importance, "feature_importances_"):
        plot_feature_importance(model_for_importance, feature_names, top_n=20)

    # 7. Tabla comparativa final
    comparison = compare_models(all_metrics)

    best_f1 = all_metrics[best_model_name]["f1"]
    print(f"\nMejor modelo: {best_model_name} (F1={best_f1:.4f})")
    print("\nNota: Los datos sinteticos tienen AUC ~0.53 en test temporal,")
    print("lo que limita el F1 maximo alcanzable. Con datos reales de ENAP")
    print("se esperan mejoras significativas dado que las relaciones causales")
    print("seran mas fuertes y consistentes.")
    print("\nEntrenamiento completado exitosamente.")


if __name__ == "__main__":
    main()
