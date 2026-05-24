"""Funciones de evaluacion de modelos predictivos."""

import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true, y_prob):
    """Encuentra el threshold que maximiza F1-Score usando la curva precision-recall.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_prob : array-like
        Probabilidades predichas para la clase positiva.

    Returns
    -------
    float
        Threshold optimo.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision y recall tienen len(thresholds)+1 elementos
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    logger.info("  Threshold optimo: %.4f (F1=%.4f)", best_threshold, best_f1)
    return best_threshold


def plot_threshold_analysis(y_true, y_prob, model_name, save_path=None):
    """Grafica F1, Precision y Recall vs threshold.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    model_name : str
    save_path : str or Path, optional
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision[:-1], label="Precision", linewidth=2)
    ax.plot(thresholds, recall[:-1], label="Recall", linewidth=2)
    ax.plot(thresholds, f1_scores, label="F1-Score", linewidth=2, linestyle="--")

    best_idx = np.argmax(f1_scores)
    ax.axvline(thresholds[best_idx], color="red", linestyle=":", alpha=0.7,
               label=f"Optimo ({thresholds[best_idx]:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = save_path or PROCESSED_DIR / f"threshold_analysis_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Threshold analysis guardado en %s", save_path)


def plot_precision_recall_curve(y_true, y_prob_dict, save_path=None):
    """Curva Precision-Recall comparativa para multiples modelos.

    Parameters
    ----------
    y_true : array-like
    y_prob_dict : dict
        {"Model Name": probabilities, ...}
    save_path : str or Path, optional
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_prob in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", linewidth=2)

    # Baseline: proporcion de positivos
    baseline = y_true.mean() if hasattr(y_true, 'mean') else np.mean(y_true)
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Prediccion de Ausentismo")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_path = save_path or PROCESSED_DIR / "pr_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve guardada en %s", save_path)


def print_classification_report(y_true, y_pred, y_prob, model_name):
    """Imprime reporte de clasificacion y metricas clave."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Classification Report")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred))

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(
        "  %s | AUC: %.4f | Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f",
        model_name, auc, acc, prec, rec, f1,
    )

    return {"auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_roc_curve(y_true, y_prob_dict, save_path=None):
    """Grafica curva ROC para multiples modelos.

    Parameters
    ----------
    y_true : array-like
    y_prob_dict : dict
        {"Model Name": probabilities, ...}
    save_path : str or Path, optional
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Prediccion de Ausentismo")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    save_path = save_path or PROCESSED_DIR / "roc_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve guardada en %s", save_path)


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Bar chart horizontal de las top_n features mas importantes."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="steelblue",
        edgecolor="navy",
        alpha=0.8,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {top_n} Features Mas Importantes")
    ax.grid(True, alpha=0.3, axis="x")

    save_path = save_path or PROCESSED_DIR / "feature_importance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance guardada en %s", save_path)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Heatmap de confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Ausente", "Ausente"],
        yticklabels=["No Ausente", "Ausente"],
        ax=ax,
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Confusion Matrix - {model_name}")

    save_path = save_path or PROCESSED_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix guardada en %s", save_path)


def compare_models(results_dict):
    """Tabla comparativa de metricas entre modelos.

    Parameters
    ----------
    results_dict : dict
        {"Model Name": {"auc": ..., "accuracy": ..., ...}, ...}

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(results_dict).T
    df.index.name = "Modelo"
    df = df.round(4)
    logger.info("Comparacion de modelos:\n%s", df.to_string())
    return df
