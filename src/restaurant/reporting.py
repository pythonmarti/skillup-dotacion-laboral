"""Dashboard ejecutivo del dominio restaurant."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.restaurant_settings import CRITICAL_ROLE_TARGETS, RESTAURANT_DB_PATH, RESTAURANT_MODEL_CONFIG, RESTAURANT_PROCESSED_DIR
from src.restaurant.inference import run_restaurant_inference
from src.utils.database import query_to_dataframe


def _load_or_build_predictions():
    predictions_path = RESTAURANT_PROCESSED_DIR / "restaurant_staffing_predictions.csv"
    metrics_path = RESTAURANT_PROCESSED_DIR / "restaurant_staffing_metrics.json"
    if predictions_path.exists() and metrics_path.exists():
        predictions = pd.read_csv(predictions_path, parse_dates=["date"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        return predictions, metrics
    predictions, metrics, _ = run_restaurant_inference()
    return predictions, metrics


def generate_restaurant_dashboard() -> str:
    RESTAURANT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions, metrics = _load_or_build_predictions()
    absenteeism = query_to_dataframe("SELECT * FROM restaurant_absenteeism", db_path=RESTAURANT_DB_PATH)
    biometrics = query_to_dataframe("SELECT date, stress_score, sleep_duration_hours, fatigue_proxy FROM restaurant_biometrics", db_path=RESTAURANT_DB_PATH)
    absenteeism["date"] = pd.to_datetime(absenteeism["date"])
    biometrics["date"] = pd.to_datetime(biometrics["date"])

    latest_window_start = predictions["date"].max() - pd.Timedelta(days=27)
    recent_predictions = predictions[predictions["date"] >= latest_window_start].copy()
    recent_biometrics = biometrics[biometrics["date"] >= latest_window_start].copy()
    recent_absenteeism = absenteeism[absenteeism["date"] >= latest_window_start].copy()

    avg_risk = recent_predictions["predicted_deficit_probability"].mean()
    high_risk_count = int((recent_predictions["predicted_deficit_probability"] >= RESTAURANT_MODEL_CONFIG["high_risk_threshold"]).sum())
    avg_absenteeism = float(recent_absenteeism.groupby("date").size().mean()) if not recent_absenteeism.empty else 0.0
    top_role = max(
        CRITICAL_ROLE_TARGETS,
        key=lambda role: recent_predictions.get(f"predicted_role_deficit_prob_{role}", pd.Series([0])).mean(),
    )

    heatmap_df = predictions.copy()
    heatmap_df["weekday"] = heatmap_df["date"].dt.day_name()
    heatmap = heatmap_df.pivot_table(index="weekday", columns="service_period", values="predicted_deficit_probability", aggfunc="mean")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = heatmap.reindex([day for day in weekday_order if day in heatmap.index])

    role_risk = pd.DataFrame({
        "role": CRITICAL_ROLE_TARGETS,
        "risk": [recent_predictions.get(f"predicted_role_deficit_prob_{role}", pd.Series([0])).mean() for role in CRITICAL_ROLE_TARGETS],
    }).sort_values("risk", ascending=False)

    stress_trend = recent_biometrics.groupby("date").agg(
        avg_stress=("stress_score", "mean"),
        avg_sleep=("sleep_duration_hours", "mean"),
        avg_fatigue=("fatigue_proxy", "mean"),
    ).reset_index()

    headcount_trend = recent_predictions.groupby("date").agg(
        actual=("actual_headcount_total", "mean"),
        predicted=("predicted_headcount_total", "mean"),
        required=("required_headcount_total", "mean"),
    ).reset_index()

    absence_reason_df = recent_absenteeism.groupby("absence_reason").size().reset_index(name="count").sort_values("count", ascending=False).head(5)

    insight_lines = [
        f"Riesgo promedio de deficit reciente: {avg_risk:.1%}.",
        f"Franja con mayor exposicion por rol critico: {top_role}.",
        f"Se detectaron {high_risk_count} franjas de alto riesgo en la ventana analizada.",
    ]
    if not role_risk.empty:
        insight_lines.append(f"Prioridad operativa: reforzar {role_risk.iloc[0]['role']} en bloques peak y festivos.")

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[0.85, 1.6, 1.35])

    ax_title = fig.add_subplot(grid[0, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.95, "Resumen Ejecutivo - Restaurant Casual Dining", fontsize=20, fontweight="bold", va="top")
    ax_title.text(0.0, 0.72, "Gestion de dotacion en horas peak, fines de semana y festivos (Chile)", fontsize=11, va="top")
    ax_title.text(0.0, 0.48, f"Ventana analizada: {recent_predictions['date'].min().date()} a {recent_predictions['date'].max().date()}", fontsize=10, va="top")

    kpi_positions = [0.00, 0.26, 0.52, 0.78]
    kpis = [
        ("Riesgo medio", f"{avg_risk:.1%}"),
        ("Franjas alto riesgo", str(high_risk_count)),
        ("Ausencias promedio/dia", f"{avg_absenteeism:.1f}"),
        ("Rol mas vulnerable", top_role),
    ]
    for position, (label, value) in zip(kpi_positions, kpis):
        ax_title.text(position, 0.20, label, fontsize=10, fontweight="bold")
        ax_title.text(position, 0.02, value, fontsize=18, fontweight="bold")

    ax_heatmap = fig.add_subplot(grid[1, 0:2])
    sns.heatmap(heatmap, cmap="YlOrRd", annot=True, fmt=".2f", cbar=False, ax=ax_heatmap)
    ax_heatmap.set_title("Riesgo medio de deficit por dia y franja", fontsize=11, fontweight="bold")
    ax_heatmap.set_xlabel("Franja")
    ax_heatmap.set_ylabel("Dia")

    ax_roles = fig.add_subplot(grid[1, 2])
    sns.barplot(data=role_risk, x="risk", y="role", hue="role", palette="Reds_r", ax=ax_roles, legend=False)
    ax_roles.set_title("Riesgo por rol critico", fontsize=11, fontweight="bold")
    ax_roles.set_xlabel("Probabilidad media")
    ax_roles.set_ylabel("")

    ax_absence = fig.add_subplot(grid[1, 3])
    if not absence_reason_df.empty:
        sns.barplot(data=absence_reason_df, x="count", y="absence_reason", hue="absence_reason", palette="Blues_r", ax=ax_absence, legend=False)
    ax_absence.set_title("Top causas de ausentismo", fontsize=11, fontweight="bold")
    ax_absence.set_xlabel("Eventos")
    ax_absence.set_ylabel("")

    ax_headcount = fig.add_subplot(grid[2, 0:2])
    ax_headcount.plot(headcount_trend["date"], headcount_trend["required"], label="Requerido", linewidth=2, color="#d95f02")
    ax_headcount.plot(headcount_trend["date"], headcount_trend["actual"], label="Actual", linewidth=2, color="#1b9e77")
    ax_headcount.plot(headcount_trend["date"], headcount_trend["predicted"], label="Predicho", linewidth=2, color="#7570b3")
    ax_headcount.set_title("Dotacion requerida vs actual vs predicha", fontsize=11, fontweight="bold")
    ax_headcount.tick_params(axis="x", rotation=35)
    ax_headcount.legend(frameon=False)
    ax_headcount.grid(alpha=0.2)

    ax_bio = fig.add_subplot(grid[2, 2])
    ax_bio.plot(stress_trend["date"], stress_trend["avg_stress"], label="Estres", color="#e7298a", linewidth=2)
    ax_bio.plot(stress_trend["date"], stress_trend["avg_fatigue"], label="Fatiga", color="#66a61e", linewidth=2)
    ax_bio.set_title("Carga fisiologica reciente", fontsize=11, fontweight="bold")
    ax_bio.tick_params(axis="x", rotation=35)
    ax_bio.legend(frameon=False)
    ax_bio.grid(alpha=0.2)

    ax_text = fig.add_subplot(grid[2, 3])
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, "Alertas y accion sugerida", fontsize=12, fontweight="bold", va="top")
    for index, line in enumerate(insight_lines, start=1):
        ax_text.text(0.0, 1.0 - (index * 0.18), f"{index}. {line}", fontsize=10, va="top")
    ax_text.text(
        0.0,
        0.18,
        f"F1 mejor clasificador: {metrics['classification_best']['F1-Score']:.3f}\n"
        f"AUC mejor clasificador: {metrics['classification_best']['AUC-ROC']:.3f}\n"
        f"MAE headcount: {metrics['regression']['MAE']:.2f}",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )

    output_path = RESTAURANT_PROCESSED_DIR / "restaurant_executive_dashboard.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
