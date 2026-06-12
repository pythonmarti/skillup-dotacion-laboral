"""Dashboard ejecutivo del dominio clinic ambulatorio."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.clinic_settings import CLINIC_CRITICAL_ROLE_TARGETS, CLINIC_DB_PATH, CLINIC_MODEL_CONFIG, CLINIC_PROCESSED_DIR
from src.clinic.inference import run_clinic_inference
from src.utils.database import query_to_dataframe


def _load_or_build_predictions():
    predictions_path = CLINIC_PROCESSED_DIR / "clinic_staffing_predictions.csv"
    metrics_path = CLINIC_PROCESSED_DIR / "clinic_staffing_metrics.json"
    if predictions_path.exists() and metrics_path.exists():
        predictions = pd.read_csv(predictions_path, parse_dates=["date"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        return predictions, metrics
    predictions, metrics, _ = run_clinic_inference()
    return predictions, metrics


def generate_clinic_dashboard() -> str:
    CLINIC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions, metrics = _load_or_build_predictions()
    patient_flow = query_to_dataframe("SELECT * FROM clinic_patient_flow", db_path=CLINIC_DB_PATH)
    absenteeism = query_to_dataframe("SELECT * FROM clinic_absenteeism", db_path=CLINIC_DB_PATH)
    biometrics = query_to_dataframe("SELECT date, stress_score, fatigue_proxy, cognitive_load_score, reaction_time_ms FROM clinic_biometrics", db_path=CLINIC_DB_PATH)

    patient_flow["date"] = pd.to_datetime(patient_flow["date"])
    absenteeism["date"] = pd.to_datetime(absenteeism["date"])
    biometrics["date"] = pd.to_datetime(biometrics["date"])

    latest_window_start = predictions["date"].max() - pd.Timedelta(days=27)
    recent_predictions = predictions[predictions["date"] >= latest_window_start].copy()
    recent_flow = patient_flow[patient_flow["date"] >= latest_window_start].copy()
    recent_absenteeism = absenteeism[absenteeism["date"] >= latest_window_start].copy()
    recent_biometrics = biometrics[biometrics["date"] >= latest_window_start].copy()

    avg_risk = float(recent_predictions["predicted_deficit_probability"].mean())
    high_risk_count = int((recent_predictions["predicted_deficit_probability"] >= CLINIC_MODEL_CONFIG["high_risk_threshold"]).sum())
    avg_stations = float(recent_flow["active_care_stations"].mean()) if not recent_flow.empty else 0.0
    top_unit = recent_predictions.groupby("clinical_unit")["predicted_deficit_probability"].mean().sort_values(ascending=False).index[0]

    heatmap = recent_predictions.pivot_table(index="clinical_unit", columns="shift", values="predicted_deficit_probability", aggfunc="mean")
    role_risk = pd.DataFrame({
        "role": CLINIC_CRITICAL_ROLE_TARGETS,
        "risk": [recent_predictions.get(f"predicted_role_deficit_prob_{role}", pd.Series([0])).mean() for role in CLINIC_CRITICAL_ROLE_TARGETS],
    }).sort_values("risk", ascending=False)

    headcount_trend = recent_predictions.groupby("date").agg(
        required=("required_headcount_total", "mean"),
        actual=("actual_headcount_total", "mean"),
        predicted=("predicted_headcount_total", "mean"),
    ).reset_index()
    workload_trend = recent_flow.groupby("date").agg(
        avg_volume=("actual_patient_volume", "mean"),
        avg_stations=("active_care_stations", "mean"),
        avg_wait=("average_wait_minutes", "mean"),
    ).reset_index()
    cognitive_trend = recent_biometrics.groupby("date").agg(
        avg_stress=("stress_score", "mean"),
        avg_fatigue=("fatigue_proxy", "mean"),
        avg_cognitive_load=("cognitive_load_score", "mean"),
    ).reset_index()
    absence_reason_df = recent_absenteeism.groupby("absence_reason").size().reset_index(name="count").sort_values("count", ascending=False).head(5)

    insight_lines = [
        f"Riesgo promedio reciente de deficit: {avg_risk:.1%}.",
        f"Unidad con mayor exposicion: {top_unit}.",
        f"Promedio reciente de boxes activos: {avg_stations:.1f}.",
        f"Se detectaron {high_risk_count} segmentos de alto riesgo en la ventana analizada.",
    ]

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[0.85, 1.6, 1.35])

    ax_title = fig.add_subplot(grid[0, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.95, "Resumen Ejecutivo - Clinica Ambulatoria", fontsize=20, fontweight="bold", va="top")
    ax_title.text(0.0, 0.72, "Gestion de dotacion ambulatoria con agenda, procedimientos, boxes activos y alerta respiratoria", fontsize=11, va="top")
    ax_title.text(0.0, 0.48, f"Ventana analizada: {recent_predictions['date'].min().date()} a {recent_predictions['date'].max().date()}", fontsize=10, va="top")

    kpi_positions = [0.00, 0.26, 0.52, 0.78]
    kpis = [
        ("Riesgo medio", f"{avg_risk:.1%}"),
        ("Segmentos alto riesgo", str(high_risk_count)),
        ("Boxes activos", f"{avg_stations:.1f}"),
        ("Unidad mas expuesta", top_unit),
    ]
    for position, (label, value) in zip(kpi_positions, kpis):
        ax_title.text(position, 0.20, label, fontsize=10, fontweight="bold")
        ax_title.text(position, 0.02, value, fontsize=18, fontweight="bold")

    ax_heatmap = fig.add_subplot(grid[1, 0:2])
    sns.heatmap(heatmap, cmap="YlOrRd", annot=True, fmt=".2f", cbar=False, ax=ax_heatmap)
    ax_heatmap.set_title("Riesgo medio de deficit por unidad y turno", fontsize=11, fontweight="bold")
    ax_heatmap.set_xlabel("Turno")
    ax_heatmap.set_ylabel("Unidad")

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
    ax_headcount.plot(headcount_trend["date"], headcount_trend["required"], label="Requerida", linewidth=2, color="#d95f02")
    ax_headcount.plot(headcount_trend["date"], headcount_trend["actual"], label="Actual", linewidth=2, color="#1b9e77")
    ax_headcount.plot(headcount_trend["date"], headcount_trend["predicted"], label="Predicha", linewidth=2, color="#7570b3")
    ax_headcount.set_title("Dotacion requerida vs actual vs predicha", fontsize=11, fontweight="bold")
    ax_headcount.tick_params(axis="x", rotation=35)
    ax_headcount.legend(frameon=False)
    ax_headcount.grid(alpha=0.2)

    ax_workload = fig.add_subplot(grid[2, 2])
    ax_workload.plot(workload_trend["date"], workload_trend["avg_volume"], label="Pacientes", color="#1d4ed8", linewidth=2)
    ax_workload.plot(workload_trend["date"], workload_trend["avg_wait"], label="Espera", color="#ea580c", linewidth=2)
    ax_workload.plot(cognitive_trend["date"], cognitive_trend["avg_cognitive_load"], label="Carga cognitiva", color="#9333ea", linewidth=2)
    ax_workload.set_title("Carga asistencial reciente", fontsize=11, fontweight="bold")
    ax_workload.tick_params(axis="x", rotation=35)
    ax_workload.legend(frameon=False)
    ax_workload.grid(alpha=0.2)

    ax_text = fig.add_subplot(grid[2, 3])
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, "Alertas y accion sugerida", fontsize=12, fontweight="bold", va="top")
    for index, line in enumerate(insight_lines, start=1):
        ax_text.text(0.0, 1.0 - (index * 0.16), f"{index}. {line}", fontsize=10, va="top")
    ax_text.text(
        0.0,
        0.16,
        f"F1 mejor clasificador: {metrics['classification_best']['F1-Score']:.3f}\n"
        f"AUC mejor clasificador: {metrics['classification_best']['AUC-ROC']:.3f}\n"
        f"MAE headcount: {metrics['regression']['MAE']:.2f}",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )

    output_path = CLINIC_PROCESSED_DIR / "clinic_executive_dashboard.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
