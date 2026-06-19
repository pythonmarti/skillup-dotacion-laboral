"""Dashboard interactivo con Streamlit para operar dominios de SkillUp."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config.clinic_settings import CLINIC_CRITICAL_ROLE_TARGETS, CLINIC_MODEL_CONFIG, CLINIC_PROCESSED_DIR, CLINIC_RAW_DIR
from config.restaurant_settings import CRITICAL_ROLE_TARGETS, RESTAURANT_PROCESSED_DIR, RESTAURANT_RAW_DIR
from config.settings import PROCESSED_DIR, RAW_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER_SCRIPT = PROJECT_ROOT / "scripts" / "run_pipeline.py"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass(frozen=True)
class DomainArtifacts:
    raw_dir: Path
    processed_dir: Path
    metrics_json: Path
    predictions_csv: Path
    risk_shap_global_csv: Path
    headcount_shap_global_csv: Path
    risk_shap_segment_csv: Path
    risk_shap_effects_csv: Path
    risk_shap_explanations_csv: Path
    risk_shap_summary_json: Path
    dashboard_images: list[Path]
    summary_title: str
    summary_subtitle: str
    segment_label: str


def get_domain_artifacts(domain: str) -> DomainArtifacts:
    if domain == "clinic":
        return DomainArtifacts(
            raw_dir=CLINIC_RAW_DIR,
            processed_dir=CLINIC_PROCESSED_DIR,
            metrics_json=CLINIC_PROCESSED_DIR / "clinic_staffing_metrics.json",
            predictions_csv=CLINIC_PROCESSED_DIR / "clinic_staffing_predictions.csv",
            risk_shap_global_csv=CLINIC_PROCESSED_DIR / "clinic_risk_shap_global.csv",
            headcount_shap_global_csv=CLINIC_PROCESSED_DIR / "clinic_headcount_shap_global.csv",
            risk_shap_segment_csv=CLINIC_PROCESSED_DIR / "clinic_risk_shap_segment.csv",
            risk_shap_effects_csv=CLINIC_PROCESSED_DIR / "clinic_risk_shap_effects.csv",
            risk_shap_explanations_csv=CLINIC_PROCESSED_DIR / "clinic_risk_shap_explanations.csv",
            risk_shap_summary_json=CLINIC_PROCESSED_DIR / "clinic_risk_shap_summary.json",
            dashboard_images=[CLINIC_PROCESSED_DIR / "clinic_executive_dashboard.png"],
            summary_title="Clinica Ambulatoria",
            summary_subtitle="Riesgo de deficit por unidad ambulatoria y turno con foco en agenda, procedimientos, boxes activos y roles criticos.",
            segment_label="Unidad / turno",
        )
    if domain == "restaurant":
        return DomainArtifacts(
            raw_dir=RESTAURANT_RAW_DIR,
            processed_dir=RESTAURANT_PROCESSED_DIR,
            metrics_json=RESTAURANT_PROCESSED_DIR / "restaurant_staffing_metrics.json",
            predictions_csv=RESTAURANT_PROCESSED_DIR / "restaurant_staffing_predictions.csv",
            risk_shap_global_csv=RESTAURANT_PROCESSED_DIR / "restaurant_risk_shap_global.csv",
            headcount_shap_global_csv=RESTAURANT_PROCESSED_DIR / "restaurant_headcount_shap_global.csv",
            risk_shap_segment_csv=RESTAURANT_PROCESSED_DIR / "restaurant_risk_shap_segment.csv",
            risk_shap_effects_csv=RESTAURANT_PROCESSED_DIR / "restaurant_risk_shap_effects.csv",
            risk_shap_explanations_csv=RESTAURANT_PROCESSED_DIR / "restaurant_risk_shap_explanations.csv",
            risk_shap_summary_json=RESTAURANT_PROCESSED_DIR / "restaurant_risk_shap_summary.json",
            dashboard_images=[RESTAURANT_PROCESSED_DIR / "restaurant_executive_dashboard.png"],
            summary_title="Restaurant Casual Dining",
            summary_subtitle="Riesgo de déficit por franja, roles críticos y presión operativa en calendario Chile.",
            segment_label="Franja",
        )
    return DomainArtifacts(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        metrics_json=PROCESSED_DIR / "staffing_inference_metrics.json",
        predictions_csv=PROCESSED_DIR / "staffing_inference_predictions.csv",
        risk_shap_global_csv=PROCESSED_DIR / "industrial_risk_shap_global.csv",
        headcount_shap_global_csv=PROCESSED_DIR / "industrial_headcount_shap_global.csv",
        risk_shap_segment_csv=PROCESSED_DIR / "industrial_risk_shap_segment.csv",
        risk_shap_effects_csv=PROCESSED_DIR / "industrial_risk_shap_effects.csv",
        risk_shap_explanations_csv=PROCESSED_DIR / "industrial_risk_shap_explanations.csv",
        risk_shap_summary_json=PROCESSED_DIR / "industrial_risk_shap_summary.json",
        dashboard_images=[
            PROCESSED_DIR / "headcount_actual_vs_predicted.png",
            PROCESSED_DIR / "roc_curve_staffing.png",
            PROCESSED_DIR / "pr_curve_staffing.png",
            PROCESSED_DIR / "calibration_curve_staffing.png",
            PROCESSED_DIR / "feature_importance_staffing.png",
            PROCESSED_DIR / "confusion_matrix_modelo_2_(xgboost).png",
            PROCESSED_DIR / "confusion_matrix_modelo_3_(calibrado).png",
        ],
        summary_title="Industrial Staffing",
        summary_subtitle="Cobertura esperada por área y turno con foco en déficit operacional y disponibilidad real.",
        segment_label="Area / turno",
    )


def _critical_roles(domain: str) -> list[str]:
    if domain == "restaurant":
        return CRITICAL_ROLE_TARGETS
    if domain == "clinic":
        return CLINIC_CRITICAL_ROLE_TARGETS
    return []


def _uses_total_headcount(domain: str) -> bool:
    return domain in {"restaurant", "clinic"}


def _required_headcount_col(domain: str) -> str:
    return "required_headcount_total" if _uses_total_headcount(domain) else "required_headcount"


def _actual_headcount_col(domain: str) -> str:
    return "actual_headcount_total" if _uses_total_headcount(domain) else "actual_headcount"


def _predicted_headcount_col(domain: str) -> str:
    return "predicted_headcount_total" if _uses_total_headcount(domain) else "predicted_headcount"


def _observed_deficit_col(domain: str) -> str:
    return "has_deficit_total" if _uses_total_headcount(domain) else "has_deficit"


def load_metrics_payload(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def list_artifacts(folder: Path, label: str) -> list[dict[str, str]]:
    if not folder.exists():
        return []
    records = []
    for path in sorted(folder.glob("*")):
        size = path.stat().st_size if path.is_file() else 0
        records.append({
            "type": label,
            "file": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": f"{size:,}" if size else "-",
        })
    return records


def run_pipeline_command(domain: str, stage: str, employees: int, days: int, seed: int) -> tuple[int, str]:
    command = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--domain",
        domain,
        "--stage",
        stage,
        "--employees",
        str(employees),
        "--days",
        str(days),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout
    if result.stderr:
        combined = f"{combined}\n{result.stderr}" if combined else result.stderr
    return result.returncode, combined


def _inject_branding() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f8fafc;
        }
        .hero-card {
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e5e7eb;
            color: #0f172a;
            margin-bottom: 1rem;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        }
        .kpi-card {
            background: white;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .kpi-label {
            color: #64748b;
            font-size: 0.86rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            color: #0f172a;
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .section-title {
            color: #0f172a;
            font-weight: 800;
            font-size: 1.15rem;
            margin-top: 0.4rem;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    if not predictions_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(predictions_csv)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_optional_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_optional_json(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def build_filters(domain: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtros analíticos")
        if "date" in filtered.columns:
            min_date = filtered["date"].min().date()
            max_date = filtered["date"].max().date()
            start_date, end_date = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            filtered = filtered[(filtered["date"].dt.date >= start_date) & (filtered["date"].dt.date <= end_date)]

        if domain == "restaurant":
            if "service_period" in filtered.columns:
                options = sorted(filtered["service_period"].dropna().unique().tolist())
                selected = st.multiselect("Franjas", options, default=options)
                filtered = filtered[filtered["service_period"].isin(selected)]
        elif domain == "clinic":
            if "clinical_unit" in filtered.columns:
                options = sorted(filtered["clinical_unit"].dropna().unique().tolist())
                selected = st.multiselect("Unidades clínicas", options, default=options)
                filtered = filtered[filtered["clinical_unit"].isin(selected)]
            if "shift" in filtered.columns:
                options = sorted(filtered["shift"].dropna().unique().tolist())
                selected = st.multiselect("Turnos", options, default=options)
                filtered = filtered[filtered["shift"].isin(selected)]
        else:
            if "plant_area" in filtered.columns:
                options = sorted(filtered["plant_area"].dropna().unique().tolist())
                selected = st.multiselect("Áreas", options, default=options)
                filtered = filtered[filtered["plant_area"].isin(selected)]
            if "shift" in filtered.columns:
                options = sorted(filtered["shift"].dropna().unique().tolist())
                selected = st.multiselect("Turnos", options, default=options)
                filtered = filtered[filtered["shift"].isin(selected)]

    return filtered


def render_hero(artifacts: DomainArtifacts, domain: str, df: pd.DataFrame) -> None:
    total_rows = len(df)
    date_range = "Sin datos"
    if not df.empty and "date" in df.columns:
        date_range = f"{df['date'].min().date()} a {df['date'].max().date()}"
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; color:#64748b;">SkillUp Executive Ops</div>
            <div style="font-size:2rem; font-weight:800; margin-top:0.35rem;">{artifacts.summary_title}</div>
            <div style="font-size:1rem; color:#334155; margin-top:0.35rem; max-width:980px;">{artifacts.summary_subtitle}</div>
            <div style="font-size:0.95rem; color:#64748b; margin-top:0.8rem;">Dominio: <b>{domain}</b> | Ventana analizada: <b>{date_range}</b> | Registros visibles: <b>{total_rows:,}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(domain: str, df: pd.DataFrame, metrics_payload: dict) -> None:
    if df.empty:
        st.warning("No hay datos filtrados para mostrar KPIs.")
        return

    if domain == "restaurant":
        risk = float(df["predicted_deficit_probability"].mean())
        high_risk = int((df["predicted_deficit_probability"] >= 0.75).sum())
        gap = float((df["required_headcount_total"] - df["predicted_headcount_total"]).clip(lower=0).mean())
        top_role = max(
            CRITICAL_ROLE_TARGETS,
            key=lambda role: df.get(f"predicted_role_deficit_prob_{role}", pd.Series([0])).mean(),
        )
        cards = [
            ("Riesgo medio", f"{risk:.1%}"),
            ("Franjas alto riesgo", f"{high_risk}"),
            ("Brecha media esperada", f"{gap:.2f}"),
            ("Rol más expuesto", top_role),
        ]
    elif domain == "clinic":
        risk = float(df["predicted_deficit_probability"].mean())
        high_risk = int((df["predicted_deficit_probability"] >= CLINIC_MODEL_CONFIG["high_risk_threshold"]).sum())
        gap = float((df["required_headcount_total"] - df["predicted_headcount_total"]).clip(lower=0).mean())
        top_unit = df.groupby("clinical_unit")["predicted_deficit_probability"].mean().sort_values(ascending=False).index[0]
        top_role = max(
            CLINIC_CRITICAL_ROLE_TARGETS,
            key=lambda role: df.get(f"predicted_role_deficit_prob_{role}", pd.Series([0])).mean(),
        )
        cards = [
            ("Riesgo medio", f"{risk:.1%}"),
            ("Segmentos alto riesgo", f"{high_risk}"),
            ("Brecha media esperada", f"{gap:.2f}"),
            ("Unidad / rol crítico", f"{top_unit} / {top_role}"),
        ]
    else:
        risk = float(df["predicted_deficit_probability"].mean())
        high_risk = int((df["predicted_deficit_probability"] >= 0.70).sum())
        gap = float((df["required_headcount"] - df["predicted_headcount"]).clip(lower=0).mean())
        top_segment = (
            df.groupby(["plant_area", "shift"])["predicted_deficit_probability"].mean().sort_values(ascending=False).head(1)
        )
        top_label = "N/D"
        if not top_segment.empty:
            top_label = f"{top_segment.index[0][0]} / {top_segment.index[0][1]}"
        cards = [
            ("Riesgo medio", f"{risk:.1%}"),
            ("Franjas alto riesgo", f"{high_risk}"),
            ("Brecha media esperada", f"{gap:.2f}"),
            ("Segmento crítico", top_label),
        ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if metrics_payload.get("best_classifier"):
        st.caption(
            f"Mejor clasificador activo: {metrics_payload['best_classifier'].get('name', 'N/D')} | "
            f"Threshold: {metrics_payload['best_classifier'].get('threshold', 0):.3f}"
        )


def _safe_numeric_frame(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned = cleaned[np.isfinite(cleaned[col])]
    return cleaned


def render_shap_narrative(domain: str, summary_payload: dict) -> bool:
    if not summary_payload:
        return False
    st.markdown('<div class="section-title">Narrativa explicativa del modelo</div>', unsafe_allow_html=True)
    drivers_up = summary_payload.get("drivers_up", [])
    drivers_down = summary_payload.get("drivers_down", [])
    segment_highlight = summary_payload.get("segment_highlight", {})

    lines = []
    if drivers_up:
        top_names = ", ".join(f"`{row['feature_label']}`" for row in drivers_up[:2])
        lines.append(f"Los factores que mas empujan el riesgo son {top_names}.")
    if drivers_down:
        top_names = ", ".join(f"`{row['feature_label']}`" for row in drivers_down[:2])
        lines.append(f"Los factores que mas lo compensan son {top_names}.")
    if segment_highlight:
        lines.append(segment_highlight.get("message", ""))

    if domain == "restaurant":
        lines.append("La lectura recomendada para operaciones es usar estas explicaciones para decidir refuerzos por franja, reservas, delivery y fatiga del equipo.")
    elif domain == "clinic":
        lines.append("La lectura recomendada para operaciones es usar estas explicaciones para decidir refuerzos por agenda, boxes, procedimientos y ausentismo del equipo.")
    else:
        lines.append("La lectura recomendada para operaciones es usar estas explicaciones para anticipar cobertura, fatiga y ausentismo por area y turno.")

    st.info(" ".join(line for line in lines if line))
    return True


def render_shap_global_chart(title: str, df: pd.DataFrame, direction_axis_title: str) -> bool:
    cleaned = _safe_numeric_frame(df, ["mean_abs_shap", "mean_shap"])
    if cleaned.empty:
        return False
    top_df = cleaned.nlargest(10, "mean_abs_shap").copy()
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    chart = (
        alt.Chart(top_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Impacto medio absoluto (SHAP)"),
            y=alt.Y("feature_label:N", sort="-x", title="Factor"),
            color=alt.Color(
                "mean_shap:Q",
                title=direction_axis_title,
                scale=alt.Scale(domainMid=0, scheme="redblue"),
            ),
            tooltip=[
                "feature_label:N",
                alt.Tooltip("mean_abs_shap:Q", format=".4f", title="Impacto medio"),
                alt.Tooltip("mean_shap:Q", format=".4f", title=direction_axis_title),
                "impact_direction:N",
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")
    return True


def render_shap_segment_heatmap(df: pd.DataFrame, segment_label: str) -> bool:
    cleaned = _safe_numeric_frame(df, ["mean_shap", "mean_abs_shap"])
    if cleaned.empty:
        return False
    top_features = (
        cleaned.groupby("feature_label", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .head(8)["feature_label"]
        .tolist()
    )
    heatmap_df = cleaned[cleaned["feature_label"].isin(top_features)].copy()
    st.markdown('<div class="section-title">Drivers del riesgo por segmento</div>', unsafe_allow_html=True)
    chart = (
        alt.Chart(heatmap_df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("segment_label:N", title=segment_label),
            y=alt.Y("feature_label:N", sort=top_features, title="Factor"),
            color=alt.Color("mean_shap:Q", title="Empuje neto del riesgo", scale=alt.Scale(domainMid=0, scheme="redblue")),
            tooltip=[
                "segment_label:N",
                "feature_label:N",
                alt.Tooltip("mean_shap:Q", format=".4f", title="Impacto neto"),
                alt.Tooltip("mean_abs_shap:Q", format=".4f", title="Impacto absoluto"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")
    return True


def render_shap_effect_chart(df: pd.DataFrame, select_key: str, segment_label: str) -> bool:
    cleaned = _safe_numeric_frame(df, ["feature_value", "shap_value"])
    if cleaned.empty:
        return False
    options = cleaned["feature_label"].dropna().unique().tolist()
    if not options:
        return False
    selected_feature = st.selectbox("Factor a explorar", options, key=select_key)
    feature_df = cleaned[cleaned["feature_label"] == selected_feature].copy()
    st.markdown('<div class="section-title">Como cambia el impacto segun el valor del factor</div>', unsafe_allow_html=True)
    chart = (
        alt.Chart(feature_df)
        .mark_circle(size=70, opacity=0.72)
        .encode(
            x=alt.X("feature_value:Q", title=selected_feature),
            y=alt.Y("shap_value:Q", title="Impacto SHAP sobre el riesgo"),
            color=alt.Color("segment_label:N", title=segment_label),
            tooltip=[
                "segment_label:N",
                alt.Tooltip("feature_value:Q", format=".3f", title="Valor"),
                alt.Tooltip("shap_value:Q", format=".4f", title="Impacto"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")
    return True


def render_shap_explanations_table(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    columns = [
        col for col in [
            "date",
            "service_period",
            "clinical_unit",
            "shift",
            "plant_area",
            "predicted_deficit_probability",
            "top_up_driver_1",
            "top_up_driver_2",
            "top_down_driver_1",
            "top_down_driver_2",
        ]
        if col in df.columns
    ]
    preview_df = df[columns].head(20).copy() if columns else df.head(20).copy()
    st.markdown('<div class="section-title">Explicaciones de segmentos mas sensibles</div>', unsafe_allow_html=True)
    st.dataframe(preview_df, width="stretch")
    return True


def render_shap_layout(domain: str, artifacts: DomainArtifacts) -> bool:
    risk_global_df = load_optional_csv(artifacts.risk_shap_global_csv)
    headcount_global_df = load_optional_csv(artifacts.headcount_shap_global_csv)
    risk_segment_df = load_optional_csv(artifacts.risk_shap_segment_csv)
    risk_effects_df = load_optional_csv(artifacts.risk_shap_effects_csv)
    risk_explanations_df = load_optional_csv(artifacts.risk_shap_explanations_csv)

    if risk_global_df.empty and headcount_global_df.empty:
        return False

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        render_shap_global_chart(
            "Factores que mas explican el riesgo de deficit",
            risk_global_df,
            "Empuje neto del riesgo",
        )
        render_shap_global_chart(
            "Factores que mas explican la dotacion disponible",
            headcount_global_df,
            "Empuje neto sobre la dotacion",
        )
    with col_right:
        render_shap_segment_heatmap(risk_segment_df, artifacts.segment_label)
        render_shap_effect_chart(risk_effects_df, f"shap_effect_{domain}", artifacts.segment_label)

    render_shap_explanations_table(risk_explanations_df)
    return True


def _weekday_heatmap_base(df: pd.DataFrame) -> pd.DataFrame:
    heatmap_df = df.copy()
    heatmap_df["weekday"] = heatmap_df["date"].dt.day_name()
    heatmap_df["weekday"] = pd.Categorical(heatmap_df["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    return heatmap_df


def render_risk_heatmap(domain: str, df: pd.DataFrame) -> None:
    if df.empty or "date" not in df.columns:
        return
    heatmap_df = _weekday_heatmap_base(df)
    x_col = "service_period" if domain == "restaurant" else "shift"
    observed_col = _observed_deficit_col(domain)
    use_observed = observed_col in heatmap_df.columns
    value_col = observed_col if use_observed else "predicted_deficit_probability"
    title = "Mapa de déficit observado por día y segmento" if use_observed else "Mapa de riesgo por día y segmento"
    color_title = "Déficit" if use_observed else "Riesgo"

    chart = (
        alt.Chart(heatmap_df)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X(f"{x_col}:N", title="Franja" if domain == "restaurant" else "Turno"),
            y=alt.Y("weekday:N", sort=WEEKDAY_ORDER, title="Día"),
            color=alt.Color(f"mean({value_col}):Q", title=color_title, scale=alt.Scale(scheme="yelloworangered")),
            tooltip=["weekday:N", f"{x_col}:N", alt.Tooltip(f"mean({value_col}):Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.altair_chart(chart, width="stretch")


def render_trend_chart(df: pd.DataFrame, domain: str) -> None:
    if df.empty:
        return
    trend = df.groupby("date", as_index=False).agg(
        predicted_deficit_probability=("predicted_deficit_probability", "mean"),
        predicted_headcount=(_predicted_headcount_col(domain), "mean"),
        actual_headcount=(_actual_headcount_col(domain), "mean"),
        required_headcount=(_required_headcount_col(domain), "mean"),
    )

    line = alt.Chart(trend).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("date:T", title="Fecha"),
        y=alt.Y("predicted_deficit_probability:Q", title="Riesgo medio"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("predicted_deficit_probability:Q", format=".3f")],
    )
    st.markdown('<div class="section-title">Tendencia temporal del riesgo</div>', unsafe_allow_html=True)
    st.altair_chart(line.properties(height=280), width="stretch")


def render_capacity_chart(df: pd.DataFrame, domain: str) -> None:
    if df.empty:
        return
    if domain == "industrial":
        scatter_df = df[["actual_headcount", "predicted_headcount", "predicted_deficit_probability", "plant_area", "shift"]].copy()
        max_value = float(max(scatter_df["actual_headcount"].max(), scatter_df["predicted_headcount"].max()))
        base = alt.Chart(scatter_df)
        points = base.mark_circle(size=70, opacity=0.75).encode(
            x=alt.X("actual_headcount:Q", title="Dotación real"),
            y=alt.Y("predicted_headcount:Q", title="Dotación predicha"),
            color=alt.Color("predicted_deficit_probability:Q", title="Riesgo", scale=alt.Scale(scheme="orangered")),
            tooltip=[
                alt.Tooltip("plant_area:N", title="Área"),
                alt.Tooltip("shift:N", title="Turno"),
                alt.Tooltip("actual_headcount:Q", title="Real"),
                alt.Tooltip("predicted_headcount:Q", title="Predicha"),
                alt.Tooltip("predicted_deficit_probability:Q", title="Riesgo", format=".3f"),
            ],
        )
        diagonal = pd.DataFrame({"x": [0, max_value + 1], "y": [0, max_value + 1]})
        line = alt.Chart(diagonal).mark_line(strokeDash=[6, 6], color="#64748b").encode(x="x:Q", y="y:Q")
        st.markdown('<div class="section-title">Dotación real vs dotación predicha</div>', unsafe_allow_html=True)
        st.altair_chart((points + line).properties(height=280), width="stretch")
        return

    grouped = df.groupby("date", as_index=False).agg(
        required=(_required_headcount_col(domain), "mean"),
        actual=(_actual_headcount_col(domain), "mean"),
        predicted=(_predicted_headcount_col(domain), "mean"),
    )
    tidy = grouped.melt("date", var_name="series", value_name="headcount")
    chart = (
        alt.Chart(tidy)
        .mark_line(point=True, strokeWidth=2.2)
        .encode(
            x=alt.X("date:T", title="Fecha"),
            y=alt.Y("headcount:Q", title="Dotación"),
            color=alt.Color("series:N", title="Serie", scale=alt.Scale(range=["#ea580c", "#0f766e", "#4338ca"])),
            tooltip=["date:T", "series:N", alt.Tooltip("headcount:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.markdown('<div class="section-title">Dotación requerida vs actual vs predicha</div>', unsafe_allow_html=True)
    st.altair_chart(chart, width="stretch")


def render_segment_chart(domain: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    if domain == "restaurant":
        role_rows = []
        for role in CRITICAL_ROLE_TARGETS:
            col = f"predicted_role_deficit_prob_{role}"
            if col in df.columns:
                role_rows.append({"segment": role, "risk": float(df[col].mean())})
        chart_df = pd.DataFrame(role_rows)
        title = "Riesgo por rol crítico"
        x_title = "Rol"
    elif domain == "clinic":
        chart_df = df.groupby("clinical_unit", as_index=False)["predicted_deficit_probability"].mean().rename(columns={"clinical_unit": "segment", "predicted_deficit_probability": "risk"})
        title = "Riesgo medio por unidad clínica"
        x_title = "Unidad clínica"
    else:
        chart_df = df.groupby("plant_area", as_index=False)["predicted_deficit_probability"].mean().rename(columns={"plant_area": "segment", "predicted_deficit_probability": "risk"})
        title = "Riesgo medio por área"
        x_title = "Área"

    if chart_df.empty:
        return

    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("segment:N", sort="-y", title=x_title),
            y=alt.Y("risk:Q", title="Probabilidad media"),
            color=alt.Color("risk:Q", scale=alt.Scale(scheme="orangered"), legend=None),
            tooltip=["segment:N", alt.Tooltip("risk:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def render_metrics_panel(metrics_payload: dict) -> None:
    st.markdown('<div class="section-title">Detalle de métricas</div>', unsafe_allow_html=True)
    if not metrics_payload:
        st.info("Aún no hay métricas generadas para este dominio.")
        return

    metrics = metrics_payload.get("metrics", {})
    summary_rows = []
    for block_name, block_metrics in metrics.items():
        if block_name == "role_models":
            continue
        if isinstance(block_metrics, dict):
            summary_rows.append({
                "modelo": block_name,
                "AUC": block_metrics.get("AUC-ROC"),
                "F1": block_metrics.get("F1-Score"),
                "Precision": block_metrics.get("Precision"),
                "Recall": block_metrics.get("Recall"),
                "Brier": block_metrics.get("Brier Score"),
                "MAE": block_metrics.get("MAE"),
                "R2": block_metrics.get("R2"),
            })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")

    role_metrics = metrics.get("role_models", {})
    if role_metrics:
        st.markdown('<div class="section-title">Métricas por rol</div>', unsafe_allow_html=True)
        role_df = pd.DataFrame([
            {
                "rol": role,
                "AUC": values.get("AUC-ROC"),
                "F1": values.get("F1-Score"),
                "Precision": values.get("Precision"),
                "Recall": values.get("Recall"),
            }
            for role, values in role_metrics.items()
        ])
        st.dataframe(role_df, width="stretch")

    with st.expander("Ver JSON completo de métricas"):
        st.json(metrics_payload)


def render_predictions_preview(predictions_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Predicciones / scoring</div>', unsafe_allow_html=True)
    if predictions_df.empty:
        st.info("No hay archivo de predicciones generado todavía.")
        return
    st.dataframe(predictions_df.head(50), width="stretch")


def render_artifacts_table(artifacts: DomainArtifacts) -> None:
    records = list_artifacts(artifacts.raw_dir, "raw") + list_artifacts(artifacts.processed_dir, "processed")
    st.markdown('<div class="section-title">Artefactos generados</div>', unsafe_allow_html=True)
    if not records:
        st.info("No hay artefactos todavía.")
        return
    st.dataframe(pd.DataFrame(records), width="stretch")


def render_static_gallery(artifacts: DomainArtifacts) -> None:
    available_images = [path for path in artifacts.dashboard_images if path.exists()]
    with st.expander("Galería de artefactos visuales estáticos"):
        if not available_images:
            st.info("No hay imágenes disponibles para este dominio.")
            return
        selected_name = st.selectbox("Vista estática", [path.name for path in available_images], key=f"gallery_{artifacts.summary_title}")
        selected_path = next(path for path in available_images if path.name == selected_name)
        st.image(str(selected_path), width="stretch")


def render_executive_narrative(domain: str, df: pd.DataFrame) -> None:
    summary_payload = load_optional_json(get_domain_artifacts(domain).risk_shap_summary_json)
    if render_shap_narrative(domain, summary_payload):
        return

    st.markdown('<div class="section-title">Narrativa ejecutiva</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("Sin datos filtrados para generar narrativa.")
        return

    if domain == "restaurant":
        top_period = df.groupby("service_period")["predicted_deficit_probability"].mean().sort_values(ascending=False).index[0]
        top_role = max(CRITICAL_ROLE_TARGETS, key=lambda role: df.get(f"predicted_role_deficit_prob_{role}", pd.Series([0])).mean())
        message = (
            f"La mayor exposición se concentra en la franja `{top_period}`. "
            f"El rol con mayor riesgo relativo es `{top_role}`. "
            "La recomendación operativa es anticipar reemplazos o movimiento de personal cross-trained en los bloques con mayor demanda."
        )
    elif domain == "clinic":
        top_unit = df.groupby(["clinical_unit", "shift"])["predicted_deficit_probability"].mean().sort_values(ascending=False)
        unit_name, shift_name = top_unit.index[0]
        top_role = max(CLINIC_CRITICAL_ROLE_TARGETS, key=lambda role: df.get(f"predicted_role_deficit_prob_{role}", pd.Series([0])).mean())
        message = (
            f"El segmento más crítico es `{unit_name} / {shift_name}`. "
            f"El rol con mayor exposición relativa es `{top_role}`. "
            "Conviene activar staff flotante, revisar sobreagenda y anticipar reemplazos en ventanas de mayor carga asistencial o procedimientos."
        )
    else:
        grouped = df.groupby(["plant_area", "shift"])["predicted_deficit_probability"].mean().sort_values(ascending=False)
        top_area, top_shift = grouped.index[0]
        message = (
            f"El segmento más crítico es `{top_area} / {top_shift}`. "
            "Conviene revisar cobertura programada, fatiga acumulada y ausentismo reciente antes de la siguiente ventana operativa."
        )
    st.info(message)


def render_staffing_definitions() -> None:
    st.markdown('<div class="section-title">Cómo leer la dotación</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Dotación actual</div>
                <div style="color:#334155; font-size:0.95rem; line-height:1.45;">
                    Personal que efectivamente estuvo disponible u operando en la franja analizada.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Dotación requerida</div>
                <div style="color:#334155; font-size:0.95rem; line-height:1.45;">
                    Personal que el negocio necesita para atender la demanda esperada o real sin degradar la operación.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Dotación predicha</div>
                <div style="color:#334155; font-size:0.95rem; line-height:1.45;">
                    Personal que el modelo estima que realmente estará disponible u operando para esa franja.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_domain_specific_layout(domain: str, filtered_df: pd.DataFrame) -> None:
    artifacts = get_domain_artifacts(domain)
    if render_shap_layout(domain, artifacts):
        return

    if domain == "restaurant":
        col_left, col_right = st.columns([1.25, 1])
        with col_left:
            render_trend_chart(filtered_df, domain)
            render_capacity_chart(filtered_df, domain)
        with col_right:
            render_risk_heatmap(domain, filtered_df)
            render_segment_chart(domain, filtered_df)
        return

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        render_trend_chart(filtered_df, domain)
        render_capacity_chart(filtered_df, domain)
    with col_right:
        render_risk_heatmap(domain, filtered_df)
        render_segment_chart(domain, filtered_df)


def main() -> None:
    st.set_page_config(page_title="SkillUp Executive Dashboard", layout="wide")
    _inject_branding()

    with st.sidebar:
        st.header("Control")
        domain = st.selectbox("Dominio", ["industrial", "restaurant", "clinic"], index=0)
        stage = st.selectbox("Stage", ["generate", "etl", "train", "infer", "report", "full"], index=5)
        default_employees = 72 if domain == "restaurant" else (96 if domain == "clinic" else 200)
        employees = st.number_input("Employees", min_value=1, value=default_employees, step=1)
        days = st.number_input("Days", min_value=1, value=180, step=1)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        run_clicked = st.button("Ejecutar pipeline", type="primary", width="stretch")
        refresh_clicked = st.button("Refrescar vista", width="stretch")

    artifacts = get_domain_artifacts(domain)

    if run_clicked:
        with st.spinner("Ejecutando pipeline..."):
            return_code, logs = run_pipeline_command(domain, stage, int(employees), int(days), int(seed))
        st.session_state["last_logs"] = logs
        st.session_state["last_return_code"] = return_code
        if return_code == 0:
            st.success("Pipeline ejecutado correctamente.")
        else:
            st.error(f"El pipeline terminó con código {return_code}.")

    if refresh_clicked:
        st.success("Vista refrescada.")

    predictions_df = load_predictions(artifacts.predictions_csv)
    metrics_payload = load_metrics_payload(artifacts.metrics_json)
    filtered_df = build_filters(domain, predictions_df)

    render_hero(artifacts, domain, filtered_df)
    render_kpis(domain, filtered_df, metrics_payload)

    tab_exec, tab_metrics, tab_data, tab_logs = st.tabs(["Executive View", "Métricas", "Datos", "Logs"])

    with tab_exec:
        render_staffing_definitions()
        render_executive_narrative(domain, filtered_df)
        render_domain_specific_layout(domain, filtered_df)
        render_static_gallery(artifacts)

    with tab_metrics:
        render_metrics_panel(metrics_payload)

    with tab_data:
        col_pred, col_art = st.columns([1.3, 1])
        with col_pred:
            render_predictions_preview(filtered_df)
        with col_art:
            render_artifacts_table(artifacts)

    with tab_logs:
        st.code(st.session_state.get("last_logs", "Aún no se ha ejecutado ningún pipeline desde la UI."), language="text")


if __name__ == "__main__":
    main()
