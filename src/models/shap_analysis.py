"""Generacion de artefactos SHAP reutilizables para la UI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap


DOMAIN_FEATURE_LABELS: dict[str, dict[str, str]] = {
    "restaurant": {
        "forecast_covers": "Cubiertos proyectados",
        "actual_covers": "Cubiertos reales",
        "forecast_sales": "Ventas proyectadas",
        "actual_sales": "Ventas reales",
        "reservation_count": "Reservas",
        "delivery_order_volume": "Pedidos delivery",
        "walk_in_ratio": "Clientes sin reserva",
        "promo_flag": "Promocion activa",
        "local_event_flag": "Evento local",
        "avg_stress_score": "Estres promedio del equipo",
        "avg_sleep_duration_hours": "Horas de sueno promedio",
        "avg_sleep_efficiency_pct": "Eficiencia de sueno promedio",
        "avg_fatigue_proxy": "Fatiga promedio",
        "avg_age": "Edad promedio del equipo",
        "avg_bmi": "Carga fisica promedio del equipo",
        "absentee_rate": "Ausentismo",
        "short_notice_absentee_rate": "Ausentismo de corto aviso",
        "forecast_required_headcount_total": "Dotacion requerida proyectada",
        "actual_headcount_total": "Dotacion real",
        "scheduled_overtime_hours": "Horas extra programadas",
        "scheduled_avg_consecutive_work_days": "Racha promedio de dias trabajados",
        "deficit_count_total_lag1": "Deficit del turno comparable anterior",
        "forecast_covers_lag1": "Cubiertos del turno comparable anterior",
        "forecast_covers_lag7": "Cubiertos de la semana anterior",
    },
    "industrial": {
        "required_headcount": "Dotacion requerida",
        "actual_headcount": "Dotacion real",
        "absentee_rate": "Ausentismo",
        "stress_score": "Estres promedio",
        "sleep_duration_hours": "Horas de sueno promedio",
    },
    "clinic": {
        "forecast_patient_volume": "Pacientes proyectados",
        "scheduled_procedures": "Procedimientos agendados",
        "active_care_stations": "Boxes activos",
        "average_wait_minutes": "Espera promedio",
        "respiratory_case_ratio": "Carga respiratoria",
        "absentee_rate": "Ausentismo",
        "avg_stress_score": "Estres promedio del equipo",
        "avg_cognitive_load_score": "Carga cognitiva promedio",
    },
}


def humanize_feature_name(feature_name: str, domain: str) -> str:
    domain_map = DOMAIN_FEATURE_LABELS.get(domain, {})
    if feature_name in domain_map:
        return domain_map[feature_name]

    prefix_map = {
        "service_period_": "Franja ",
        "shift_": "Turno ",
        "plant_area_": "Area ",
        "clinical_unit_": "Unidad ",
        "season_": "Temporada ",
        "avg_": "Promedio de ",
        "scheduled_": "Programado ",
        "actual_": "Real ",
    }
    for prefix, label in prefix_map.items():
        if feature_name.startswith(prefix):
            return f"{label}{feature_name.removeprefix(prefix).replace('_', ' ')}"

    suffix_map = {
        "_lag1": " vs periodo comparable anterior",
        "_lag7": " vs misma ventana semana anterior",
    }
    for suffix, label in suffix_map.items():
        if feature_name.endswith(suffix):
            base_name = feature_name.removesuffix(suffix)
            return f"{humanize_feature_name(base_name, domain)}{label}"

    pretty = feature_name.replace("_pct", " porcentaje")
    pretty = pretty.replace("_rmssd", " RMSSD")
    pretty = pretty.replace("_bpm", " BPM")
    pretty = pretty.replace("_km", " km")
    pretty = pretty.replace("_ms", " ms")
    pretty = pretty.replace("_hrs", " horas")
    pretty = pretty.replace("_", " ")
    return pretty.capitalize()


def _normalize_shap_values(raw_values: object) -> np.ndarray:
    if isinstance(raw_values, list):
        raw_values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    if hasattr(raw_values, "values"):
        raw_values = raw_values.values
    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _build_tree_shap_values(model: object, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    try:
        explanation = explainer(X, check_additivity=False)
        return _normalize_shap_values(explanation)
    except TypeError:
        return _normalize_shap_values(explainer.shap_values(X))


def _build_segment_label(context_df: pd.DataFrame, segment_cols: list[str]) -> pd.Series:
    if not segment_cols:
        return pd.Series(["Global"] * len(context_df), index=context_df.index)
    if len(segment_cols) == 1:
        return context_df[segment_cols[0]].astype(str)
    return context_df[segment_cols].astype(str).agg(" / ".join, axis=1)


def _top_feature_messages(global_df: pd.DataFrame, effect_n: int = 3) -> dict[str, list[dict[str, object]]]:
    positive = global_df[global_df["mean_shap"] > 0].nlargest(effect_n, "mean_abs_shap")
    negative = global_df[global_df["mean_shap"] < 0].nlargest(effect_n, "mean_abs_shap")
    return {
        "drivers_up": [
            {
                "feature": row["feature"],
                "feature_label": row["feature_label"],
                "message": f"Cuando sube '{row['feature_label']}', el riesgo tiende a subir.",
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "mean_shap": float(row["mean_shap"]),
            }
            for _, row in positive.iterrows()
        ],
        "drivers_down": [
            {
                "feature": row["feature"],
                "feature_label": row["feature_label"],
                "message": f"Cuando sube '{row['feature_label']}', el riesgo tiende a bajar.",
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "mean_shap": float(row["mean_shap"]),
            }
            for _, row in negative.iterrows()
        ],
    }


def _select_sample_indices(X: pd.DataFrame, priority_scores: pd.Series | None, max_rows: int) -> pd.Index:
    if len(X) <= max_rows:
        return X.index
    if priority_scores is None:
        return X.sample(n=max_rows, random_state=42).index

    ranked = priority_scores.sort_values(ascending=False)
    priority_n = min(max_rows // 2, len(ranked))
    priority_idx = ranked.head(priority_n).index
    remaining = X.index.difference(priority_idx)
    random_n = max_rows - len(priority_idx)
    random_idx = remaining.to_series().sample(n=random_n, random_state=42, replace=False).values
    return pd.Index(list(priority_idx) + list(random_idx))


def generate_shap_artifacts(
    *,
    domain: str,
    prefix: str,
    output_dir: Path,
    X: pd.DataFrame,
    context_df: pd.DataFrame,
    risk_model: object,
    headcount_model: object,
    risk_priority_scores: pd.Series | None,
    segment_cols: list[str],
    max_rows: int = 500,
    top_features: int = 12,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = _select_sample_indices(X, risk_priority_scores, max_rows=max_rows)
    X_sample = X.loc[sample_idx].copy()
    context_sample = context_df.loc[sample_idx].copy()
    context_sample["segment_label"] = _build_segment_label(context_sample, segment_cols)

    risk_shap = _build_tree_shap_values(risk_model, X_sample)
    risk_global = pd.DataFrame({
        "feature": X_sample.columns,
        "feature_label": [humanize_feature_name(feature, domain) for feature in X_sample.columns],
        "mean_abs_shap": np.abs(risk_shap).mean(axis=0),
        "mean_shap": risk_shap.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    risk_global["impact_direction"] = np.where(risk_global["mean_shap"] >= 0, "Sube riesgo", "Baja riesgo")
    risk_global_path = output_dir / f"{prefix}_risk_shap_global.csv"
    risk_global.to_csv(risk_global_path, index=False)

    risk_top_features = risk_global.head(top_features)["feature"].tolist()
    risk_top_labels = {feature: humanize_feature_name(feature, domain) for feature in risk_top_features}

    risk_segment_rows = []
    grouped = context_sample.groupby("segment_label")
    for segment_label, segment_index in grouped.groups.items():
        segment_positions = X_sample.index.get_indexer(segment_index)
        if len(segment_positions) == 0:
            continue
        segment_shap = risk_shap[segment_positions]
        for feature in risk_top_features:
            feature_pos = X_sample.columns.get_loc(feature)
            risk_segment_rows.append({
                "segment_label": segment_label,
                "feature": feature,
                "feature_label": risk_top_labels[feature],
                "mean_shap": float(segment_shap[:, feature_pos].mean()),
                "mean_abs_shap": float(np.abs(segment_shap[:, feature_pos]).mean()),
            })
    risk_segment_df = pd.DataFrame(risk_segment_rows)
    risk_segment_path = output_dir / f"{prefix}_risk_shap_segment.csv"
    risk_segment_df.to_csv(risk_segment_path, index=False)

    risk_effect_rows = []
    for feature in risk_top_features[:6]:
        feature_pos = X_sample.columns.get_loc(feature)
        feature_df = context_sample.copy()
        feature_df["feature"] = feature
        feature_df["feature_label"] = risk_top_labels[feature]
        feature_df["feature_value"] = X_sample[feature].values
        feature_df["shap_value"] = risk_shap[:, feature_pos]
        risk_effect_rows.append(feature_df)
    risk_effects_df = pd.concat(risk_effect_rows, ignore_index=True) if risk_effect_rows else pd.DataFrame()
    risk_effects_path = output_dir / f"{prefix}_risk_shap_effects.csv"
    risk_effects_df.to_csv(risk_effects_path, index=False)

    explanation_rows = []
    ranked_context = context_sample.copy()
    if risk_priority_scores is not None:
        ranked_context["priority_score"] = risk_priority_scores.loc[sample_idx].values
        ranked_context = ranked_context.sort_values("priority_score", ascending=False)
    for row_index in ranked_context.head(40).index:
        row_pos = X_sample.index.get_loc(row_index)
        row_values = risk_shap[row_pos]
        top_positive_idx = np.argsort(row_values)[-3:][::-1]
        top_negative_idx = np.argsort(row_values)[:3]
        explanation = context_sample.loc[row_index].to_dict()
        for slot, feature_idx in enumerate(top_positive_idx, start=1):
            feature = X_sample.columns[feature_idx]
            explanation[f"top_up_driver_{slot}"] = humanize_feature_name(feature, domain)
            explanation[f"top_up_driver_value_{slot}"] = float(row_values[feature_idx])
        for slot, feature_idx in enumerate(top_negative_idx, start=1):
            feature = X_sample.columns[feature_idx]
            explanation[f"top_down_driver_{slot}"] = humanize_feature_name(feature, domain)
            explanation[f"top_down_driver_value_{slot}"] = float(row_values[feature_idx])
        explanation_rows.append(explanation)
    explanations_df = pd.DataFrame(explanation_rows)
    explanations_path = output_dir / f"{prefix}_risk_shap_explanations.csv"
    explanations_df.to_csv(explanations_path, index=False)

    headcount_shap = _build_tree_shap_values(headcount_model, X_sample)
    headcount_global = pd.DataFrame({
        "feature": X_sample.columns,
        "feature_label": [humanize_feature_name(feature, domain) for feature in X_sample.columns],
        "mean_abs_shap": np.abs(headcount_shap).mean(axis=0),
        "mean_shap": headcount_shap.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    headcount_global["impact_direction"] = np.where(headcount_global["mean_shap"] >= 0, "Sube dotacion", "Baja dotacion")
    headcount_global_path = output_dir / f"{prefix}_headcount_shap_global.csv"
    headcount_global.to_csv(headcount_global_path, index=False)

    risk_summary = _top_feature_messages(risk_global)
    if not risk_segment_df.empty:
        strongest_segment_row = risk_segment_df.iloc[risk_segment_df["mean_abs_shap"].abs().argmax()]
        risk_summary["segment_highlight"] = {
            "segment_label": strongest_segment_row["segment_label"],
            "feature_label": strongest_segment_row["feature_label"],
            "message": f"En {strongest_segment_row['segment_label']}, '{strongest_segment_row['feature_label']}' es uno de los factores mas explicativos del riesgo.",
        }
    summary_path = output_dir / f"{prefix}_risk_shap_summary.json"
    summary_path.write_text(json.dumps(risk_summary, indent=2), encoding="utf-8")

    return {
        "risk_global": risk_global_path,
        "risk_segment": risk_segment_path,
        "risk_effects": risk_effects_path,
        "risk_explanations": explanations_path,
        "headcount_global": headcount_global_path,
        "risk_summary": summary_path,
    }


def generate_role_shap_global_artifacts(
    *,
    domain: str,
    prefix: str,
    output_dir: Path,
    X: pd.DataFrame,
    role_models: dict[str, object],
    priority_scores: pd.Series | None,
    max_rows: int = 500,
    top_features: int = 12,
) -> Path | None:
    if not role_models:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = _select_sample_indices(X, priority_scores, max_rows=max_rows)
    X_sample = X.loc[sample_idx].copy()

    rows = []
    for role, model in role_models.items():
        role_shap = _build_tree_shap_values(model, X_sample)
        role_df = pd.DataFrame({
            "role": role,
            "feature": X_sample.columns,
            "feature_label": [humanize_feature_name(feature, domain) for feature in X_sample.columns],
            "mean_abs_shap": np.abs(role_shap).mean(axis=0),
            "mean_shap": role_shap.mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(top_features)
        role_df["impact_direction"] = np.where(role_df["mean_shap"] >= 0, "Sube deficit de rol", "Baja deficit de rol")
        rows.append(role_df)

    if not rows:
        return None

    output_path = output_dir / f"{prefix}_role_shap_global.csv"
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
    return output_path
