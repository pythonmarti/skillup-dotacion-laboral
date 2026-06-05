"""Generador de fichas medicas PDF con layout amigable para OCR."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

EMPLOYEE_COLUMNS = [
    "employee_id",
    "name",
    "age",
    "gender",
    "bmi",
    "education_level",
    "plant_area",
    "position",
    "seniority_years",
    "shift_pattern",
    "distance_to_work_km",
    "children",
    "social_drinker",
    "smoker",
    "hire_date",
]

SECTION_LAYOUT = [
    ("Identificacion", ["employee_id", "name", "age", "gender"]),
    ("Perfil laboral", ["plant_area", "position", "shift_pattern", "seniority_years", "hire_date"]),
    ("Datos personales relevantes", ["bmi", "education_level", "distance_to_work_km", "children"]),
    ("Habitos", ["social_drinker", "smoker"]),
]

SECTION_BOXES = {
    "Identificacion": (0.07, 0.76, 0.86, 0.13),
    "Perfil laboral": (0.07, 0.47, 0.86, 0.25),
    "Datos personales relevantes": (0.07, 0.28, 0.86, 0.19),
    "Habitos": (0.07, 0.155, 0.86, 0.075),
    "Observaciones medicas": (0.07, 0.085, 0.86, 0.055),
}


def _normalize_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _draw_field_row(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    value: str,
) -> None:
    ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, linewidth=0.45, color="black"))
    divider_x = x + (width * 0.32)
    ax.plot([divider_x, divider_x], [y, y + height], color="black", linewidth=0.4)
    ax.text(x + 0.01, y + (height / 2), label, fontsize=7.5, fontweight="bold", ha="left", va="center")
    ax.text(divider_x + 0.01, y + (height / 2), value, fontsize=7.5, fontweight="bold", ha="left", va="center")


def _draw_section(ax: plt.Axes, title: str, columns: list[str], employee: pd.Series, box: tuple[float, float, float, float]) -> None:
    x, y, width, height = box
    ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, linewidth=0.8, color="black"))
    title_y = y + height - 0.022
    divider_y = y + height - 0.05
    ax.text(x + 0.012, title_y, title, fontsize=10, fontweight="bold", ha="left", va="center")
    ax.plot([x, x + width], [divider_y, divider_y], color="black", linewidth=0.5)

    inner_x = x + 0.01
    inner_width = width - 0.02
    top_padding = 0.012
    bottom_padding = 0.012
    available_height = max(0.02, divider_y - y - top_padding - bottom_padding)
    row_height = available_height / max(len(columns), 1)

    for index, column in enumerate(columns):
        row_y = divider_y - top_padding - ((index + 1) * row_height)
        _draw_field_row(ax, inner_x, row_y, inner_width, row_height, f"{column}:", _normalize_value(employee[column]))


def _draw_habits_section(ax: plt.Axes, employee: pd.Series, box: tuple[float, float, float, float]) -> None:
    x, y, width, height = box
    ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, linewidth=0.8, color="black"))
    title_y = y + height - 0.015
    divider_y = y + height - 0.033
    ax.text(x + 0.012, title_y, "Habitos", fontsize=10, fontweight="bold", ha="left", va="center")
    ax.plot([x, x + width], [divider_y, divider_y], color="black", linewidth=0.5)

    bottom_padding = 0.006
    top_padding = 0.004
    row_height = (divider_y - y - top_padding - bottom_padding) / 2
    labels = [("social_drinker", bool(employee["social_drinker"])), ("smoker", bool(employee["smoker"]))]

    for index, (label, value) in enumerate(labels):
        row_y = divider_y - top_padding - ((index + 1) * row_height)
        value_text = "(X) Si    ( ) No" if value else "( ) Si    (X) No"
        _draw_field_row(ax, x + 0.01, row_y, width - 0.02, row_height, f"{label}:", value_text)


def _build_ocr_block(employee: pd.Series) -> str:
    lines = ["STRUCTURED_DATA_START", "template_version: 1"]
    for column in EMPLOYEE_COLUMNS:
        lines.append(f"{column}: {_normalize_value(employee[column])}")
    lines.append("STRUCTURED_DATA_END")
    return "\n".join(lines)


def create_medical_form(employee: pd.Series, output_path: Path, include_structured_page: bool = False) -> Path:
    """Renderiza una ficha medica PDF individual para un empleado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context({"pdf.compression": 0, "pdf.fonttype": 42}):
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            ax.text(0.5, 0.965, "FICHA MEDICA LABORAL", fontsize=18, fontweight="bold", ha="center")
            ax.text(0.5, 0.942, "Formato estandar para captura de datos", fontsize=9, fontweight="bold", ha="center")
            ax.text(0.07, 0.915, "template_version: 1", fontsize=9, family="monospace")
            ax.add_patch(plt.Rectangle((0.05, 0.055), 0.90, 0.88, fill=False, linewidth=1.0, color="black"))

            for section_title, columns in SECTION_LAYOUT:
                if section_title == "Habitos":
                    _draw_habits_section(ax, employee, SECTION_BOXES[section_title])
                else:
                    _draw_section(ax, section_title, columns, employee, SECTION_BOXES[section_title])

            obs_x, obs_y, obs_width, obs_height = SECTION_BOXES["Observaciones medicas"]
            ax.add_patch(plt.Rectangle((obs_x, obs_y), obs_width, obs_height, fill=False, linewidth=0.8, color="black"))
            obs_title_y = obs_y + obs_height - 0.015
            obs_divider_y = obs_y + obs_height - 0.030
            ax.text(obs_x + 0.012, obs_title_y, "Observaciones medicas", fontsize=10, fontweight="bold", ha="left", va="center")
            ax.plot([obs_x, obs_x + obs_width], [obs_divider_y, obs_divider_y], color="black", linewidth=0.5)
            for line_y in [obs_y + 0.010, obs_y + 0.020]:
                ax.plot([obs_x + 0.01, obs_x + obs_width - 0.02], [line_y, line_y], color="black", linewidth=0.5)

            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            if include_structured_page:
                ocr_fig = plt.figure(figsize=(8.27, 11.69))
                ocr_ax = ocr_fig.add_axes([0, 0, 1, 1])
                ocr_ax.set_xlim(0, 1)
                ocr_ax.set_ylim(0, 1)
                ocr_ax.axis("off")

                ocr_ax.text(0.5, 0.965, "DATOS ESTRUCTURADOS", fontsize=18, fontweight="bold", ha="center")
                ocr_ax.text(0.5, 0.942, "Pagina estructurada para digitalizacion", fontsize=10, ha="center")
                ocr_ax.add_patch(plt.Rectangle((0.05, 0.055), 0.90, 0.88, fill=False, linewidth=1.0, color="black"))
                ocr_ax.text(0.08, 0.90, _build_ocr_block(employee), fontsize=12, family="monospace", ha="left", va="top")

                pdf.savefig(ocr_fig, dpi=300)
                plt.close(ocr_fig)

    return output_path


def generate_medical_forms(
    employees_df: pd.DataFrame,
    output_dir: Path,
    include_structured_page: bool = False,
) -> list[Path]:
    """Genera una ficha medica PDF por empleado."""
    missing_columns = [column for column in EMPLOYEE_COLUMNS if column not in employees_df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en employees_df: {missing_columns}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for _, employee in employees_df.iterrows():
        employee_id = _normalize_value(employee["employee_id"])
        output_path = output_dir / f"{employee_id}_medical_form.pdf"
        generated_files.append(create_medical_form(employee, output_path, include_structured_page=include_structured_page))

    logger.info("Fichas medicas generadas: %d PDFs en %s", len(generated_files), output_dir)
    return generated_files


def generate_medical_forms_from_csv(
    input_csv: Path,
    output_dir: Path,
    include_structured_page: bool = False,
) -> list[Path]:
    """Carga employees.csv y genera las fichas medicas PDF."""
    employees_df = pd.read_csv(input_csv)
    return generate_medical_forms(employees_df, output_dir, include_structured_page=include_structured_page)
