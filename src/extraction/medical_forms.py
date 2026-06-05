"""Extraccion de campos desde fichas medicas PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.generators.medical_forms import EMPLOYEE_COLUMNS

SECTION_HEADERS = {
    "Identificacion",
    "Perfil laboral",
    "Datos personales relevantes",
    "Habitos",
    "Observaciones medicas",
    "FICHA MEDICA LABORAL",
    "Formato estandar para captura de datos",
}


def _run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "sin detalles"
        raise RuntimeError(f"Fallo ejecutando {' '.join(command)}: {stderr}")
    return result.stdout


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrae texto nativo desde un PDF usando pdftotext."""
    if shutil.which("pdftotext") is None:
        raise RuntimeError("No se encontro 'pdftotext' en el sistema")
    return _run_command(["pdftotext", str(pdf_path), "-"])


def extract_text_with_ocr(pdf_path: Path) -> str:
    """Fallback OCR para PDFs escaneados cuando tesseract este disponible."""
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "El PDF no contiene texto util y no hay OCR disponible. Instala 'tesseract' para habilitar el fallback."
        )
    raise RuntimeError("OCR fallback aun no esta habilitado en este entorno porque falta una utilidad de rasterizacion")


def _normalize_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_labeled_value(lines: list[str], label: str) -> str:
    target = f"{label}:"
    for index, line in enumerate(lines):
        if line == target:
            values: list[str] = []
            for candidate in lines[index + 1:]:
                if candidate.endswith(":"):
                    break
                if candidate in SECTION_HEADERS:
                    break
                values.append(candidate)
                if label not in {"social_drinker", "smoker"}:
                    break
            if values:
                return " ".join(values)
    raise ValueError(f"No se encontro valor para el campo '{label}'")


def _parse_boolean(raw_value: str) -> bool:
    normalized = raw_value.strip()
    if normalized in {"True", "False"}:
        return normalized == "True"
    if "(X) Si" in normalized:
        return True
    if "(X) No" in normalized:
        return False
    raise ValueError(f"No se pudo interpretar booleano: {raw_value}")


def parse_medical_form_text(text: str) -> dict[str, object]:
    """Parsea el texto extraido de una ficha medica al esquema employees.csv."""
    lines = _normalize_lines(text)
    record: dict[str, object] = {}

    for column in EMPLOYEE_COLUMNS:
        raw_value = _extract_labeled_value(lines, column)
        if column in {"age", "education_level", "seniority_years", "children"}:
            record[column] = int(raw_value)
        elif column in {"bmi", "distance_to_work_km"}:
            record[column] = float(raw_value)
        elif column in {"social_drinker", "smoker"}:
            record[column] = _parse_boolean(raw_value)
        else:
            record[column] = raw_value

    return record


def extract_medical_form(pdf_path: Path, allow_ocr_fallback: bool = True) -> dict[str, object]:
    """Extrae un registro estructurado desde un PDF de ficha medica."""
    text = extract_text_from_pdf(pdf_path)
    if text.strip():
        return parse_medical_form_text(text)
    if allow_ocr_fallback:
        return parse_medical_form_text(extract_text_with_ocr(pdf_path))
    raise RuntimeError(f"No se pudo extraer texto del PDF {pdf_path}")


def extract_medical_forms_to_dataframe(pdf_paths: list[Path], allow_ocr_fallback: bool = True) -> pd.DataFrame:
    """Extrae multiples PDFs a un DataFrame estructurado."""
    records = [extract_medical_form(path, allow_ocr_fallback=allow_ocr_fallback) for path in pdf_paths]
    return pd.DataFrame(records, columns=EMPLOYEE_COLUMNS)


def extract_medical_forms_dir_to_csv(
    input_dir: Path,
    output_csv: Path,
    pattern: str = "*.pdf",
    allow_ocr_fallback: bool = True,
) -> pd.DataFrame:
    """Procesa un directorio de fichas medicas PDF y exporta un CSV compatible con employees.csv."""
    pdf_paths = sorted(input_dir.glob(pattern))
    if not pdf_paths:
        raise ValueError(f"No se encontraron PDFs en {input_dir} con patron {pattern}")

    df = extract_medical_forms_to_dataframe(pdf_paths, allow_ocr_fallback=allow_ocr_fallback)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def validate_employees_csv_match(reference_csv: Path, candidate_csv: Path) -> None:
    """Valida que dos CSVs de empleados sean exactamente iguales."""
    reference_df = pd.read_csv(reference_csv, parse_dates=["hire_date"])
    candidate_df = pd.read_csv(candidate_csv, parse_dates=["hire_date"])

    reference_df = reference_df[EMPLOYEE_COLUMNS].reset_index(drop=True)
    candidate_df = candidate_df[EMPLOYEE_COLUMNS].reset_index(drop=True)

    try:
        assert_frame_equal(candidate_df, reference_df, check_dtype=True, check_like=False)
    except AssertionError as exc:
        raise ValueError(
            f"El CSV extraido no coincide exactamente con el archivo de referencia '{reference_csv}'. {exc}"
        ) from exc
