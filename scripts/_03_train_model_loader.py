"""Adaptador para invocar scripts/03_train_model.py desde otros scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def run() -> None:
    script_path = Path(__file__).resolve().parent / "03_train_model.py"
    spec = importlib.util.spec_from_file_location("skillup_train_model_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el script de entrenamiento: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
