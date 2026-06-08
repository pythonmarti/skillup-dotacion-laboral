#!/usr/bin/env python3
"""Lanza la UI interactiva con Streamlit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "src" / "ui" / "dashboard_app.py"


if __name__ == "__main__":
    raise SystemExit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
            cwd=str(PROJECT_ROOT),
            check=False,
        ).returncode
    )
