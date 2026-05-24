"""Pipeline completo: generacion -> ETL -> entrenamiento."""

import logging
import runpy
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SEP = "-" * 60


def run_step(step_label: str, script_path: Path) -> None:
    """Ejecuta un script como paso del pipeline."""
    logger.info(_SEP)
    logger.info("%s", step_label)
    logger.info(_SEP)
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        logger.info("Completado: %s", step_label)
    except Exception as exc:
        logger.error("Error en '%s': %s", step_label, exc)
        raise


def main():
    print("Cargando...")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETO - SkillUp Dotacion Laboral")
    logger.info("=" * 60)

    run_step("PASO 1/3 - Generacion de datos sinteticos", SCRIPTS_DIR / "01_generate_data.py")
    run_step("PASO 2/3 - ETL y feature engineering",       SCRIPTS_DIR / "02_run_etl.py")
    run_step("PASO 3/3 - Entrenamiento de modelos",        SCRIPTS_DIR / "03_train_model.py")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)

print("=========")

main()
