"""Pipeline completo: generacion -> ETL -> entrenamiento."""

import sys
import logging
import runpy
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_step(step_name, script_path):
    """Ejecuta un script como paso del pipeline."""
    logger.info(f"Iniciando: {step_name}")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        logger.info(f"Completado: {step_name}")
    except Exception as e:
        logger.error(f"Error en {step_name}: {e}")
        raise


def main():
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETO - SkillUp")
    logger.info("=" * 60)

    # Paso 1: Generacion de datos sinteticos
    run_step("Generacion de datos sinteticos", SCRIPTS_DIR / "01_generate_data.py")

    # Paso 2: ETL
    run_step("ETL y feature engineering", SCRIPTS_DIR / "02_run_etl.py")

    # Paso 3: Entrenamiento de modelos
    run_step("Entrenamiento de modelos", SCRIPTS_DIR / "03_train_model.py")

    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
