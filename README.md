# SkillUp - Predicción de Dotación Laboral y Riesgo de Déficit

Sistema de predicción de dotación de personal y riesgo de déficit operacional a nivel de **Área-Turno-Día** . El sistema combina datos demográficos con métricas biométricas agregadas provenientes de wearables (ritmo cardíaco, HRV, sueño, estrés y fatiga acumulada) para predecir la disponibilidad real del personal.

---

## Paradigma del Proyecto

1. **Modelado Fisiológico Realista:** Los generadores simulan periodos de enfermedad multi-día precedidos por fases de incubación (1-2 días) con degradación fisiológica (HR más alto, HRV menor, peor calidad de sueño y aumento de estrés).
2. **Nivel de Agregación (Área-Turno-Día):** La planificación industrial requiere saber si la dotación total programada cumplirá con el mínimo operacional (`REQUIRED_STAFF` definido en `config/settings.py`).
3. **Métricas de conjunto:** Las variables biométricas individuales se agregan para representar el estado colectivo de salud, estrés y fatiga del conjunto de trabajadores programados.

---

## Modelos Implementados

Se entrenan y evalúan tres aproximaciones para la toma de decisiones:
1. **Modelo 1: Regresor de Headcount (RandomForest)**
   - **Objetivo:** Predecir exactamente cuántos operadores asistirán en un Área, Turno y Día específico.
   - **Métricas:** MAE, RMSE y $R^2$.
2. **Modelo 2: Clasificador de Riesgo de Déficit (XGBoost)**
   - **Objetivo:** Clasificar si habrá déficit de personal (`actual_headcount < required_headcount`) y dar una probabilidad.
   - **Métricas:** AUC-ROC, PR-AUC, F1-Score, Accuracy, Precision y Recall.
3. **Modelo 3: Ensamble Calibrado (Isotonic Calibrated XGBoost)**
   - **Objetivo:** Calibrar las probabilidades de déficit del modelo XGBoost utilizando validación cruzada. Es vital para que un "70% de riesgo de déficit" en un dashboard o consumidor externo corresponda exactamente a una frecuencia del 70% de casos reales.
   - **Métricas:** Brier Score y Curva de Calibración.

---

## Requisitos e Instalación

- **Python 3.13+**
- **uv**

Para instalar las dependencias y sincronizar el entorno virtual:
```bash
uv sync
```

---

## Guía de Ejecución

Puedes ejecutar el pipeline completo de punta a punta con un solo comando:

```bash
uv run python scripts/04_full_pipeline.py
```

Este script orquesta los siguientes pasos de forma secuencial:
1. **Generación de Datos (`01_generate_data.py`):** Crea perfiles demográficos, registros de turnos de trabajo, series temporales biométricas (wearables) y eventos de ausentismo médico/casual en `data/raw/`.
2. **ETL y Feature Engineering (`02_run_etl.py`):** Limpia biometría, imputa datos vacíos con KNN, normaliza (Z-Scores individuales), crea ventanas móviles (fatiga de 14 días), agrega todo a nivel de Área-Turno-Día y guarda la matriz resultante en `ml_features` dentro de la base de datos SQLite `data/skillup.db`.
3. **Entrenamiento y Evaluación (`03_train_model.py`):** Aplica validación temporal (split ordenado por fecha), entrena los 3 modelos descritos, optimiza los umbrales de decisión basados en el conjunto de entrenamiento, evalúa en test y genera gráficos de reporte en `data/processed/`.

---

## Visualización de Resultados y Métricas

Una vez que ejecutas el pipeline, los resultados de desempeño se guardan en el directorio `data/processed/`. Puedes abrir e inspeccionar los siguientes archivos:

### Tabla Comparativa de Métricas
- El archivo **[model_comparison_metrics.csv](data/processed/model_comparison_metrics.csv)** resume el rendimiento de los modelos de clasificación (Modelo 2 vs Modelo 3).

### Gráficos de Rendimiento y Negocio
- **[headcount_actual_vs_predicted.png](data/processed/headcount_actual_vs_predicted.png):** Gráfico de dispersión para el **Modelo 1** comparando la dotación real vs. la predicha. La cercanía a la línea diagonal roja indica precisión perfecta.
- **[calibration_curve_staffing.png](data/processed/calibration_curve_staffing.png):** Curva de calibración. Compara el **Modelo 2** (XGBoost directo) y el **Modelo 3** (Calibrado). Muestra cómo la calibración isotónica alinea las probabilidades predichas con la realidad del negocio.
- **[roc_curve_staffing.png](data/processed/roc_curve_staffing.png):** Curva ROC para comparar la capacidad de discriminación de los Modelos 2 y 3.
- **[pr_curve_staffing.png](data/processed/pr_curve_staffing.png):** Curva Precision-Recall, crítica para conjuntos de datos donde el déficit es un evento desbalanceado.
- **[feature_importance_staffing.png](data/processed/feature_importance_staffing.png):** Gráfico de barras con las 20 variables más importantes para predecir el déficit (por ejemplo, fatiga colectiva, promedio de estrés, turnos específicos y lags históricos).

---

## Pruebas Unitarias (Tests)

Para ejecutar la suite de pruebas unitarias que validan la lógica de simulación, la degradación biométrica y el cálculo agregación/déficit:

```bash
uv run pytest tests/ -v
```

---

## Estructura del Proyecto

```
skillup/
├── config/
│   └── settings.py              # Constantes, límites de dotación por área
├── data/
│   ├── raw/                     # Datos sintéticos generados (CSVs)
│   ├── processed/               # Gráficas de métricas y CSV de comparación
│   └── skillup.db               # Base de datos SQLite
├── scripts/
│   ├── 01_generate_data.py      # Orquestador de generación de datos
│   ├── 02_run_etl.py            # Orquestador ETL
│   ├── 03_train_model.py        # Orquestador de entrenamiento de modelos
│   └── 04_full_pipeline.py      # Pipeline completo de punta a punta
├── src/
│   ├── generators/              # Lógica de generación sintética
│   │   ├── employees.py         # Demográficos
│   │   ├── work_records.py      # Calendario y turnos
│   │   ├── biometrics.py        # Series de wearables (incubación/enfermedad)
│   │   └── absenteeism.py       # Registro de ausentismo médico y casual
│   ├── etl/                     # Pipeline de datos
│   │   ├── extract.py           # Extractor de CSV
│   │   ├── transform.py         # Limpieza, normalización y agregación Área-Turno-Día
│   │   └── load.py              # Carga e importación a SQLite con esquemas
│   ├── models/                  # Lógica de Machine Learning
│   │   ├── features.py          # Preparación de matriz X, y para dotación
│   │   ├── staffing_models.py   # Definición de Modelos 1, 2 y 3 y métricas
│   │   ├── train.py             # Funciones auxiliares de entrenamiento
│   │   └── evaluate.py          # Visualizaciones y reporte
│   └── utils/                   # Herramientas
│       ├── database.py          # Conectores SQLite
│       └── validators.py        # Validadores de rangos fisiológicos
├── tests/
│   ├── test_generators.py       # Tests heredados de generación
│   └── test_staffing.py         # Tests agregados de simulación y dotación
├── pyproject.toml               # Dependencias y configuración de Pyright
└── pyrightconfig.json           # Configuración del servidor de lenguajes
```
