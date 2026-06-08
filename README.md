# SkillUp - Predicción de Dotación y Riesgo de Déficit

SkillUp es una plataforma de simulación, ETL, modelado, inferencia y visualización para problemas de dotación laboral. Actualmente soporta dos dominios operativos:

- `industrial`: cobertura por `plant_area + shift + date`
- `restaurant`: cobertura por `date + service_period` para un restaurante casual dining con calendario de Chile

Además incluye un flujo documental de fichas médicas en PDF, extracción a CSV estructurado y una UI interactiva para operar pipelines y revisar dashboards.

---

## Requisitos

- Python `3.13+`
- `uv`

Instalación:

```bash
uv sync
```

---

## Dominios Disponibles

### Industrial

- unidad de decisión: `Área + Turno + Día`
- foco: disponibilidad real, fatiga colectiva, ausentismo y déficit operacional
- salidas: métricas de modelos, predicciones, artefactos entrenados y gráficos de evaluación

### Restaurant

- unidad de decisión: `Fecha + Franja horaria`
- franjas: `11_13`, `13_15`, `19_21`, `21_23`
- foco: horas peak, fines de semana, festivos chilenos, déficit general y por rol crítico
- salidas: predicciones operativas, recomendaciones de staffing y dashboard ejecutivo

---

## Ejecución por Dominio

El proyecto se opera con un selector de dominio y stages estandarizados.

Comando base:

```bash
uv run python scripts/run_pipeline.py --domain <industrial|restaurant> --stage <generate|etl|train|infer|report|full>
```

Listar dominios:

```bash
uv run python scripts/run_pipeline.py --list-domains
```

Stages disponibles:

- `generate`
- `etl`
- `train`
- `infer`
- `report`
- `full`

### Pipeline completo industrial

```bash
uv run python scripts/run_pipeline.py --domain industrial --stage full
```

Atajos equivalentes:

```bash
uv run python scripts/run_full_pipeline.py --domain industrial
uv run python scripts/04_full_pipeline.py --domain industrial
```

### Pipeline completo restaurant

```bash
uv run python scripts/run_pipeline.py --domain restaurant --stage full --employees 72 --days 180 --seed 42
```

### Ejemplos por stage

```bash
uv run python scripts/run_pipeline.py --domain industrial --stage generate
uv run python scripts/run_pipeline.py --domain industrial --stage etl
uv run python scripts/run_pipeline.py --domain industrial --stage train
uv run python scripts/run_pipeline.py --domain industrial --stage infer
uv run python scripts/run_pipeline.py --domain restaurant --stage report
```

---

## Qué hace cada flujo

### Industrial

`full` ejecuta:

1. generación de `employees.csv`, `work_records.csv`, `biometrics.csv`, `absenteeism.csv`
2. ETL y creación de `ml_features`
3. entrenamiento de modelos de dotación y déficit
4. inferencia sobre `ml_features`

Salida principal en:

- `data/raw/`
- `data/processed/`
- `data/skillup.db`

### Restaurant

`full` ejecuta:

1. generación de datos en `data/restaurant/raw/`
2. ETL y creación de `restaurant_ml_features`
3. entrenamiento de modelos generales y por rol crítico
4. inferencia operativa
5. dashboard ejecutivo en `data/restaurant/processed/restaurant_executive_dashboard.png`

Salida principal en:

- `data/restaurant/raw/`
- `data/restaurant/processed/`
- `data/restaurant/restaurant_skillup.db`

---

## UI Interactiva

La aplicación incluye una UI con Streamlit para operar pipelines y visualizar dashboards interactivos.

Lanzamiento:

```bash
uv run python scripts/run_dashboard_ui.py
```

La UI permite:

- seleccionar `industrial` o `restaurant`
- ejecutar pipelines por dominio y stage
- revisar logs de ejecución
- visualizar KPIs y gráficos interactivos
- consultar métricas, predicciones y artefactos por dominio
- abrir una galería secundaria de imágenes estáticas generadas

---

## Flujo de Fichas Médicas

### Generar PDFs

```bash
uv run python scripts/05_generate_medical_forms.py
```

### Extraer PDFs a CSV

```bash
uv run python scripts/06_extract_medical_forms.py
```

### Validar contra `employees.csv`

```bash
uv run python scripts/06_extract_medical_forms.py --validate-against data/raw/employees.csv
```

### Ejecutar ETL usando el CSV extraído

```bash
uv run python scripts/02_run_etl.py --employees-path data/raw/employees_from_forms.csv
```

### Flujo unificado PDF -> CSV -> ETL

```bash
uv run python scripts/07_forms_to_etl.py
```

### Flujo unificado PDF -> CSV -> ETL -> métricas del modelo

```bash
uv run python scripts/08_forms_to_model_metrics.py
```

### Inferencia con artefactos entrenados

```bash
uv run python scripts/09_run_inference.py
```

---

## Artefactos Relevantes

### Industrial

Ubicación: `data/processed/`

- `model_comparison_metrics.csv`
- `headcount_regressor.pkl`
- `deficit_classifier_xgboost.pkl`
- `deficit_classifier_calibrated.pkl`
- `model_artifacts.json`
- `staffing_inference_predictions.csv`
- `staffing_inference_metrics.json`
- gráficos de evaluación (`roc`, `pr`, `calibration`, `feature_importance`, etc.)

### Restaurant

Ubicación: `data/restaurant/processed/`

- `restaurant_ml_features.csv`
- `restaurant_model_metrics.csv`
- `restaurant_model_artifacts.json`
- `restaurant_staffing_predictions.csv`
- `restaurant_staffing_metrics.json`
- `restaurant_executive_dashboard.png`
- modelos por rol crítico y modelo general

---

## Cómo Leer la Dotación

- **Dotación actual**: personal que efectivamente estuvo disponible u operando.
- **Dotación requerida**: personal que el negocio necesita para atender la demanda sin degradar la operación.
- **Dotación predicha**: personal que el modelo estima que estará disponible.

Interpretación operativa:

```text
si dotación predicha < dotación requerida
=> existe riesgo de déficit esperado
```

---

## Documentación Técnica

Consulta:

- `docs/arquitectura_validada.md`
- `docs/pipeline_actualizado.md`

El segundo documento explica la arquitectura multi-dominio actual, el flujo industrial, el flujo restaurant, la UI y el pipeline de fichas médicas.

---

## Estructura del Proyecto

```text
skillup/
├── config/
│   ├── settings.py
│   └── restaurant_settings.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── restaurant/
│   └── skillup.db
├── docs/
│   ├── arquitectura_validada.md
│   └── pipeline_actualizado.md
├── scripts/
│   ├── run_pipeline.py
│   ├── run_full_pipeline.py
│   ├── run_dashboard_ui.py
│   ├── 01_generate_data.py
│   ├── 02_run_etl.py
│   ├── 03_train_model.py
│   ├── 04_full_pipeline.py
│   ├── 05_generate_medical_forms.py
│   ├── 06_extract_medical_forms.py
│   ├── 07_forms_to_etl.py
│   ├── 08_forms_to_model_metrics.py
│   └── 09_run_inference.py
├── src/
│   ├── domains/
│   ├── etl/
│   ├── extraction/
│   ├── generators/
│   ├── models/
│   ├── restaurant/
│   ├── ui/
│   └── utils/
└── tests/
```
