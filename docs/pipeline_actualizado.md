# Pipeline Actualizado

## 1. Resumen

El proyecto ya no es un pipeline único. Hoy funciona como una plataforma **multi-dominio** con runner común y stages estandarizados.

Dominios activos:

- `industrial`
- `restaurant`

Además existen dos capacidades transversales:

- flujo documental de fichas médicas PDF
- UI interactiva con Streamlit

## 2. Punto de entrada común

Comando principal:

```bash
uv run python scripts/run_pipeline.py --domain <industrial|restaurant> --stage <generate|etl|train|infer|report|full>
```

Atajo para pipeline completo:

```bash
uv run python scripts/run_full_pipeline.py --domain industrial
```

Compat wrapper histórico:

```bash
uv run python scripts/04_full_pipeline.py --domain industrial
```

## 3. Stages

- `generate`: genera datos raw del dominio
- `etl`: transforma los datos y crea la tabla de features
- `train`: entrena modelos y persiste artefactos
- `infer`: genera predicciones y métricas de scoring
- `report`: genera dashboard ejecutivo o reporte visual
- `full`: ejecuta la cadena completa

## 4. Capa de dominios

Archivos principales:

- `src/domains/base.py`
- `src/domains/registry.py`
- `src/domains/cli.py`
- `src/domains/industrial.py`
- `src/domains/restaurant.py`

Responsabilidad:

- registrar dominios disponibles
- mapear stages a funciones concretas
- permitir expansión a nuevos casos sin tocar los existentes

## 5. Dominio industrial

### 5.1 Unidad de decisión

```text
plant_area + shift + date
```

### 5.2 Flujo funcional

1. `generate`
   - crea `employees.csv`
   - crea `work_records.csv`
   - crea `biometrics.csv`
   - crea `absenteeism.csv`

2. `etl`
   - limpia biometría
   - agrega variables a nivel área-turno-día
   - persiste `ml_features` en SQLite

3. `train`
   - entrena regresor de headcount
   - entrena clasificador de déficit
   - entrena clasificador calibrado
   - guarda artefactos en `data/processed/`

4. `infer`
   - genera `staffing_inference_predictions.csv`
   - genera `staffing_inference_metrics.json`

### 5.3 Artefactos principales

Ubicación:

```text
data/processed/
```

Archivos relevantes:

- `headcount_regressor.pkl`
- `deficit_classifier_xgboost.pkl`
- `deficit_classifier_calibrated.pkl`
- `model_artifacts.json`
- `staffing_inference_predictions.csv`
- `staffing_inference_metrics.json`
- `model_comparison_metrics.csv`

## 6. Dominio restaurant

### 6.1 Unidad de decisión

```text
date + service_period
```

### 6.2 Configuración del dominio

Archivo:

- `config/restaurant_settings.py`

Define:

- directorios del dominio
- base SQLite del dominio
- calendario Chile
- roles críticos
- franjas horarias
- configuración del modelado

### 6.3 Franjas horarias

- `11_13`
- `13_15`
- `19_21`
- `21_23`

### 6.4 Datos raw generados

Ubicación:

```text
data/restaurant/raw/
```

Archivos:

- `restaurant_employees.csv`
- `restaurant_calendar_cl.csv`
- `restaurant_demand.csv`
- `restaurant_work_records.csv`
- `restaurant_biometrics.csv`
- `restaurant_absenteeism.csv`

### 6.5 ETL restaurant

Archivo:

- `src/restaurant/etl.py`

Salida:

- `restaurant_ml_features`

Persistencia:

- CSV en `data/restaurant/processed/restaurant_ml_features.csv`
- SQLite en `data/restaurant/restaurant_skillup.db`

### 6.6 Modelos restaurant

Archivo:

- `src/restaurant/train.py`

Modelos:

- regresor de dotación total
- clasificador general de déficit
- clasificador general calibrado
- clasificadores por rol crítico

Roles críticos actuales:

- `garzon`
- `cocinero_linea`
- `jefe_turno`

### 6.7 Inferencia restaurant

Archivo:

- `src/restaurant/inference.py`

Salidas:

- `restaurant_staffing_predictions.csv`
- `restaurant_staffing_metrics.json`

Incluye:

- dotación predicha
- probabilidad de déficit general
- riesgo de déficit por rol crítico
- recomendación operativa textual

### 6.8 Dashboard ejecutivo restaurant

Archivo:

- `src/restaurant/reporting.py`

Salida:

- `restaurant_executive_dashboard.png`

Objetivo:

- entregar una vista de una plana para decisiones de staffing
- resaltar peak hours, riesgo, déficit por rol y señales fisiológicas

## 7. Flujo documental de fichas médicas

Capacidades:

- generar fichas médicas PDF
- extraer campos desde PDFs a CSV compatible con `employees.csv`
- validar equivalencia exacta con el CSV base
- integrar ese CSV al ETL

Archivos clave:

- `src/generators/medical_forms.py`
- `src/extraction/medical_forms.py`

Scripts:

- `05_generate_medical_forms.py`
- `06_extract_medical_forms.py`
- `07_forms_to_etl.py`
- `08_forms_to_model_metrics.py`
- `09_run_inference.py`

## 8. UI interactiva

La UI está implementada con Streamlit.

Archivos:

- `src/ui/dashboard_app.py`
- `scripts/run_dashboard_ui.py`

Comando:

```bash
uv run python scripts/run_dashboard_ui.py
```

Capacidades:

- seleccionar dominio
- ejecutar pipelines por stage
- revisar logs
- ver KPIs ejecutivos
- explorar gráficos interactivos
- consultar predicciones y métricas
- revisar artefactos raw y processed

## 9. Cómo interpretar la dotación

- **Dotación actual**: personal que efectivamente estuvo disponible u operando.
- **Dotación requerida**: personal que el negocio necesita para cubrir la demanda.
- **Dotación predicha**: personal que el modelo estima que estará disponible.

Regla práctica:

```text
si dotación predicha < dotación requerida
=> existe riesgo de déficit esperado
```

## 10. Comandos recomendados

### Industrial completo

```bash
uv run python scripts/run_pipeline.py --domain industrial --stage full
```

### Restaurant completo

```bash
uv run python scripts/run_pipeline.py --domain restaurant --stage full --employees 72 --days 180 --seed 42
```

### Dashboard restaurant

```bash
uv run python scripts/run_pipeline.py --domain restaurant --stage report
```

### UI local

```bash
uv run python scripts/run_dashboard_ui.py
```

## 11. Estado actual del proyecto

Estado operativo hoy:

- runner multi-dominio implementado
- dominio industrial operativo
- dominio restaurant operativo
- flujo de fichas médicas operativo
- UI interactiva operativa

La arquitectura ya está preparada para incorporar un tercer dominio sin rediseñar el core del sistema.
