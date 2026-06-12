# Arquitectura Validada

## 1. Estado actual del proyecto

SkillUp ya no debe entenderse como un pipeline único. La arquitectura vigente es **multi-dominio** y soporta dos verticales operativas:

- `industrial`
- `restaurant`

Ambos dominios comparten una base común de ejecución, persistencia y visualización, pero cada uno tiene:

- generación de datos propia
- ETL propio
- modelado propio
- inferencia propia
- artefactos propios

La orquestación se realiza mediante un runner común.

---

## 2. Punto de entrada real

El entrypoint recomendado hoy para iniciar el repositorio es:

```bash
make up
```

Esto levanta la UI de Streamlit en Docker sobre `http://localhost:8501`.

Para ejecutar pipelines desde Docker:

```bash
make pipeline DOMAIN=industrial STAGE=full
make pipeline DOMAIN=restaurant STAGE=full ARGS="--employees 72 --days 180 --seed 42"
```

El entrypoint equivalente en local es:

```bash
uv run python scripts/run_pipeline.py --domain <industrial|restaurant> --stage <generate|etl|train|infer|report|full>
```

Atajo para pipeline completo:

```bash
uv run python scripts/run_full_pipeline.py --domain industrial
```

Compatibilidad histórica:

```bash
uv run python scripts/04_full_pipeline.py --domain industrial
```

Esto reemplaza la idea antigua de que `scripts/04_full_pipeline.py` era el centro único del sistema.

---

## 3. Capa de orquestación

La capa de orquestación está en:

- `src/domains/base.py`
- `src/domains/registry.py`
- `src/domains/cli.py`

Responsabilidades:

- registrar dominios disponibles
- resolver el dominio solicitado por CLI
- exponer stages estándar
- ejecutar pipelines completos o parciales

Stages soportados:

- `generate`
- `etl`
- `train`
- `infer`
- `report`
- `full`

---

## 4. Dominio industrial

Archivo de integración:

- `src/domains/industrial.py`

Unidad de decisión:

```text
plant_area + shift + date
```

Activos del dominio:

- generadores clásicos en `src/generators/`
- ETL clásico en `src/etl/`
- modelos en `src/models/`

Persistencia principal:

- raw: `data/raw/`
- processed: `data/processed/`
- db: `data/skillup.db`

Artefactos relevantes:

- `ml_features`
- `headcount_regressor.pkl`
- `deficit_classifier_xgboost.pkl`
- `deficit_classifier_calibrated.pkl`
- `model_artifacts.json`
- `staffing_inference_predictions.csv`
- `staffing_inference_metrics.json`

---

## 5. Dominio restaurant

Archivo de integración:

- `src/domains/restaurant.py`

Unidad de decisión:

```text
date + service_period
```

Franjas horarias activas:

- `11_13`
- `13_15`
- `19_21`
- `21_23`

Configuración del dominio:

- `config/restaurant_settings.py`

Características principales:

- calendario Chile
- demanda por franja
- déficit general
- déficit por rol crítico
- dashboard ejecutivo

Roles críticos actuales:

- `garzon`
- `cocinero_linea`
- `jefe_turno`

Persistencia principal:

- raw: `data/restaurant/raw/`
- processed: `data/restaurant/processed/`
- db: `data/restaurant/restaurant_skillup.db`

Artefactos relevantes:

- `restaurant_ml_features.csv`
- `restaurant_model_metrics.csv`
- `restaurant_model_artifacts.json`
- `restaurant_staffing_predictions.csv`
- `restaurant_staffing_metrics.json`
- `restaurant_executive_dashboard.png`

---

## 6. ETL validado

### Industrial

El ETL industrial mantiene la lógica original:

- limpieza de biometría
- imputación
- normalización individual
- rolling features
- lag features
- agregación área-turno-día

Salida principal:

```text
ml_features
```

### Restaurant

El ETL restaurant construye features a nivel de `date + service_period` a partir de:

- empleados
- calendario Chile
- demanda
- work records
- biometría
- ausentismo

Salida principal:

```text
restaurant_ml_features
```

---

## 7. Modelado validado

### Industrial

Modelos activos:

- regresor de headcount
- clasificador de déficit
- clasificador calibrado

### Restaurant

Modelos activos:

- regresor de dotación total
- clasificador general de déficit
- clasificador calibrado general
- clasificadores de déficit por rol crítico

En ambos dominios se guarda metadata del modelo con:

- feature set
- thresholds
- métricas
- clasificador seleccionado

---

## 8. Inferencia validada

### Industrial

Script principal de inferencia:

- `scripts/09_run_inference.py`

También disponible por runner:

```bash
uv run python scripts/run_pipeline.py --domain industrial --stage infer
```

### Restaurant

Disponible por runner:

```bash
uv run python scripts/run_pipeline.py --domain restaurant --stage infer
```

La inferencia restaurant genera:

- dotación predicha
- probabilidad de déficit general
- probabilidad de déficit por rol
- recomendación operativa textual

---

## 9. Flujo documental validado

El sistema incorpora una capacidad paralela de documentos PDF.

Componentes principales:

- `src/generators/medical_forms.py`
- `src/extraction/medical_forms.py`

Scripts:

- `05_generate_medical_forms.py`
- `06_extract_medical_forms.py`
- `07_forms_to_etl.py`
- `08_forms_to_model_metrics.py`

Capacidades validadas:

- generación de fichas médicas PDF
- extracción estructurada a CSV
- validación exacta contra `employees.csv`
- integración al ETL y al entrenamiento

---

## 10. UI validada

La UI actual está implementada con **Streamlit**.

Archivos:

- `src/ui/dashboard_app.py`
- `scripts/run_dashboard_ui.py`

Comando recomendado:

```bash
make up
```

Comando local:

```bash
uv run python scripts/run_dashboard_ui.py
```

Capacidades:

- ejecutar pipelines por dominio
- visualizar KPIs ejecutivos
- filtrar datos por dominio
- explorar gráficos interactivos
- consultar métricas y predicciones
- revisar artefactos y logs

---

## 11. Interpretación funcional de la dotación

- **Dotación actual**: personal que realmente estuvo disponible u operando.
- **Dotación requerida**: personal que el negocio necesita para cubrir la demanda.
- **Dotación predicha**: personal que el modelo estima que estará disponible.

Regla práctica:

```text
si dotación predicha < dotación requerida
=> riesgo de déficit esperado
```

---

## 12. Qué quedó obsoleto del documento anterior

El documento previo describía correctamente el flujo industrial original, pero quedó desactualizado en estos puntos:

- asumía un solo pipeline
- no contemplaba dominios seleccionables
- no contemplaba el dominio `restaurant`
- no contemplaba la UI interactiva
- no contemplaba el flujo de fichas médicas PDF

Por eso este archivo ahora debe leerse como referencia arquitectónica vigente, y `docs/pipeline_actualizado.md` como explicación operativa completa del sistema actual.
