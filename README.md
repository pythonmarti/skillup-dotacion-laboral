# SkillUp - Prediccion de Dotacion Laboral

Sistema de prediccion de ausentismo laboral basado en datos biometricos de wearables, aplicando tecnicas de machine learning para optimizar la planificacion de dotacion en plantas industriales.

## Referencia Cientifica

Basado en la metodologia del paper **PMC12604190**: uso de Random Forest y Gradient Boosting para prediccion de ausentismo, alcanzando AUC 0.89 y Accuracy 84% con datos biometricos y laborales.

## Requisitos

- Python 3.10+
- pip

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

Ejecutar los scripts en orden:

```bash
# 1. Generar datos sinteticos (empleados, biometricos, ausentismo)
python scripts/01_generate_data.py

# 2. Ejecutar ETL y feature engineering
python scripts/02_run_etl.py

# 3. Entrenar y evaluar modelos
python scripts/03_train_model.py

# 4. Pipeline completo (ejecuta los 3 pasos anteriores)
python scripts/04_full_pipeline.py
```

## Estructura del Proyecto

```
skillup/
├── config/
│   └── settings.py              # Constantes, rangos, paths
├── data/
│   ├── raw/                     # CSVs generados
│   ├── processed/               # Modelos y graficas
│   └── skillup.db               # Base de datos SQLite
├── scripts/
│   ├── 01_generate_data.py      # Generacion de datos sinteticos
│   ├── 02_run_etl.py            # ETL y feature engineering
│   ├── 03_train_model.py        # Entrenamiento de modelos
│   └── 04_full_pipeline.py      # Pipeline completo
├── src/
│   ├── generators/              # Generadores de datos sinteticos
│   │   ├── employees.py         # Perfiles demograficos
│   │   ├── work_records.py      # Registros de trabajo
│   │   ├── biometrics.py        # Datos biometricos de wearables
│   │   └── absenteeism.py       # Eventos de ausentismo
│   ├── etl/                     # Pipeline ETL
│   │   ├── extract.py           # Carga de CSVs
│   │   ├── transform.py         # Limpieza y transformacion
│   │   └── load.py              # Carga a SQLite
│   ├── models/                  # Modelos predictivos
│   │   ├── features.py          # Preparacion de features
│   │   ├── train.py             # Entrenamiento RF y GB
│   │   └── evaluate.py          # Evaluacion y graficas
│   └── utils/                   # Utilidades
│       ├── database.py          # Helpers SQLite
│       └── validators.py        # Validacion de rangos
├── tests/
│   └── test_generators.py       # Tests de generadores
├── requirements.txt
└── README.md
```

## Volumen de Datos Esperado

| Tabla | Filas Aproximadas |
|-------|-------------------|
| employees | 200 |
| work_records | ~36,000 (200 empleados x 180 dias - descansos) |
| biometrics | ~36,000 (1 registro diario por dia trabajado) |
| absenteeism | ~2,000-4,000 eventos |
| ml_features | ~36,000 (features consolidadas) |

## Tests

```bash
pytest tests/ -v
```
