# SkillUp: Documentación Técnica del Pipeline de Dotación Laboral

## 1. Visión General de la Arquitectura

SkillUp es una plataforma **multi-dominio** con orquestación centralizada y especialización por vertical de negocio. La arquitectura sigue un patrón de **runner común + stages estandarizados** donde cada dominio (industrial, restaurant, clinic) implementa cinco etapas: `generate`, `etl`, `train`, `infer`, `report`.

### 1.1 Componentes del núcleo

```
src/domains/
├── base.py        # DomainPipeline dataclass + run_stage() + run_full()
├── registry.py    # DOMAIN_REGISTRY dict + get_domain_pipeline()
├── cli.py         # argparse wrapper para --domain y --stage
├── industrial.py  # Pipeline dominio industrial (legacy generators + ETL)
├── restaurant.py  # Pipeline dominio restaurant
└── clinic.py      # Pipeline dominio clinic
```

**`DomainPipeline`** (`src/domains/base.py:13`): dataclass inmutable que expone un `StageRunner` por etapa. Cada `StageRunner` es un `Callable[[object], None]` que recibe un objeto de argumentos (típicamente un `argparse.Namespace`). `run_full()` itera secuencialmente `generate → etl → train → infer → report`, omitiendo stages no implementados.

**`DOMAIN_REGISTRY`** (`src/domains/registry.py:10`): diccionario `str → DomainPipeline` que resuelve el dominio solicitado por CLI. Agregar un nuevo dominio requiere únicamente registrar una nueva entrada; el runner y la UI no requieren modificación.

### 1.2 Decisión de diseño: Stages estandarizados como interfaz común

La uniformidad de stages (`generate`, `etl`, `train`, `infer`, `report`) es una decisión arquitectónica deliberada:

- **Ventaja**: permite que la UI (`src/ui/dashboard_app.py`) y los scripts de orquestación (`scripts/run_pipeline.py`) operen sobre cualquier dominio sin conocer sus detalles internos. La UI itera sobre `DOMAIN_REGISTRY`, expone los stages disponibles, y llama a `pipeline.run_stage(stage, args)`.
- **Trade-off**: obliga a que todo dominio exprese su lógica en estas cinco etapas. Si un dominio requiriera un flujo distinto (ej. streaming en tiempo real), requeriría extender la interfaz base.

---

## 2. Etapa 1: Generación de Datos Sintéticos (`generate`)

### 2.1 Filosofía de los generadores sintéticos

Todos los generadores son deterministas dada una semilla (`seed`). Usan distribuciones de probabilidad con parámetros fijos (no calibrados contra datos reales) para producir datasets con **correlaciones plausibles pero simplificadas**. Esto implica:

- Los datos **no provienen de una operación real**.
- Las relaciones entre variables se inyectan mediante fórmulas deterministas con ruido estocástico, no mediante estimación estadística sobre datos observados.
- El objetivo es **demostrar el pipeline end-to-end**, no afirmar validez externa.

La reproducibilidad está garantizada por tres mecanismos: (a) `np.random.default_rng(seed)` en cada generador, (b) `Faker.seed(seed)` para nombres sintéticos, (c) `pd.Timestamp` fijos como fechas de inicio. Con la misma seed y parámetros, la salida es idéntica.

### 2.2 Generador de empleados (`src/generators/employees.py`)

**Dominio: industrial.**

Genera `n` perfiles con las siguientes distribuciones:

| Variable | Distribución | Parámetros | Justificación |
|---|---|---|---|
| `age` | Normal truncada | μ=38, σ=8, [22,62] | Población laboral activa con sesgo etario medio |
| `gender` | Bernoulli | p(M)=0.75 | Sesgo de género esperable en industria pesada |
| `bmi` | Normal truncada | μ=27, σ=4, [18,45] | IMC poblacional con sobrepeso leve promedio |
| `education_level` | Categórica | [0.15, 0.35, 0.40, 0.10] | Distribución sesgada hacia educación media-superior |
| `plant_area` | Categórica ponderada | Pesos por área [0.25, 0.25, 0.15, 0.20, 0.15] | Mayor peso en producción pesada |
| `seniority_years` | Exponencial truncada | λ = (edad-20)*0.4, [0, edad-20] | Correlacionada positivamente con edad, sesgada a baja antigüedad |
| `distance_to_work_km` | Log-Normal truncada | μ=2.5, σ=0.8, [1, 120] | Distribución típica de commuting |
| `children` | Poisson | λ=1.2 | Tasa de fecundidad razonable para el rango etario |
| `smoker` | Bernoulli | p=0.20 | Prevalencia de tabaquismo en población chilena |
| `shift_pattern` | Categórica | fijo=0.3, rotativo=0.7; oficinas siempre fijo | Las áreas operativas requieren rotación; oficinas son diurnas |

**Justificación del diseño**: El catálogo de variables se eligió para cubrir tres dimensiones: demográfica (edad, género), fisiológica basal (BMI, tabaquismo, edad) y laboral-contextual (área, antigüedad, distancia, turno). Cada una tiene un rol downstream: las basales alimentan el generador biométrico; las laborales determinan asignación de turnos; las contextuales introducen heterogeneidad individual.

Para los dominios **restaurant** y **clinic**, la lógica es análoga pero con distribuciones adaptadas al perfil de cada sector (ej. restaurant: edad μ=30, σ=7, mayor proporción femenina; clinic: incluye asignación primaria/secundaria de unidad y flag `can_float`).

**Código**: `src/restaurant/generate.py:66` (`generate_restaurant_employees`), `src/clinic/generate.py` (sección de generación de empleados).

### 2.3 Generador de registros de trabajo (`src/generators/work_records.py`)

Para cada empleado y cada día del horizonte de simulación, determina:

1. **Asignación de turno**: según `shift_pattern` del empleado. Si es `fijo`, se asigna el turno correspondiente al área. Si es `rotativo`, rota cada 7 días entre diurno, vespertino y nocturno.
2. **Días de descanso**: ciclo semanal de 5 días de trabajo + 2 de descanso.
3. **Horas trabajadas**: `Uniform(7.5, 8.5)` si trabaja.
4. **Horas extra**: `Exponential(1.5)` con probabilidad base 0.15, modulada por `base_risk` del área (a mayor riesgo, mayor probabilidad de overtime).
5. **Días consecutivos**: contador acumulativo reiniciado en cada descanso.
6. **Carga operativa** (`workload_score`): `Normal(base_risk * 20 + 30, 10)`. Las áreas de mayor riesgo generan sistemáticamente mayor carga.

**Justificación de la carga por área**: como el dominio industrial no modela producción diaria variable, la heterogeneidad operacional entre áreas se captura mediante `base_risk` fijo. Esto es una simplificación que asume que ciertas áreas son estructuralmente más exigentes, independientemente de la demanda diaria. En una implementación con datos reales, este campo sería reemplazado por métricas de throughput o carga productiva medida.

### 2.4 Generador biométrico (`src/generators/biometrics.py`)

Este es el componente más elaborado del pipeline sintético. Genera series temporales diarias de señales fisiológicas por empleado, con tres mecanismos entrelazados:

#### 2.4.1 Baselines individuales (`_build_baselines`)

Cada empleado recibe valores base determinados por edad, BMI y tabaquismo:

```
hr_base    = 72 - (age-30)*0.15 + N(0,3) + (3 si smoker) + (4 si BMI>30)
hrv_base   = 45 - (age-30)*0.4 - max(0, BMI-25)*0.5 + N(0,5), floor=10
spo2_base  = 97.5 + N(0,0.5) - (1.0 si smoker)
sleep_base = 7.0 + N(0,0.5)
stress_base = 35 + N(0,5)
steps_base = 8000 + N(0,1500)
```

**Justificación fisiológica**: Los coeficientes están basados en literatura ocupacional básica. La edad reduce HRV y eleva HR levemente. El tabaquismo deprime SpO2 (~1%). El BMI alto eleva HR basal y reduce HRV. Estos no pretenden ser valores clínicos exactos sino generar heterogeneidad inter-individuo plausible.

#### 2.4.2 Eventos de salud predeterminados (`_generate_health_events`)

Para cada empleado se pre-genera una secuencia de estados de salud sobre el horizonte temporal completo:

- **Probabilidad diaria de enfermar**: 0.012 (≈4.4 eventos/año).
- **Duración**: `Poisson(avg_days - 1) + 1` según la categoría de enfermedad (musculoesquelético=5d, mental=7d, respiratorio=3d, etc.).
- **Fase de incubación**: 2 días previos a la enfermedad, con estados intermedios (1=incubación temprana, 2=incubación tardía, 3=enfermedad activa).
- **Período refractario**: 5 días post-enfermedad sin posibilidad de nuevo evento.

Esta estructura es una simplificación de un modelo SIR (Susceptible-Infectado-Recuperado) aplicado a nivel individual, donde la "infección" es estocástica y la recuperación es determinista.

**Justificación**: Sin una cadena de deterioro→incubación→enfermedad, el ausentismo sería completamente aleatorio y desconectado de las señales biométricas. La fase de incubación permite que el modelo observe señales degradadas **antes** de la ausencia, creando una relación predictiva temporalmente válida (sin fuga de información hacia el futuro).

#### 2.4.3 Generación de señales con autocorrelación temporal

Para cada día y empleado, las señales fisiológicas se generan con un modelo autoregresivo de orden 1:

```
target = baseline + ajuste_contextual + N(0, σ)
valor  = α * valor_previo + (1-α) * target
```

Donde `α = 0.7` es el coeficiente de autocorrelación. Esto produce series que no saltan bruscamente día a día, simulando la inercia fisiológica real.

**Ajustes contextuales** (sumados al target):

| Condición | Efecto en HR | Efecto en SpO₂ | Efecto en Temp | Efecto en Estrés |
|---|---|---|---|---|
| Turno nocturno | -5 bpm | -0.3% | — | +10 pts |
| Área con calor | +8 bpm | — | +0.4°C | — |
| Incubación (fase 1-2) | +4 a +8 | -0.2 a -0.5% | +0.1 a +0.3°C | +10 a +20 |
| Enfermedad (fase 3) | +10 a +18 | -0.6 a -1.5% | +0.3 a +0.6°C | +25 a +40 |

**Justificación**: El turno nocturno deprime HR (ritmo circadiano) y eleva estrés (desincronización). El calor eleva HR por vasodilatación y temperatura periférica. La enfermedad produce taquicardia, hipoxemia leve y fiebre. Estos efectos son direccionalmente correctos, aunque las magnitudes son arbitrarias (no calibradas).

### 2.5 Generador de ausentismo (`src/generators/absenteeism.py`)

La probabilidad de ausencia se modela con umbrales deterministas, no con una regresión logística:

1. **Ausencia médica**: si el biométrico del día indica `_is_sick == 1` (fase 3 de enfermedad), la ausencia es **determinista** (`is_absent = True`). La razón se toma de `_absence_reason`.
2. **Ausencia no médica residual**: probabilidad fija de 0.003 por empleado-día (~0.3%), para evitar un sistema completamente determinista.
3. **Horas perdidas**: `Uniform(7.5, 8.5)` para toda ausencia.

**Justificación**: El acoplamiento determinista entre enfermedad y ausencia es una simplificación fuerte. En la realidad existen factores moderadores (presentismo, licencias parciales, trabajo remoto). Sin embargo, para un sistema sintético que busca demostrar la cadena causal `señales → enfermedad → ausencia → déficit`, esta simplificación produce datasets con señal clara y entrenable. En producción, este acoplamiento se reemplazaría por un modelo probabilístico calibrado con datos reales de ausentismo.

### 2.6 Generadores de demanda (restaurant y clinic)

**Restaurant** (`src/restaurant/generate.py`): la demanda se modela por franja horaria (`11_13`, `13_15`, `19_21`, `21_23`) con:

```
forecast_covers = base_covers[franja] * multiplicadores_contexto + ruido
```

Multiplicadores: día de semana, fin de semana, feriado, víspera de feriado, ventana de pago, vacaciones escolares, presión turística estacional, promociones activas, eventos locales y clima. El clima actúa como shock que altera el mix sala vs delivery: a peor clima, más delivery y menos sala.

**Clinic** (`src/clinic/generate.py`): baselines diferenciados por unidad y turno con multiplicadores análogos (campaña de vacunación, alerta respiratoria, presión de bloque electivo, ventana de pago). La unidad de imagenología incluye `backlog` que se acumula cuando la capacidad es insuficiente.

**Justificación de la granularidad**: ambos dominios modelan demanda agregada por bloque horario, no minuto a minuto ni paciente a paciente. Esta decisión responde a que la variable objetivo (dotación) se decide a nivel de bloque operativo, no a nivel de evento individual. Modelar granularidad más fina no aportaría precisión incremental para este caso de uso y multiplicaría la complejidad de la simulación.

---

## 3. Etapa 2: ETL (`etl`)

### 3.1 ETL Industrial: Pipeline de Transformación en 7 Pasos

El ETL industrial (`src/etl/transform.py`) opera sobre los CSVs raw y produce la tabla `ml_features` materializada en SQLite.

#### Paso 1: Validación de rangos fisiológicos (`validate_physiological_ranges`)

Todo valor fuera de `PHYSIO_RANGES` (definido en `config/settings.py:29`) se reemplaza por `NaN`. Los rangos son deliberadamente amplios para capturar variabilidad patológica sin descartar señales extremas legítimas:

```python
"hr_mean_bpm": (30, 220),       # Bradicardia severa a taquicardia máxima
"spo2_mean_pct": (70, 100),     # Hipoxemia severa a saturación normal
"skin_temp_mean_c": (28, 42),   # Hipotermia a fiebre alta
"stress_score": (0, 100),        # Escala normalizada
```

#### Paso 2: Remoción de artefactos (`remove_artifacts`)

- **Calidad baja**: registros con `data_quality_score < 0.3` → todas las columnas biométricas a `NaN`. Simula un sensor defectuoso o mal colocado.
- **HR constante**: si `hr_std_bpm < 0.5` y `hours_worked > 4`, las columnas de frecuencia cardíaca se anulan. Simula un sensor desprendido que reporta un valor fijo.

**Justificación**: en datos reales de wearables, entre 5-20% de registros presentan artefactos por movimiento, desprendimiento o batería baja. Eliminarlos antes de la imputación evita que valores espurios contaminen medias móviles y normalizaciones.

#### Paso 3: Imputación sin fuga temporal (`impute_missing_values`)

Estrategia híbrida que respeta la dirección del tiempo:

- **Gaps ≤ 2 días**: forward-fill (último valor conocido hacia adelante).
- **Gaps > 2 días**: expanding mean (media de todos los valores previos del mismo empleado).
- **NaN residuales**: relleno con 0.

**Por qué no KNN, MICE o interpolación**: estas técnicas usarían información de días posteriores al gap (fuga temporal) o de otros empleados (mezcla de contextos individuales). La imputación con expanding mean usa exclusivamente datos pasados del mismo individuo, garantizando que el modelo entrene sin acceso a información del futuro.

#### Paso 4: Normalización intra-individuo (`normalize_by_individual`)

Z-score por empleado usando **ventana expansiva** (no ventana centrada):

```
zscore_t = (x_t - mean_{1..t}) / std_{1..t}
```

Con `min_periods=2`. Esto produce valores normalizados sin fuga temporal: el z-score en el día `t` solo usa datos hasta `t`.

**Justificación**: las señales fisiológicas absolutas tienen alta varianza inter-individuo. Un HR de 80 bpm puede ser basal para una persona y elevado para otra. La normalización intra-sujeto convierte valores absolutos en desviaciones respecto al propio historial, que es una señal más informativa para detectar deterioro.

#### Paso 5: Features de ventanas móviles y lags

**Rolling features** (`create_rolling_features`):
```
hr_mean_bpm_7d_mean   = rolling(7).mean()
hr_mean_bpm_7d_std    = rolling(7).std()
hr_mean_bpm_trend     = mean_7d_actual - mean_7d_anterior (shift 7)
stress_score_3d_mean  = rolling(3).mean()
fatigue_14d           = rolling(14).sum() de stress_score
```

Ventanas: 3, 7, 14 y 30 días. La tendencia captura si una métrica está mejorando o empeorando respecto a la semana anterior.

**Lag features** (`create_lag_features`):
```
hr_mean_bpm_lag1, lag3, lag5, lag7
hrv_rmssd_ms_lag{1,3,5,7}
stress_score_lag{1,3,5,7}
sleep_duration_hours_lag{1,3,5,7}
steps_lag{1,3,5,7}
```

**Desviación del baseline** (`create_baseline_deviation_features`):
```
hr_mean_bpm_baseline_dev = hr_mean_bpm - expanding_mean(hr_mean_bpm)
```

Similar al z-score pero preservando la unidad original.

#### Paso 6: Merge y features temporales

**Merge** (`merge_all_sources`): left join secuencial de biometrics ← work_records ← absenteeism ← employees sobre `(employee_id, date)`.

**Features temporales** (`create_temporal_features`):
```
day_of_week (0=Lunes..6=Domingo)
is_monday, is_friday, is_weekend_adjacent
week_of_year
overtime_7d_sum, overtime_14d_sum
```

#### Paso 7: Agregación a nivel área-turno-día y creación de targets

**Agregación** (`aggregate_to_area_shift_date`): las features a nivel empleado-día se agregan a nivel `plant_area + shift + date` mediante:

```
avg_hr_mean_bpm = mean(hr_mean_bpm_zscore) agrupado por área, turno, día
absentee_rate   = count(is_absent) / count(empleados programados)
```

Se incorporan también lags de métricas agregadas (`area_shift_absentee_rate_lag1..7`, `actual_headcount_lag1..7`).

**Targets**:
```
actual_headcount = personas efectivamente presentes
required_headcount = REQUIRED_STAFF[area][turno]  (fijo por configuración)
has_deficit = 1 si actual_headcount < required_headcount else 0
deficit_count = max(0, required_headcount - actual_headcount)
```

### 3.2 ETL Restaurant: Diferencias Clave

El ETL de restaurant (`src/restaurant/etl.py`) difiere del industrial en aspectos fundamentales:

1. **Unidad de agregación**: `date + service_period` (no `plant_area + shift + date`).
2. **Dotación requerida dinámica**: calculada por `estimate_required_staff()` en función de `covers`, `delivery_orders`, `service_period`, `is_holiday`, `is_weekend`, `local_event_flag`. Las reglas son:
   ```
   garzon         = ceil(effective_demand / 22)
   cocinero_linea  = ceil((covers + delivery_orders) / 30)
   ayudante_cocina = ceil((covers + delivery_orders) / 45)
   ```
3. **Dos dotaciones**: `scheduled` (programada) y `actual` (descontando ausentes).
4. **Features biométricas agregadas**: promedios por franja de `hr_mean_bpm`, `hrv_rmssd_ms`, `sleep_duration_hours`, `sleep_efficiency_pct`, `stress_score`, `fatigue_proxy`, `age`, `bmi`, `distance_to_work_km`.
5. **Features de ausentismo**: `absentee_rate`, `short_notice_absentee_rate`, `absent_count_total`.
6. **Lags intra-franja**: `feature_lag1` = valor de la misma franja el período anterior; `feature_lag7` = misma franja hace 7 días. Se agrupa por `service_period` antes de shift.
7. **Targets por rol crítico**: además del déficit total, se calculan `deficit_role_{garzon|cocinero_linea|jefe_turno}` y `has_deficit_role_{rol}`.

### 3.3 Persistencia

Tanto el ETL industrial como restaurant/clinic persisten en **SQLite con WAL mode** (`src/utils/database.py:14`). La elección de SQLite sobre PostgreSQL/MySQL se justifica por:

- **Cero configuración**: no requiere servidor externo, compatible con Docker y desarrollo local.
- **WAL mode**: permite lecturas concurrentes sin bloquear escrituras.
- **Reproducibilidad**: el archivo `.db` es portable y versionable.
- **Debugging**: queries SQL directas para inspeccionar estados intermedios del pipeline.

---

## 4. Etapa 3: Entrenamiento de Modelos (`train`)

### 4.1 Estrategia de Split Temporal

Todo split de datos es **temporal, no aleatorio** (`src/models/features.py:8`):

```python
def temporal_train_test_split(df, test_ratio=0.2):
    dates = sorted(unique(df[date_col]))
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]
    train = df[df[date_col] < split_date]
    test  = df[df[date_col] >= split_date]
```

En el pipeline de entrenamiento (`src/restaurant/train.py`, `src/models/train.py`) se aplica un split en tres particiones:

```
|------ train (64%) ------|-- val (16%) --|-- test (20%) --|
```

**Justificación**: un split aleatorio permitiría que el modelo "vea el futuro" al tener datos de diciembre en entrenamiento y datos de octubre en test. En series temporales, el modelo debe predecir hacia adelante, por lo que todas las fechas de test deben ser posteriores a todas las fechas de entrenamiento. Esto es una restricción más dura que el típico `train_test_split` aleatorio, y produce estimaciones de rendimiento más realistas (típicamente más bajas) para el despliegue en producción.

### 4.2 Modelos Entrenados

#### Modelo 1: Regresor de Dotación

**Algoritmo**: `RandomForestRegressor` con `n_estimators=100` (`src/models/staffing_models.py:26`).

Predice `actual_headcount` (o `actual_headcount_total` en restaurant). Es un problema de regresión estándar con target continuo. Random Forest se elige sobre boosted trees por su menor tendencia al overfitting en datasets sintéticos de dimensionalidad moderada.

#### Modelo 2: Clasificador de Déficit (XGBoost)

**Algoritmo**: `XGBClassifier` con hiperparámetros base:

```python
XGBClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=4,
    eval_metric="logloss", random_state=42
)
```

Predice `has_deficit` (binario). La profundidad máxima baja (`max_depth=4`) actúa como regularización implícita para árboles poco profundos que capturen patrones generalizables, no ruido.

#### Modelo 3: Clasificador Calibrado

**Algoritmo**: `CalibratedClassifierCV(estimator=XGBClassifier(...), method="isotonic", cv=3)`.

La calibración isotónica (pool-adjacent-violators) ajusta las probabilidades predichas para que reflejen frecuencias empíricas reales. Un modelo bien calibrado que predice "70% de probabilidad de déficit" debería observar déficit ~70% de las veces que emite esa predicción.

**Por qué se necesita calibración**: los modelos basados en árboles (Random Forest, XGBoost) tienden a producir probabilidades extremas (muy cercanas a 0 o 1) que no están bien calibradas. En un contexto de negocio donde la probabilidad se usa para tomar decisiones con umbrales, la calibración es crítica.

#### Modelos por Rol Crítico

En restaurant y clinic, se entrena un clasificador XGBoost independiente por cada rol crítico (garzón, cocinero_linea, jefe_turno; médico, enfermera, TENS). Cada uno predice `has_deficit_role_{rol}`. La matriz de features es la misma que para el modelo general, pero el target es específico del rol.

**Justificación**: un solo modelo multi-etiqueta sería más eficiente pero menos interpretable. Modelos independientes por rol permiten análisis SHAP específico por rol y recomendaciones diferenciadas ("refuerza garzón, no cocina").

### 4.3 Manejo de Clases Desbalanceadas

El déficit de personal es inherentemente un evento de clase minoritaria (típicamente 10-30% de las observaciones). Se aplican tres estrategias complementarias:

1. **`scale_pos_weight` en XGBoost** (`src/models/train.py:123`): pondera la clase positiva proporcionalmente al desbalance (`neg_count / pos_count`). Esto modifica la función de pérdida para penalizar más los errores en la clase minoritaria.

2. **`class_weight='balanced'` en Random Forest**: análogo para sklearn.

3. **SMOTE-ENN** (`src/models/balancing.py:8`): genera muestras sintéticas de la clase minoritaria (SMOTE) y luego elimina instancias ruidosas de ambas clases (Edited Nearest Neighbors). Se aplica dentro de un `ImbPipeline` para evitar fuga de datos durante la validación cruzada.

**Por qué no solo sobremuestreo**: SMOTE solo puede generar ruido si la clase minoritaria tiene outliers. ENN limpia esos casos, produciendo una frontera de decisión más suave. El pipeline con `ImbPipeline` garantiza que el resampling ocurra dentro de cada fold de CV, no sobre el dataset completo.

### 4.4 Búsqueda de Hiperparámetros

Cada modelo se entrena con `RandomizedSearchCV` (`src/models/train.py`):

- **Espacio de búsqueda**: grids amplios (ej. XGBoost: `max_depth=[3,5,7,9]`, `learning_rate=[0.01,0.05,0.1,0.2]`, `subsample=[0.7,0.8,0.9]`, `colsample_bytree=[0.6,0.7,0.8,0.9]`, `gamma=[0,0.1,0.2,0.5]`, `reg_alpha=[0,0.01,0.1,1.0]`, `reg_lambda=[0.5,1.0,2.0,5.0]`).
- **Número de iteraciones**: `MODEL_CONFIG["n_iter_search"] = 10`.
- **Scoring**: `f1` como métrica objetivo, pues balancea precisión y recall en contexto desbalanceado.
- **CV estratificado temporal**: `TimeSeriesSplit(n_splits=3)` cuando `temporal=True`.

**Trade-off `n_iter=10`**: una búsqueda exhaustiva (GridSearchCV) con estos espacios requeriría cientos de miles de ajustes. RandomizedSearchCV con 10 iteraciones explora una fracción del espacio pero es computacionalmente viable (minutos en CPU). En producción con datos reales, se aumentaría `n_iter` o se usaría optimización bayesiana (Optuna/Hyperopt).

### 4.5 Selección del Mejor Clasificador

XGBoost (Modelo 2) y Calibrado (Modelo 3) compiten. El ganador se elige usando **solo el conjunto de validación** (el 16% intermedio). Métricas de decisión: F1-score, AUC-ROC y Brier Score (calibración). El ganador se reentrena en `train + val` combinados y se evalúa una única vez sobre el test ciego (20% final).

**Justificación**: usar el test set para seleccionar modelos introduciría sobreajuste por fuga de información. El test set solo se toca una vez, al final, para reportar la métrica final no sesgada.

### 4.6 Stacking Ensemble (Industrial)

Adicionalmente, el pipeline industrial entrena un **Stacking Classifier** (`src/models/train.py:205`):

```
Base learners: RandomForest + XGBoost + LightGBM
Meta-learner:  LogisticRegression(class_weight='balanced')
```

Cada base learner genera predicciones de probabilidad; el meta-learner aprende a combinarlas óptimamente. El stacking usa `StratifiedKFold` para generar predicciones out-of-fold que alimentan al meta-learner, evitando overfitting.

---

## 5. Etapa 4: Inferencia (`infer`)

### 5.1 Generación de Predicciones

La inferencia carga los artefactos entrenados desde `data/processed/` y genera predicciones sobre todas las observaciones disponibles:

```python
# src/models/inference.py (industrial)
regressor      = joblib.load("headcount_regressor.pkl")
classifier     = joblib.load("deficit_classifier_xgboost.pkl")
calibrated     = joblib.load("deficit_classifier_calibrated.pkl")

df["predicted_headcount"]         = regressor.predict(X)
df["predicted_deficit_prob_xgb"]  = classifier.predict_proba(X)[:, 1]
df["predicted_deficit_prob_cal"]  = calibrated.predict_proba(X)[:, 1]
df["predicted_has_deficit"]       = (prob > threshold).astype(int)
```

Las métricas se calculan **solo sobre el período test** (post `split_date`), usando `y_true` del test que el modelo nunca vio durante entrenamiento.

### 5.2 Umbral Óptimo de Decisión

El umbral de clasificación no es 0.5 por defecto, sino que se optimiza para maximizar F1-score sobre los datos de validación (`find_optimal_threshold` en `src/models/evaluate.py:30`):

```python
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + ε)
best_threshold = thresholds[argmax(f1_scores)]
```

**Justificación**: en datos desbalanceados, un umbral fijo de 0.5 subestima la clase minoritaria. El umbral optimizado por F1 produce un balance operacionalmente útil: suficientes alertas para ser accionables, pero no tantas falsas alarmas que generen fatiga de alertas.

### 5.3 Recomendación Operativa Textual

La inferencia de restaurant y clinic genera una columna `recommendation` con texto en lenguaje natural. Ejemplo de lógica (`src/restaurant/inference.py`):

```
Si prob_deficit > 0.7 AND role_deficit_garzon > 0.7:
    → "Reforzar garzón: riesgo {prob}%. Considere activar backup de runner."
Si prob_deficit > 0.5 AND role_deficit_cocinero_linea > 0.6:
    → "Cocina tensionada. Fatiga acumulada + delivery alto. Evaluar cocinero de retén."
```

Esto cierra la brecha entre predicción numérica y acción operativa, eliminando la necesidad de que el usuario interprete probabilidades crudas.

---

## 6. Interpretabilidad con SHAP (`src/models/shap_analysis.py`)

### 6.1 Fundamentos de SHAP

SHAP (SHapley Additive exPlanations) asigna a cada feature un valor que representa su contribución marginal a la predicción, promediada sobre todas las posibles combinaciones de features. Proviene de los **valores de Shapley** en teoría de juegos cooperativos, donde cada feature es un "jugador" y la predicción es el "pago".

Formalmente, para una observación `x`:

```
f(x) = φ₀ + Σᵢ φᵢ(x)
```

Donde `φ₀` es el valor base (predicción promedio) y `φᵢ(x)` es la contribución de la feature `i`. Los valores SHAP satisfacen tres propiedades deseables:

1. **Aditividad**: las contribuciones suman la predicción total.
2. **Consistencia**: si una feature se vuelve más importante, su valor SHAP no disminuye.
3. **Ausencia**: una feature que no afecta la predicción recibe valor 0.

### 6.2 Artefactos SHAP Generados

El módulo `shap_analysis.py` produce cuatro tipos de artefactos:

1. **SHAP values globales**: magnitud promedio del impacto de cada feature (bar plot).
2. **SHAP values por segmento**: heatmap de impacto de cada feature en cada turno/franja/unidad.
3. **SHAP values por instancia**: explicación individual para cada observación en el dataset.
4. **SHAP dependence plots**: scatter del valor de una feature vs su impacto SHAP.

### 6.3 Traducción de Features Técnicas a Etiquetas de Negocio

El archivo `shap_analysis.py:13` contiene un diccionario `DOMAIN_FEATURE_LABELS` que mapea nombres de columnas técnicas a etiquetas interpretables:

```python
"forecast_covers"          → "Cubiertos proyectados"
"avg_sleep_duration_hours" → "Horas de sueño promedio"
"absentee_rate_lag7"       → "Ausentismo de la semana anterior"
"avg_stress_score"         → "Estres promedio del equipo"
"avg_bmi"                  → "Carga fisica promedio del equipo"
```

Esta capa de traducción permite que la UI (`src/ui/dashboard_app.py`) muestre gráficos con etiquetas que un operador no técnico puede interpretar.

### 6.4 Limitación Fundamental

SHAP explica **cómo decide el modelo**, no **qué causa el fenómeno en el mundo real**. Si SHAP muestra que `avg_fatigue_proxy` es la feature más importante, significa que el modelo aprendió a usar esa variable como señal predictiva fuerte. No demuestra que reducir la fatiga eliminaría el déficit con certeza, porque `avg_fatigue_proxy` podría ser un proxy de otras variables no modeladas (mala planificación, sobrecarga histórica, estacionalidad).

---

## 7. Decisiones de Diseño y sus Justificaciones

### 7.1 ¿Por qué SQLite como capa intermedia y no CSV directo al modelo?

**Problema**: los CSV raw son archivos independientes con esquemas heterogéneos (empleados, work_records, biometrics, absenteeism, demanda). Unirlos en cada entrenamiento o inferencia requeriría repetir joins, limpieza e imputaciones.

**Solución**: ETL produce tablas materializadas en SQLite que son consumidas directamente por entrenamiento e inferencia. Esto garantiza:

- **Reproducibilidad**: todos los consumidores leen la misma versión del dato.
- **Eficiencia**: las transformaciones pesadas (imputación, normalización, rolling, agregación) se ejecutan una sola vez.
- **Debuggability**: queries `SELECT * FROM ml_features WHERE date > ...` permiten inspeccionar el estado exacto de los datos que ve el modelo.

**Trade-off**: SQLite no escala a datasets masivos (>100 GB). Para datos reales de una empresa grande, se migraría a PostgreSQL o DuckDB manteniendo la misma interfaz `query_to_dataframe()`.

### 7.2 ¿Por qué dotación requerida fija en industrial?

La dotación requerida industrial se toma de `REQUIRED_STAFF` (`config/settings.py:62`), un diccionario estático `{área: {turno: mínimo}}`. No depende de producción diaria, demanda ni biometría.

**Justificación**: en operaciones industriales de proceso continuo, el mínimo operativo está determinado por seguridad y continuidad, no por demanda comercial. Una torre de destilación requiere cierto número de operadores independientemente del throughput. Este supuesto es razonable para refinerías, plantas químicas y procesos continuos. Sería incorrecto para manufactura discreta con demanda variable.

### 7.3 ¿Por qué expandir en lugar de rolling centrado para normalización?

La normalización usa `expanding().mean()` (ventana creciente desde el inicio) en lugar de `rolling(window).mean()` (ventana fija centrada o desplazada).

**Justificación**: `expanding` garantiza que el valor normalizado en el día `t` solo depende de datos hasta `t`, nunca de `t+1` en adelante. Con `rolling(window=7)`, el valor en `t` usaría datos de `t-3` a `t+3` (si es centrado) o de `t-6` a `t` (si es derecho). El primero es fuga temporal; el segundo es válido pero menos estable. `expanding` es la opción más conservadora y estrictamente libre de fuga hacia adelante.

### 7.4 ¿Por qué las franjas de restaurant son 11-13, 13-15, 19-21, 21-23?

La selección de franjas responde a la estructura de un restaurante casual dining chileno:

- **11-13**: preparación y primeras mesas.
- **13-15**: peak almuerzo.
- **19-21**: peak cena.
- **21-23**: cierre y segunda ola.

La franja 15:00-19:00 no es una unidad analítica propia. **No significa** que el local esté cerrado, sino que la señal operativa de ese bloque no justificó un modelo independiente para esta iteración. En una implementación real, se agregaría si el negocio reporta déficit recurrente en ese horario.

### 7.5 ¿Por qué roles específicos como críticos?

**Restaurant**: garzón (cuello de botella de atención al cliente), cocinero de línea (cuello de botella de producción), jefe de turno (cuello de botella de coordinación). Si falla cualquiera de los tres, la operación se degrada en cascada.

**Clínica**: médico (decisión clínica), enfermera (continuidad asistencial), TENS (ejecución de procedimientos). Son los tres perfiles sin los cuales la atención ambulatoria se detiene.

Los backups son discretos y acotados: un runner puede apoyar a garzón pero no reemplazarlo completamente; un TENS puede flotar a otra unidad pero no sustituir un médico. Esto refleja la realidad de las matrices de habilidades en estas industrias.

### 7.6 ¿Por qué modelos separados por dominio en lugar de un modelo unificado?

Un modelo único que predijera dotación para "cualquier negocio" requeriría que las features de una planta industrial, un restaurante y una clínica compartieran el mismo espacio de representación. Esto es conceptualmente forzado: la demanda en restaurant (cubiertos, delivery) no tiene análogo en industrial (requerimiento fijo por área). Las señales fisiológicas relevantes difieren (calor industrial vs fatiga de servicio vs carga cognitiva clínica). La unidad de decisión es distinta.

La arquitectura multi-dominio permite que cada vertical modele sus propias reglas de negocio, features y targets, compartiendo únicamente la infraestructura de ejecución, persistencia y visualización.

---

## 8. Glosario Técnico de Features por Dominio

### 8.1 Industrial

| Feature técnica | Tipo | Descripción |
|---|---|---|
| `actual_headcount` | int | Personas presentes en área-turno-día |
| `required_headcount` | int | Mínimo operativo del área-turno (fijo) |
| `scheduled_headcount` | int | Personas programadas antes de ausencias |
| `absentee_rate` | float | Proporción de ausentes sobre programados |
| `has_deficit` | bool | Target binario de clasificación |
| `deficit_count` | int | Magnitud del déficit |
| `avg_hr_mean_bpm_zscore` | float | Z-score de HR promedio del equipo |
| `avg_stress_score_zscore` | float | Z-score de estrés promedio |
| `avg_sleep_duration_hours_zscore` | float | Z-score de sueño promedio |
| `hr_mean_bpm_trend` | float | Tendencia 7d de HR individual promedio |
| `fatigue_14d` | float | Estrés acumulado 14d |
| `actual_headcount_lag{1..7}` | float | Dotación real en días anteriores |
| `area_shift_absentee_rate_lag{1..7}` | float | Ausentismo histórico del mismo segmento |
| `plant_area_*`, `shift_*` | one-hot | Codificación de área y turno |
| `day_of_week` | int | 0=Lunes..6=Domingo |
| `is_monday`, `is_friday` | bool | Indicadores de borde de semana |

### 8.2 Restaurant

| Feature técnica | Tipo | Descripción |
|---|---|---|
| `forecast_covers` | float | Cubiertos proyectados para la franja |
| `actual_covers` | float | Cubiertos realmente atendidos |
| `reservation_count` | int | Reservas registradas |
| `delivery_order_volume` | int | Pedidos delivery |
| `walk_in_ratio` | float | Proporción estimada de clientes sin reserva |
| `forecast_sales` | float | Ventas proyectadas |
| `actual_sales` | float | Ventas reales |
| `weather_impact_score` | float | Impacto climático sobre la demanda [-0.25, 0.25] |
| `local_event_flag` | bool | Evento local que empuja afluencia |
| `tourism_pressure_score` | float | Presión turística estacional [0, 1] |
| `actual_headcount_total` | int | Dotación efectiva post-ausencias |
| `forecast_required_headcount_total` | int | Dotación requerida proyectada (con forecast covers) |
| `required_headcount_total` | int | Dotación requerida real (con actual covers) |
| `scheduled_headcount_total` | int | Dotación programada pre-ausencias |
| `absentee_rate` | float | Tasa de ausentismo en la franja |
| `short_notice_absentee_rate` | float | Tasa de ausentismo de corto aviso |
| `avg_stress_score` | float | Estrés promedio del equipo programado |
| `avg_fatigue_proxy` | float | Fatiga promedio del equipo (compuesto de sueño+estrés+días) |
| `avg_sleep_duration_hours` | float | Sueño promedio del equipo |
| `avg_sleep_efficiency_pct` | float | Eficiencia de sueño promedio |
| `avg_steps` | float | Pasos promedio (carga física) |
| `scheduled_avg_consecutive_work_days` | float | Días consecutivos promedio del equipo |
| `scheduled_overtime_hours` | float | Horas extra totales en la franja |
| `avg_age`, `avg_bmi` | float | Demográficos promedio del equipo |
| `has_deficit_total` | bool | Target binario general |
| `deficit_count_total` | int | Magnitud del déficit general |
| `has_deficit_role_{rol}` | bool | Target binario por rol crítico |
| `deficit_role_{rol}` | int | Magnitud del déficit por rol |
| `service_period_*`, `season_*` | one-hot | Codificación de franja y temporada |
| `*_lag1`, `*_lag7` | float | Lags intra-franja a 1 y 7 períodos |

### 8.3 Clinic

| Feature técnica | Tipo | Descripción |
|---|---|---|
| `patient_volume` | int | Pacientes esperados/atendidos en unidad-turno |
| `high_acuity_cases` | int | Casos de alta complejidad |
| `scheduled_procedures` | int | Procedimientos ambulatorios programados |
| `active_care_stations` | int | Boxes o estaciones activas |
| `imaging_backlog_cases` | int | Rezago acumulado en imagenología |
| `respiratory_alert_cases` | int | Casos de alerta respiratoria |
| `avg_wait_time_min` | float | Tiempo promedio de espera |
| `no_show_rate` | float | Tasa de inasistencia de pacientes |
| `cognitive_load_score` | float | Carga cognitiva promedio del equipo |
| `reaction_time_ms` | float | Proxy de enlentecimiento funcional |
| `avg_stress_score` | float | Estrés fisiológico promedio |
| `avg_sleep_duration_hours` | float | Sueño promedio del equipo |
| `absentee_rate` | float | Tasa de ausentismo |
| `can_float_count` | int | Personal con capacidad de flotar entre unidades |
| `cross_trained_available` | int | Personal con cross-training disponible |

---

## 9. Métricas de Evaluación

### 9.1 Métricas de Regresión

| Métrica | Fórmula | Interpretación |
|---|---|---|
| MAE | mean(\|y_true - y_pred\|) | Error absoluto promedio en número de personas |
| RMSE | sqrt(mean((y_true - y_pred)²)) | Penaliza errores grandes más que MAE |
| R² | 1 - SS_res / SS_tot | Proporción de varianza explicada |

### 9.2 Métricas de Clasificación

| Métrica | Fórmula | Relevancia para el negocio |
|---|---|---|
| AUC-ROC | Área bajo curva ROC | Capacidad discriminativa general |
| Average Precision | Área bajo curva PR | Mejor que AUC para clases desbalanceadas |
| Brier Score | mean((prob - y_true)²) | Calibración de probabilidades |
| F1-Score | 2 * P * R / (P + R) | Balance precisión-recall; usado para threshold óptimo |
| Recall | TP / (TP + FN) | ¿Cuántos déficit reales capturamos? |
| Precision | TP / (TP + FP) | ¿Cuántas alertas son déficit reales? |

**Priorización**: en contexto de dotación laboral, el **recall** (capturar déficit reales) suele priorizarse sobre precision (evitar falsas alarmas). El costo de un déficit no anticipado (operación degradada, seguridad comprometida, clientes perdidos) es típicamente mayor que el costo de una falsa alarma (refuerzo innecesario).

---

## 10. Pipeline de Fichas Médicas PDF

### 10.1 Generación (`src/generators/medical_forms.py`)

Genera PDFs con formato fijo a partir de `employees.csv`. Cada PDF contiene campos estructurados (nombre, RUT simulado, edad, BMI, presión arterial, antecedentes, aptitud laboral). El layout es fijo: los campos tienen posiciones conocidas y el formato es consistente.

### 10.2 Extracción (`src/extraction/medical_forms.py`)

Flujo:
1. `pdftotext` (poppler-utils) para extraer texto capa por capa.
2. Regex por campo para parsear valores del texto extraído.
3. Failsafe a OCR con `tesseract` si el PDF no tiene capa de texto.
4. Validación exacta contra `employees.csv` de referencia: el CSV extraído debe ser idéntico fila por fila, columna por columna al original.

**Limitación**: el parser asume PDFs generados por el propio sistema. PDFs del mundo real con layouts variables, fuentes no estándar o baja resolución requieren un enfoque distinto (posiblemente LLMs con visión o OCR + NER).

---

## 11. UI Interactiva (`src/ui/dashboard_app.py`)

Implementada con **Streamlit**. Capacidades:

- Selector de dominio (`industrial | restaurant | clinic`).
- Ejecución de pipelines por stage con visualización de logs en tiempo real.
- Panel de KPIs ejecutivos: dotación requerida vs predicha vs real, tasa de déficit, riesgo promedio.
- Gráficos SHAP interactivos con filtros por fecha, turno y segmento.
- Tabla de instancias críticas con explicaciones individuales.
- Drill-down por rol crítico (restaurant: garzón, cocinero, jefe de turno; clinic: médico, enfermera, TENS).

---

## 12. Referencia de Archivos Clave

| Archivo | Responsabilidad |
|---|---|
| `src/domains/base.py` | `DomainPipeline` dataclass y lógica de stages |
| `src/domains/registry.py` | Registro y resolución de dominios |
| `src/domains/restaurant.py` | Implementación de stages para restaurant |
| `src/domains/industrial.py` | Implementación de stages para industrial |
| `src/domains/clinic.py` | Implementación de stages para clinic |
| `src/generators/employees.py` | Generación de perfiles demográficos (industrial) |
| `src/generators/work_records.py` | Generación de registros de trabajo (industrial) |
| `src/generators/biometrics.py` | Series temporales biométricas con eventos de salud |
| `src/generators/absenteeism.py` | Ausentismo causalmente ligado a biometría |
| `src/restaurant/generate.py` | Generación de datos sintéticos restaurant (529 líneas) |
| `src/clinic/generate.py` | Generación de datos sintéticos clinic (738 líneas) |
| `src/etl/extract.py` | Lectura y validación de CSVs raw |
| `src/etl/transform.py` | Pipeline ETL industrial en 7 pasos (438 líneas) |
| `src/etl/load.py` | Persistencia en SQLite |
| `src/etl/pipeline.py` | Orquestador ETL industrial |
| `src/restaurant/etl.py` | ETL restaurant con agregación por franja (223 líneas) |
| `src/clinic/etl.py` | ETL clinic con agregación por unidad-turno |
| `src/models/features.py` | Split temporal y preparación de matriz de features |
| `src/models/staffing_models.py` | Regresor (RF) y clasificadores (XGBoost + Calibrado) |
| `src/models/train.py` | Entrenamiento con RandomizedSearchCV + Stacking |
| `src/models/balancing.py` | SMOTE-ENN y BorderlineSMOTE |
| `src/models/evaluate.py` | Métricas, curvas ROC/PR, threshold óptimo |
| `src/models/inference.py` | Inferencia industrial con métricas post split_date |
| `src/models/shap_analysis.py` | Artefactos SHAP globales y por rol (442 líneas) |
| `src/restaurant/inference.py` | Inferencia restaurant con recomendaciones textuales |
| `src/restaurant/train.py` | Entrenamiento restaurant con modelos por rol |
| `src/clinic/inference.py` | Inferencia clinic con recomendaciones textuales |
| `src/clinic/train.py` | Entrenamiento clinic con modelos por rol |
| `src/restaurant/reporting.py` | Dashboard ejecutivo restaurant |
| `src/clinic/reporting.py` | Dashboard ejecutivo clinic |
| `src/ui/dashboard_app.py` | UI Streamlit multi-dominio |
| `config/settings.py` | Configuración central: paths, áreas, turnos, rangos fisiológicos |
| `config/restaurant_settings.py` | Configuración dominio restaurant |
| `config/clinic_settings.py` | Configuración dominio clinic |

---

## 13. Documentación Complementaria

- `docs/supuestos_v2.md`: 917 líneas con Q&A extensivo sobre supuestos funcionales, estadísticos y de modelado. Incluye diccionarios de datos completos por dominio y justificación detallada de cada decisión de diseño.
- `docs/guia_lectura_graficos_shap.md`: Guía de interpretación de cada gráfico SHAP con glosario de factores por dominio y construcción conceptual de cada variable.
- `docs/arquitectura_validada.md`: Arquitectura vigente y contraste con versiones anteriores.
- `docs/pipeline_actualizado.md`: Referencia operativa de comandos, stages y artefactos.
