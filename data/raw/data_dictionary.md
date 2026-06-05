# Diccionario de Datos - `data/raw`

Este documento describe la estructura observada en los archivos CSV ubicados en `data/raw`.
Los tipos y significados se infieren a partir de los encabezados y muestras de datos disponibles.

## 1. `employees.csv`

Descripcion: catalogo maestro de colaboradores.

Granularidad: 1 fila por empleado.

Clave aparente: `employee_id`.

| Columna | Tipo inferido | Descripcion |
|---|---|---|
| `employee_id` | string | Identificador unico del empleado con formato tipo `EMP_001`. |
| `name` | string | Nombre completo del empleado. |
| `age` | entero | Edad del empleado en anios. |
| `gender` | string/categorica | Sexo o genero reportado, observado como `M` o `F`. |
| `bmi` | decimal | Indice de masa corporal del empleado. |
| `education_level` | entero/categorica ordinal | Nivel educativo codificado numericamente. |
| `plant_area` | string/categorica | Area de trabajo principal dentro de la planta, por ejemplo `destilacion` u `oficinas`. |
| `position` | string/categorica | Puesto o rol laboral del empleado. |
| `seniority_years` | entero | Antiguedad laboral en anios. |
| `shift_pattern` | string/categorica | Patron habitual de turno, por ejemplo `fijo` o `rotativo`. |
| `distance_to_work_km` | decimal | Distancia estimada entre domicilio y centro de trabajo en kilometros. |
| `children` | entero | Numero de hijos reportado. |
| `social_drinker` | booleano | Indicador de consumo social de alcohol. |
| `smoker` | booleano | Indicador de tabaquismo. |
| `hire_date` | fecha | Fecha de contratacion del empleado en formato `YYYY-MM-DD`. |

## 2. `work_records.csv`

Descripcion: registro diario de actividad laboral por empleado.

Granularidad: 1 fila por empleado por fecha.

Clave aparente: `employee_id` + `date`.

| Columna | Tipo inferido | Descripcion |
|---|---|---|
| `employee_id` | string | Identificador del empleado. |
| `date` | fecha | Fecha del registro diario. |
| `shift` | string/categorica | Turno asignado en el dia, observado como `diurno`, `vespertino` o `nocturno`. |
| `area_assigned` | string/categorica | Area operativa asignada en esa fecha. |
| `workload_score` | decimal | Puntaje de carga laboral del dia. Parece una escala continua calculada. |
| `hours_worked` | decimal | Horas efectivamente trabajadas en la fecha. |
| `overtime_hours` | decimal | Horas extra trabajadas en la fecha. |
| `consecutive_work_days` | entero | Numero de dias laborados consecutivos hasta esa fecha. En dias de descanso se observa `0`. |
| `is_holiday` | booleano | Indicador de dia feriado. |
| `is_rest_day` | booleano | Indicador de dia de descanso. |

## 3. `absenteeism.csv`

Descripcion: registro de ausencias laborales por empleado.

Granularidad: 1 fila por evento de ausencia por empleado por fecha.

Clave aparente: no completamente garantizada; operativamente puede tratarse como `employee_id` + `date` + `absence_reason`.

| Columna | Tipo inferido | Descripcion |
|---|---|---|
| `employee_id` | string | Identificador del empleado ausente. |
| `date` | fecha | Fecha de la ausencia. |
| `absence_reason` | string/categorica | Motivo principal de ausencia, por ejemplo `musculoesqueletico`, `respiratorio`, `mental_conductual`. |
| `absence_hours` | decimal | Cantidad de horas de ausencia registradas en la fecha. |
| `is_absent` | booleano/binaria | Indicador de ausencia. En las muestras observadas toma valor `1`. |

Nota: este archivo parece contener solo dias con ausencia registrada, no el calendario completo de asistencia.

## 4. `biometrics.csv`

Descripcion: mediciones biometricas y de bienestar por empleado y fecha.

Granularidad: 1 fila por empleado por fecha.

Clave aparente: `employee_id` + `date`.

| Columna | Tipo inferido | Descripcion |
|---|---|---|
| `employee_id` | string | Identificador del empleado. |
| `date` | fecha | Fecha de la medicion o consolidado diario. |
| `hr_mean_bpm` | decimal | Frecuencia cardiaca media del dia, en latidos por minuto. |
| `hr_min_bpm` | decimal | Frecuencia cardiaca minima del dia. |
| `hr_max_bpm` | decimal | Frecuencia cardiaca maxima del dia. |
| `hr_std_bpm` | decimal | Desviacion estandar de la frecuencia cardiaca diaria. |
| `hrv_rmssd_ms` | decimal | Variabilidad cardiaca medida como RMSSD, en milisegundos. |
| `spo2_mean_pct` | decimal | Saturacion media de oxigeno, en porcentaje. |
| `spo2_min_pct` | decimal | Saturacion minima de oxigeno, en porcentaje. |
| `skin_temp_mean_c` | decimal | Temperatura media de la piel, en grados Celsius. |
| `sleep_duration_hours` | decimal | Duracion total del suenio, en horas. |
| `sleep_efficiency_pct` | decimal | Eficiencia del suenio, en porcentaje. |
| `deep_sleep_pct` | decimal | Porcentaje de suenio profundo. |
| `stress_score` | decimal | Puntaje continuo de estres. Parece una metrica derivada. |
| `steps` | entero | Numero de pasos registrados en el dia. |
| `data_quality_score` | decimal | Puntaje de calidad o completitud de la medicion. En la muestra se observa entre `0` y `1`. |
| `_is_sick` | booleano/binaria | Indicador auxiliar de enfermedad o condicion de salud en esa fecha. |
| `_absence_reason` | string/categorica | Motivo de ausencia o enfermedad asociado, vacio cuando no aplica. |

Nota: las columnas con prefijo `_` parecen variables auxiliares o de etiquetado derivadas, no necesariamente mediciones directas del sensor.

## Relaciones sugeridas entre archivos

| Archivo | Relacion principal |
|---|---|
| `employees.csv` | Tabla maestra de empleados. |
| `work_records.csv` | Se relaciona con `employees.csv` por `employee_id`. |
| `absenteeism.csv` | Se relaciona con `employees.csv` por `employee_id` y con registros diarios por `employee_id` + `date`. |
| `biometrics.csv` | Se relaciona con `employees.csv` por `employee_id` y con registros diarios por `employee_id` + `date`. |

## Observaciones

- Los nombres de columnas usan `snake_case` de forma consistente.
- Hay variables categoricas codificadas como texto y al menos una variable ordinal codificada numericamente: `education_level`.
- En `employees.csv` existen datos personales identificables en la columna `name`.
- Las fechas observadas estan en formato ISO `YYYY-MM-DD`.
