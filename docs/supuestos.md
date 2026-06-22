# Supuestos Del Sistema

## Objetivo

Este documento lista los supuestos funcionales, operativos y tecnicos que hoy usa el proyecto SkillUp para todos sus flujos de datos.

El objetivo es que negocio, producto y tecnica puedan revisar:

- que se esta modelando realmente
- que no se esta modelando
- que reglas de negocio estan fijas
- que simplificaciones se introdujeron para poder generar datos sinteticos, entrenar modelos e interpretar salidas

Este documento describe el estado actual del repositorio, no una verdad universal del negocio.

---

## 1. Supuestos Transversales Del Proyecto

### 1.1 Arquitectura general

- El proyecto se organiza como plataforma multi-dominio.
- Cada dominio implementa stages estandar: `generate`, `etl`, `train`, `infer`, `report`, `full`.
- La orquestacion es comun, pero las reglas de negocio y datos son propias de cada dominio.

### 1.2 Persistencia

- Los CSV raw son el punto de entrada y de inspeccion humana.
- SQLite es la capa estructurada intermedia para ETL, entrenamiento, inferencia y reporting.
- El modelo no consume directamente los raw CSV durante su ejecucion normal; consume tablas SQLite y/o features materializadas.

### 1.3 Reproducibilidad

- Todos los generadores sinteticos parten de una semilla aleatoria (`seed`).
- Con la misma configuracion y misma semilla, se espera el mismo dataset o uno muy cercano, dependiendo de librerias externas y orden de ejecucion.

### 1.4 Naturaleza de los datos

- Los datos son sinteticos y no representan personas reales.
- Se busca realismo operacional, no exactitud estadistica perfecta contra una empresa real.
- Las correlaciones usadas son plausibles, pero simplificadas.

### 1.5 Modelos

- La dotacion requerida no la predice un modelo; se calcula con reglas de negocio.
- La dotacion actual es observada o derivada del dato simulado.
- La dotacion predicha la estima un regresor.
- El riesgo de deficit y el riesgo por rol los estiman clasificadores.

### 1.6 Interpretabilidad SHAP

- SHAP explica como decide el modelo, no demuestra causalidad real.
- Un factor importante en SHAP significa que el modelo lo usa mucho, no que el negocio necesariamente dependa solo de ese factor.

---

## 2. Flujo Industrial

## 2.1 Supuestos de negocio

- La unidad de decision es `plant_area + shift + date`.
- Las areas modeladas son:
  - `destilacion`
  - `cracking`
  - `almacenamiento`
  - `mantenimiento`
  - `oficinas`
- Los turnos modelados son:
  - `diurno`
  - `vespertino`
  - `nocturno`

### 2.1.1 Riesgo por area

- Cada area tiene un perfil de riesgo fijo.
- Algunas areas tienen `heat_exposure=True`.
- Cada area tiene `base_risk` predefinido.

Esto implica que parte del riesgo esta estructuralmente dado por el contexto del area, no por variacion dinamica del negocio.

### 2.1.2 Dotacion requerida

- La dotacion requerida esta fija por tabla de negocio.
- No depende de demanda diaria ni de biometria.
- Se toma de `REQUIRED_STAFF` en `config/settings.py`.

Supuesto implicito:

- el minimo operativo por area/turno es estable y conocido de antemano

### 2.1.3 Distribucion de empleados

- La distribucion por area no es uniforme.
- Produccion pesada tiene mayor peso que oficinas.
- El genero esta sesgado hacia masculino.
- La edad, BMI, distancia y antiguedad siguen distribuciones sinteticas controladas.

Supuesto implicito:

- el dominio industrial representa una operacion de planta intensiva, no una organizacion de servicios ligeros

### 2.1.4 Turnos de trabajo

- Oficinas opera siempre en turno diurno.
- Otras areas pueden rotar o trabajar en patron fijo.
- Los turnos rotativos cambian cada 7 dias.
- Los empleados tienen dos dias de descanso por ciclo semanal sintetico.

### 2.1.5 Calendario laboral

- Los festivos considerados en industrial son chilenos.
- El comportamiento diario se simula con base en esos festivos.

## 2.2 Supuestos biometricos y de ausentismo

- Los biometricos se generan con baselines individuales por edad, BMI y tabaquismo.
- El turno nocturno y el calor de ciertas areas degradan descanso y variabilidad cardiaca.
- Existen eventos de salud predeterminados con fases de incubacion y enfermedad.
- El ausentismo se activa principalmente cuando el biometrico ya indica estado enfermo.
- Existe una pequena probabilidad residual de ausencia no medica.

Supuesto implicito:

- el ausentismo industrial esta mucho mas ligado a salud y carga operativa que a una capa rica de causas administrativas o contractuales

## 2.3 Supuestos ETL

- El ETL limpia y recorta señales fisiologicas a rangos plausibles.
- Hace imputacion y normalizacion individual.
- Agrega a nivel area-turno-dia.
- La tabla principal de modelado es `ml_features`.

Supuesto implicito:

- la operacion relevante para staffing puede resumirse a ese nivel agregado sin necesidad de modelado empleado-a-empleado en inferencia final

## 2.4 Lo que no modela

- cambios diarios reales de produccion o throughput
- paradas de planta
- eventos sindicales o administrativos
- skill matrices detalladas por equipo o certificacion fina

---

## 3. Flujo Restaurant

## 3.1 Supuestos de negocio

- La unidad de decision es `date + service_period`.
- El modelo no trabaja por turno contractual de empleado, sino por bloque operativo de servicio.

### 3.1.1 Franjas de servicio

- Solo se modelan estas franjas:
  - `11_13`
  - `13_15`
  - `19_21`
  - `21_23`

Supuesto implicito:

- la franja `15:00-19:00` no forma parte de la unidad analitica actual
- si el local atiende en ese bloque, hoy el pipeline no lo representa como segmento propio

### 3.1.2 Tipo de restaurante

- El dominio representa un restaurant casual dining.
- Se asume calendario chileno.
- Se asume sensibilidad a:
  - reservas
  - walk-ins
  - delivery
  - fines de semana
  - festivos
  - eventos locales

### 3.1.3 Dotacion requerida

- La dotacion requerida se calcula con reglas de negocio a partir de demanda.
- No sale de un modelo.

La regla base considera:

- `covers`
- `delivery_orders`
- `service_period`
- `is_holiday`
- `is_weekend`
- `local_event_flag`

Se convierte a staff por rol con escalas operativas implicitas, por ejemplo:

- garzon segun demanda efectiva
- cocinero_linea segun covers + delivery
- ayudante_cocina segun covers + delivery

Supuesto implicito:

- existe una relacion estable entre unidades de servicio y personas requeridas
- esa productividad no se aprende del dato; se fija en formulas

### 3.1.4 Roles y cobertura

- Los roles modelados son:
  - `garzon`
  - `host`
  - `cajero`
  - `cocinero_linea`
  - `ayudante_cocina`
  - `runner`
  - `copero`
  - `jefe_turno`
- Los roles criticos son:
  - `garzon`
  - `cocinero_linea`
  - `jefe_turno`
- Se permite cross-training limitado via `backup_role`.

Supuesto implicito:

- el reemplazo entre roles no es libre; sigue reglas discretas de backup

### 3.1.5 Tipo de contrato

- Existen `full_time`, `part_time` y `weekend_only`.
- Algunos roles tienen mayor probabilidad de part-time o solo fin de semana.
- La capacidad maxima por empleado se modela en periodos por dia, no en una jornada continua completa.

### 3.1.6 Demanda

- La demanda se construye con bases por franja y multiplicadores de contexto.
- Se modela por separado:
  - `forecast_covers`
  - `actual_covers`
  - `reservation_count`
  - `walk_in_ratio`
  - `delivery_order_volume`
  - `forecast_sales`
  - `actual_sales`
- El clima afecta mezcla y volumen, pero como shock sintetico simple.

Supuesto implicito:

- la demanda relevante puede resumirse por franja en lugar de modelar secuencia minuto a minuto dentro del servicio

## 3.2 Supuestos de salud operativa

- Se modela biometria diaria del equipo, no solo productividad.
- Se asume que fatiga, estres y mal sueno elevan probabilidad de deficit y ausentismo.
- Se usa `steps` como proxy de carga fisica / movimiento.
- Existen eventos de salud sinteticos como:
  - gastrointestinal
  - respiratorio
  - fatiga
  - musculoesqueletico
  - stress

### 3.2.1 Ausentismo

- El ausentismo depende de:
  - estres
  - fatiga
  - sueno
  - dias consecutivos
  - fin de semana / festivo
  - enfermedad sintetica
- Se distingue `short_notice` cuando el aviso llega con poca anticipacion.

Supuesto implicito:

- el problema de staffing restaurant es tan fisiologico-operativo como comercial

## 3.3 Supuestos ETL y modelado

- El ETL agrega todo a nivel `date + service_period`.
- Se generan features lag `lag1` y `lag7` para la misma franja.
- El modelo aprende tanto deficit total como deficit por rol critico.

Supuesto implicito:

- la memoria reciente del mismo bloque horario es informativa para el riesgo futuro

## 3.4 Lo que no modela

- layout de salon por mesa o seccion
- secuencia intra-franja minuto a minuto
- politica real de propinas
- mezcla de productos o complejidad de platos
- tiempos exactos de coccion y despacho
- bloque 15:00-19:00 como unidad analitica propia

---

## 4. Flujo Clinic Ambulatorio

## 4.1 Supuestos de negocio

- La unidad de decision es `date + shift + clinical_unit`.
- El dominio es una clinica ambulatoria, no hospitalaria completa.

### 4.1.1 Unidades modeladas

- `consulta_general`
- `especialidades`
- `procedimientos_ambulatorios`
- `imagenologia`
- `toma_muestras`

### 4.1.2 Turnos modelados

- `morning`
- `evening`

Supuesto implicito:

- no se modela turno nocturno
- no se modela hospitalizacion ni urgencias como flujo principal

### 4.1.3 Dotacion requerida

- Se calcula con reglas por unidad.
- Las formulas usan variables como:
  - `patient_volume`
  - `high_acuity_cases`
  - `scheduled_procedures`
  - `active_care_stations`
  - `imaging_backlog_cases`
  - `respiratory_alert_cases`
  - `respiratory_case_ratio`
  - `is_holiday`
  - `is_weekend`

Supuesto implicito:

- cada unidad tiene una logica de capacidad distinta
- la productividad staff/unidad esta codificada manualmente en reglas por rol

### 4.1.4 Composicion de personal

- Los roles son:
  - `medico`
  - `enfermera`
  - `tens`
  - `admision`
  - `tecnologo_medico`
  - `coordinador_clinico`
- Los roles criticos son:
  - `medico`
  - `enfermera`
  - `tens`
- Existe asignacion primaria y respaldo por unidad.
- Existe concepto de `can_float` para apoyar otras unidades.

### 4.1.5 Flujo asistencial

- La demanda se resume con baselines por unidad y turno.
- Se usan multiplicadores por:
  - dia de semana
  - feriado
  - campana de vacunacion
  - ventana de pago
  - presion de bloque electivo
  - alerta respiratoria

Supuesto implicito:

- la clinica ambulatoria puede explicarse con un flujo agregado por unidad/turno y no requiere programacion paciente a paciente para este caso de uso

## 4.2 Supuestos de salud y ausentismo

- Igual que restaurant, se modelan proxies fisiologicos agregados.
- Se incluye `cognitive_load_score` y `reaction_time_ms` como carga mental / operacional.
- El ausentismo depende de estres, fatiga, sueno, carga cognitiva e infeccion respiratoria.

Supuesto implicito:

- el riesgo de cobertura en ambulatorio depende tanto de demanda como del desgaste del equipo

## 4.3 Lo que no modela

- hospitalizacion prolongada
- urgencias abiertas 24/7
- deriva entre clinicas
- pabellon mayor
- agenda por medico individual
- reglas contractuales o legales detalladas por gremio

---

## 5. Flujo De Fichas Medicas PDF

## 5.1 Objetivo del flujo

- Generar PDFs de fichas medicas laborales a partir de `employees.csv`.
- Extraer esos PDFs de vuelta a CSV estructurado.
- Permitir validacion exacta contra el archivo de referencia.
- Permitir alimentar ETL industrial con `employees_from_forms.csv`.

## 5.2 Supuestos del formato

- Las fichas PDF siguen un layout fijo.
- Los campos y secciones tienen nombres y posiciones esperadas.
- El parser espera un PDF con texto extraible, no una variedad libre de formularios.
- El esquema objetivo coincide con `EMPLOYEE_COLUMNS`.

Supuesto implicito:

- el PDF no es un documento medico real de alta variabilidad, sino una plantilla estandarizada para captura

## 5.3 Extraccion

- El flujo usa `pdftotext` como ruta principal.
- Si el PDF no contiene texto util, existe una ruta OCR prevista, pero hoy no esta plenamente habilitada.
- Si no existe `pdftotext`, el flujo falla.
- Si se requiere OCR y no esta disponible `tesseract`, el flujo falla.

Supuesto implicito:

- los PDFs generados o ingeridos son razonablemente legibles para extraccion estructurada

## 5.4 Validacion

- La validacion exige igualdad exacta entre CSV candidato y CSV referencia.
- No es una validacion tolerante a pequenas diferencias semanticas.

Supuesto implicito:

- este flujo se usa como prueba de round-trip estructurado, no como OCR flexible del mundo real

---

## 6. Supuestos De SQLite Y Features

- Cada dominio materializa sus features en una tabla principal.
- El entrenamiento e inferencia leen desde SQLite con queries directos.
- Las tablas SQLite se tratan como fuente confiable una vez corrido el ETL.

Supuesto implicito:

- si el ETL termino bien, el estado de la base refleja la verdad operacional del pipeline para ese dominio

---

## 7. Supuestos De UI Y Reporting

- La UI es una capa de consumo y exploracion, no la fuente primaria de verdad del pipeline.
- Las metricas y graficos dependen de artefactos ya generados por `infer` o `report`.
- Los graficos SHAP explican patrones del modelo sobre el dataset visible y sus filtros.

Supuesto implicito:

- el usuario revisa un estado de artefactos ya calculados, no una simulacion online en tiempo real

---

## 8. Riesgos De Interpretacion

- Un supuesto razonable para datos sinteticos no garantiza validez externa en produccion real.
- Reglas de negocio fijas pueden volver rigidas ciertas conclusiones.
- Si un dominio cambia de operacion real, habria que recalibrar generadores, ETL y reglas de staffing.
- La interpretabilidad SHAP puede hacer muy visible una variable que es proxy de otra no modelada.

---

## 9. Recomendaciones Para Revision De Supuestos

Al revisar este documento conviene preguntarse, por dominio:

- la unidad de decision es correcta para negocio?
- la dotacion requerida esta bien definida o requiere escalas configurables?
- faltan segmentos operativos relevantes?
- los roles criticos elegidos son correctos?
- el ausentismo esta sobrerrepresentado o subrepresentado?
- los shocks de calendario, clima o carga estan bien calibrados?
- el nivel de agregacion del ETL es suficiente para el uso esperado?

Si quieres endurecer o afinar el realismo, el mejor lugar para intervenir suele ser:

- configuraciones en `config/*.py`
- formulas de `estimate_required_staff(...)`
- generadores sinteticos por dominio
- reglas de agregacion en ETL
