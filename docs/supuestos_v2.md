# Supuestos V2

## Proposito

Este documento amplía `docs/supuestos.md` y responde en mayor detalle dudas funcionales, estadísticas y de modelado del proyecto.

La idea es dejar explícito:

- qué hace realmente cada flujo
- por qué se tomaron ciertas decisiones
- qué significan los conceptos de modelo, riesgo, dotación y SHAP
- cómo están construidos los datos
- qué simplificaciones se introducen para viabilizar el pipeline sintético

---

## 1. Por Qué El Modelo No Consume Directamente Los CSV Raw

El pipeline no usa los raw CSV como fuente principal de entrenamiento o inferencia por tres razones técnicas y una razón funcional.

### 1.1 Separación entre datos de entrada y datos analíticos

Los raw CSV son archivos de entrada o de inspección humana. Contienen información aún no consolidada, por ejemplo:

- empleados
- registros de trabajo
- biometría
- ausentismo
- demanda
- calendarios

El modelo no trabaja sobre esas fuentes de forma directa porque necesita una tabla final de features ya unificada, limpia y agregada.

### 1.2 ETL produce una vista analítica estable

El ETL crea una tabla materializada de features por dominio:

- industrial: `ml_features`
- restaurant: `restaurant_ml_features`
- clinic: `clinic_ml_features`

Esas tablas ya contienen:

- joins hechos
- variables derivadas
- lags
- agregaciones por unidad de decisión
- targets para entrenamiento

Eso evita repetir merges y transformaciones cada vez que se entrena o infiere.

### 1.3 SQLite actúa como capa intermedia reproducible

SQLite funciona aquí como una mini base analítica local. Sus ventajas en este proyecto son:

- persistir tablas intermedias y finales
- facilitar debugging con consultas simples `SELECT * FROM ...`
- asegurar que entrenamiento, inferencia y reporting lean exactamente la misma versión del dato transformado

### 1.4 Motivo funcional

Entrenamiento, inferencia y reporting comparten la misma base ya preparada. Eso reduce inconsistencias.

En otras palabras:

```text
raw CSV -> ETL -> SQLite/features materializadas -> modelo/reporting/UI
```

No se eligió CSV raw como entrada directa del modelo porque el modelo necesita datos ya consolidados a nivel de negocio, no eventos crudos.

---

## 2. Qué Significa "Correlaciones Plausibles Pero Simplificadas"

Cuando se dice que los datos tienen correlaciones plausibles, se quiere decir que los generadores intentan respetar relaciones razonables del mundo real.

Ejemplos plausibles:

- turno nocturno tiende a empeorar descanso
- mayor fatiga tiende a elevar ausentismo
- más reservas tienden a aumentar necesidad de dotación
- más exposición a calor tiende a elevar estrés fisiológico

Pero son simplificadas porque:

- no vienen calibradas con una base real de producción
- usan fórmulas sintéticas, no relaciones estimadas estadísticamente sobre una operación real
- no capturan todas las variables ocultas del negocio
- no modelan causalidad compleja multietapa

### Ejemplo concreto

En industrial, una parte del generador asume que el calor sube frecuencia cardíaca y estrés. Eso es plausible fisiológicamente.

Pero es simplificado porque no modela:

- temperatura real ambiente por hora
- EPP utilizado
- pausas efectivas
- condición médica previa detallada

Entonces la señal tiene sentido operativo, pero no pretende ser un gemelo clínico exacto.

---

## 3. Diferencia Entre Dotación Requerida, Dotación Actual, Regresor Y Clasificador

Esta es una de las distinciones más importantes del sistema.

## 3.1 Dotación requerida

La dotación requerida es la cantidad de personas que la operación **necesita** para funcionar de acuerdo con una regla de negocio.

No la aprende el modelo.

Se calcula con fórmulas o tablas como:

- industrial: tabla fija por área y turno
- restaurant: función de demanda por franja
- clinic: función de carga asistencial por unidad y turno

Pregunta que responde:

- "¿Cuántas personas debería tener para operar bien?"

## 3.2 Dotación actual

La dotación actual es la cantidad de personas que efectivamente quedaron disponibles u operando según el dato generado/observado.

En sintético significa:

- personal programado
- menos ausencias
- menos restricciones de disponibilidad según el dominio

Pregunta que responde:

- "¿Con cuántas personas terminé realmente operando?"

## 3.3 Dotación predicha

La dotación predicha es la estimación del modelo sobre cuántas personas quedarán realmente disponibles.

Esa sí la predice un modelo de regresión.

Pregunta que responde:

- "¿Cuánta gente creo que voy a tener disponible?"

## 3.4 Qué hace un regresor

Un regresor predice una variable numérica continua.

En este proyecto predice valores como:

- `predicted_headcount`
- `predicted_headcount_total`

Ejemplo conceptual:

- 18.4 personas
- 6.9 personas
- 23.1 personas

## 3.5 Qué hace un clasificador

Un clasificador predice una clase o probabilidad.

Aquí predice:

- si habrá déficit o no
- qué probabilidad hay de déficit
- qué probabilidad hay de déficit por rol crítico

Ejemplos:

- `predicted_deficit_probability = 0.73`
- `predicted_has_deficit = 1`
- `predicted_role_deficit_prob_garzon = 0.81`

## 3.6 Resumen operativo

```text
dotación requerida = necesidad de negocio
dotación actual = disponibilidad observada
dotación predicha = disponibilidad estimada por regresor
riesgo de déficit = probabilidad estimada por clasificador
```

---

## 4. Qué Significa "SHAP Explica Cómo Decide El Modelo, No Demuestra Causalidad Real"

SHAP dice:

- qué variables usó el modelo para llegar a una predicción
- cuánto pesó cada variable
- si la empujó hacia arriba o hacia abajo

Pero no dice:

- que esa variable sea la causa verdadera del fenómeno en el mundo real

### Ejemplo

Si SHAP muestra que `fatiga promedio` empuja el riesgo, eso significa:

- el modelo usa esa variable como señal fuerte para pronosticar déficit

No significa necesariamente:

- que si elimino fatiga, el déficit desaparecerá con certeza

Puede pasar que `fatiga promedio` sea también proxy de:

- mala planificación
- sobrecarga histórica
- turnos peak mal cubiertos
- estacionalidad de ausentismo

Por eso SHAP es herramienta de interpretación del modelo, no prueba experimental ni causal.

---

## 5. Flujo Industrial: Diccionario De Datos

La unidad de decisión es:

```text
plant_area + shift + date
```

Eso significa que la fila final del modelo representa la operación agregada de un área de planta en un turno de un día dado.

## 5.1 Tabla `employees`

| Campo | Tipo | Descripción | Cómo se genera | Uso |
|---|---|---|---|---|
| `employee_id` | string | Identificador del trabajador | secuencial `EMP_001...` | joins |
| `name` | string | Nombre sintético | Faker | solo trazabilidad |
| `age` | int | Edad | normal truncada | baseline fisiológico |
| `gender` | string | Sexo/género sintético | Bernoulli sesgada | perfil demográfico |
| `bmi` | float | índice de masa corporal | normal truncada | baseline fisiológico |
| `education_level` | int | nivel educacional codificado | distribución discreta | perfil socio-laboral |
| `plant_area` | string | área principal de trabajo | muestreo ponderado | unidad de decisión |
| `position` | string | cargo dentro del área | sample de catálogo por área | contexto laboral |
| `seniority_years` | int | antigüedad | correlacionada con edad | estabilidad/experiencia |
| `shift_pattern` | string | patrón de turno (`rotativo`/`fijo`) | reglas por área | turnos |
| `distance_to_work_km` | float | distancia al trabajo | lognormal truncada | perfil personal |
| `children` | int | número de hijos | Poisson | contexto personal |
| `social_drinker` | bool | consume alcohol socialmente | Bernoulli | perfil de hábitos |
| `smoker` | bool | fumador | Bernoulli | baseline fisiológico |
| `hire_date` | date | fecha de contratación | derivada de antigüedad | trazabilidad |

## 5.2 Tabla `work_records`

| Campo | Tipo | Descripción | Cómo se genera | Uso |
|---|---|---|---|---|
| `employee_id` | string | FK del empleado | join | enlace |
| `date` | date | día simulado | calendario diario | agregación temporal |
| `shift` | string | turno del día | patrón fijo/rotativo | unidad de decisión |
| `area_assigned` | string | área operada ese día | área base del empleado | unidad de decisión |
| `workload_score` | float | intensidad operativa | base normal + ajuste por riesgo área | carga laboral |
| `hours_worked` | float | horas del turno | uniforme si trabaja | jornada |
| `overtime_hours` | float | horas extra | exponencial con prob. por riesgo | desgaste |
| `consecutive_work_days` | int | días seguidos trabajando | contador por calendario | fatiga |
| `is_holiday` | bool | si el día es festivo | calendario | contexto |
| `is_rest_day` | bool | si el empleado descansó | regla de ciclo semanal | cobertura |

## 5.3 Tabla `biometrics_raw` / `biometrics_clean`

Campos relevantes:

| Campo | Tipo | Descripción | Uso |
|---|---|---|---|
| `hr_mean_bpm` | float | frecuencia cardíaca media | fatiga/estrés |
| `hrv_rmssd_ms` | float | variabilidad cardíaca | recuperación |
| `spo2_mean_pct` | float | saturación media | estado fisiológico |
| `skin_temp_mean_c` | float | temperatura periférica | calor / infección |
| `sleep_duration_hours` | float | horas de sueño | descanso |
| `sleep_efficiency_pct` | float | eficiencia del sueño | calidad de recuperación |
| `stress_score` | float | proxy sintético de estrés | fatiga y riesgo |
| `steps` | int | pasos diarios | carga física |
| `_is_sick` | int | indicador interno de enfermedad | ausentismo sintético |
| `_absence_reason` | string | motivo interno de ausencia | trazabilidad sintética |

## 5.4 Tabla `absenteeism`

| Campo | Tipo | Descripción | Uso |
|---|---|---|---|
| `employee_id` | string | trabajador ausente | join |
| `date` | date | día de ausencia | temporal |
| `absence_reason` | string | motivo sintético | explicación |
| `absence_hours` | float | horas perdidas | severidad |
| `is_absent` | int/bool | indicador de ausencia | target operativo |

## 5.5 Tabla `ml_features`

Es la tabla final del modelo.

Ejemplos de campos:

| Campo | Tipo | Descripción |
|---|---|---|
| `plant_area` | string | área agregada |
| `shift` | string | turno agregado |
| `date` | date | día agregado |
| `scheduled_headcount` | int | personas programadas |
| `actual_headcount` | int | personas realmente presentes |
| `required_headcount` | int | personas requeridas por regla |
| `has_deficit` | int/bool | si faltó personal |
| `deficit_count` | int | magnitud del déficit |
| `absentee_rate` | float | ausentismo agregado |
| `area_shift_absentee_rate_lag1..7` | float | memoria histórica |
| `actual_headcount_lag1..7` | float | memoria operativa |

---

## 6. Por Qué Cada Área Industrial Tiene Perfil De Riesgo Fijo

### 6.1 Qué significa perfil de riesgo fijo

Quiere decir que el área, por su naturaleza, aporta una carga estructural al sistema aunque no cambie la demanda diaria.

Ejemplo:

- `cracking` se considera inherentemente más riesgosa que `oficinas`

No porque hoy haya pasado algo especial, sino porque la operación base ya se supone más exigente.

### 6.2 Qué es `heat_exposure=True`

Es una bandera binaria que indica exposición térmica relevante en esa área.

Implicancia práctica en el generador biométrico:

- sube temperatura de piel
- sube frecuencia cardíaca
- puede aumentar estrés fisiológico

No significa que haya un sensor real ni temperatura real por hora. Es una simplificación de contexto ambiental.

### 6.3 Qué es `base_risk`

Es un puntaje base numérico del área usado para empujar la carga operativa.

Por ejemplo, en `work_records` se usa para ajustar `workload_score`.

Conceptualmente:

```text
workload_score = carga_base + base_risk * 20 + ruido
```

### 6.4 Por qué existe esta estructura

Porque el flujo industrial no modela demanda real por producción o throughput día a día. Entonces necesita una forma de capturar que no todas las áreas son iguales.

### 6.5 Implicancias

- el riesgo no es puramente dinámico; parte del riesgo está embebido en el contexto del área
- las áreas "difíciles" tenderán a generar más carga y señales fisiológicas peores
- el modelo puede aprender que ciertos segmentos son estructuralmente más expuestos

### 6.6 Limitación

Eso hace el dominio más simple, pero también más rígido. Si mañana quisieras que el riesgo dependa de carga productiva real diaria, habría que reemplazar o complementar este esquema.

---

## 7. Por Qué La Dotación Requerida Industrial Está Fija

La dotación requerida industrial hoy viene de `REQUIRED_STAFF` porque el dominio se diseñó como un problema de cobertura mínima operacional, no de demanda comercial variable.

La lógica es:

- cada área y turno tiene un mínimo de personas para operar con seguridad y continuidad
- ese mínimo no cambia día a día en la versión actual del modelo

### Por qué se hizo así

Porque en industria muchas veces el mínimo operativo depende más de:

- seguridad operacional
- continuidad de planta
- requisitos técnicos del turno
- operación de equipos críticos

que de una demanda externa diaria como cubiertos o pacientes.

### Ventajas de este supuesto

- simplifica la lógica
- hace clara la comparación entre personal requerido y presente
- sirve bien para un primer modelo de déficit de cobertura

### Limitaciones

- no capta días con carga productiva excepcional
- no distingue campañas, mantenciones mayores o contingencias reales

---

## 8. Por Qué Se Asumen Los Supuestos Biométricos Y De Ausentismo En Industrial

### 8.1 Baselines por edad, BMI y tabaquismo

Se usan porque son tres variables simples que permiten dar heterogeneidad fisiológica sin construir historia clínica detallada.

Razón del supuesto:

- empleados distintos no deberían partir todos con el mismo cuerpo base
- se necesita variabilidad individual mínima para que el modelo no vea una población artificialmente homogénea

### 8.2 Turno nocturno y calor degradan recuperación

Se asume esto porque es coherente con literatura ocupacional básica y con intuición operacional:

- noche altera descanso y ritmos circadianos
- calor físico eleva estrés cardiovascular y térmico

### 8.3 Eventos de salud predeterminados

Se introducen eventos de incubación y enfermedad para que el ausentismo no sea puramente aleatorio.

Razón del supuesto:

- si la persona va a ausentarse por enfermedad, es razonable que existan señales previas deterioradas

### 8.4 Ausentismo condicionado a biometría enferma

Se hace para que haya una cadena más coherente:

```text
empeora señal fisiológica -> aumenta probabilidad de ausencia
```

### 8.5 Pequeña ausencia residual no médica

Se deja una probabilidad baja de ausencia no médica para evitar que el sistema sea completamente determinista y médico-causal.

Razón:

- en la realidad existen licencias, permisos, problemas personales y contingencias administrativas

### 8.6 Implicancia global

El flujo industrial asume que la cobertura se degrada más por salud y exigencia operacional que por reglas administrativas complejas.

---

## 9. Qué Significa ETL Y Por Qué Se Agrega A Nivel Área-Turno-Día

### 9.1 Qué es ETL

ETL significa:

- `Extract`: extraer datos
- `Transform`: transformarlos
- `Load`: cargarlos en una estructura final

### 9.2 Qué es staffing en términos chilenos

En este contexto, `staffing` equivale mejor a:

- dotación
- cobertura de personal
- disponibilidad de personal operativo

### 9.3 Por qué se limpia e imputa biometría

Porque el generador puede introducir ruido, artefactos y extremos no razonables. El pipeline quiere aproximarse a un dataset que parezca provenir de sensores reales depurados.

### 9.4 Por qué se normaliza por individuo

Porque comparar pulsaciones absolutas entre personas puede ser menos útil que ver desvíos respecto de su propio baseline.

### 9.5 Por qué se agrega a `area + shift + date`

Porque la decisión final del negocio no es normalmente "qué hago con Juan a las 13:42", sino:

- "¿este turno de esta área quedó cubierto o no?"

### 9.6 Supuesto implícito real

Se asume que para la inferencia final de dotación el nivel correcto de resolución es operativo-agregado, no individual.

Eso reduce detalle, pero hace el output más accionable para jefaturas operativas.

---

## 10. Industrial: Calendario De Festivos Chilenos

El flujo industrial ahora usa festivos chilenos como base del calendario simulado.

Eso implica:

- los flags de feriado en industrial quedan alineados con el contexto país usado en el resto del proyecto
- la simulación de días especiales es coherente con una lectura de negocio chilena
- los outputs dejan de depender de un calendario previo no alineado con Chile

Limitación actual:

- el set implementado corresponde al período 2025 usado en la simulación y no a una librería completa multi-año de feriados móviles
- si se extiende el horizonte temporal o cambia el año de simulación, conviene recalibrar o externalizar este calendario

---

## 11. Restaurant: Por Qué La Unidad Es `date + service_period`

En restaurant la pregunta operativa clave no es "qué contrato tiene cada persona" sino:

- "qué tan exigente será el bloque de servicio y cuánta gente necesito en ese bloque"

Por eso la unidad analítica es:

```text
fecha + franja operativa
```

### Por qué no turno contractual

Porque el dolor operacional en restaurantes suele concentrarse en picos de servicio, no en la etiqueta contractual de la jornada.

### Por qué solo esas franjas

Las franjas elegidas representan los bloques considerados más relevantes:

- pre-almuerzo / arranque
- peak almuerzo
- peak cena
- cierre tardío / segunda ola

### Sobre 15:00-19:00

Hoy el pipeline no la modela como franja separada porque se priorizó el modelado de bloques de mayor señal operativa. Eso no significa necesariamente que el local esté cerrado; significa que esa ventana no fue convertida en unidad analítica propia.

---

## 12. Qué Significa "Casual Dining" En Términos Chilenos

En el contexto de este proyecto, `casual dining` debe leerse como algo parecido a:

- restaurante de servicio completo
- ticket medio
- atención en salón
- mezcla de clientes con reserva y sin reserva
- presencia fuerte de almuerzo y cena
- coexistencia de sala y delivery

No representa:

- fast food puro
- fine dining
- casino institucional
- café de baja complejidad

### Términos clave en chileno

`walk-ins`

- clientes que llegan sin reserva
- en chileno operativo: demanda espontánea o clientes sin reserva

`covers`

- cubiertos
- en la práctica: número de clientes atendidos/esperados

`delivery`

- pedidos a despacho

`service_period`

- franja horaria operativa del servicio

---

## 13. Por Qué La Dotación Requerida De Restaurant Es Distinta A Industrial

En industrial la necesidad base es más estructural y de seguridad.

En restaurant la necesidad es mucho más sensible a la demanda del servicio.

Por eso cambia la lógica:

- industrial: regla casi fija por área/turno
- restaurant: regla dinámica por cubiertos, delivery y contexto

### Intuición de negocio

Una planta no necesita 30% más operadores porque llegaron más clientes ese día.

Un restaurante sí puede necesitar más garzones o cocina si suben:

- reservas
- clientes espontáneos
- delivery
- eventos locales

---

## 14. Restaurant: Diccionario De Datos Operativos Clave

| Campo | Tipo | Descripción en términos chilenos | Cómo se interpreta |
|---|---|---|---|
| `covers` | int | cubiertos, es decir, clientes a atender | volumen de servicio en sala |
| `delivery_orders` | int | pedidos a despacho | carga adicional a cocina y operación |
| `service_period` | string/categoría | franja operativa (`11_13`, `13_15`, `19_21`, `21_23`) | bloque de servicio |
| `is_holiday` | bool | si el día es festivo | shock de demanda/contexto |
| `is_weekend` | bool | si cae sábado o domingo | patrón distinto de consumo |
| `local_event_flag` | bool | si hay evento local que empuja afluencia | sobre-demanda contextual |

---

## 15. Restaurant: Escalas Operativas Implícitas

Cuando el documento dice que el staff se convierte por escalas implícitas, significa que el código usa relaciones del tipo:

- cada cierta cantidad de cubiertos necesita 1 garzón
- cada cierta carga de cocina necesita 1 cocinero de línea

Ejemplos reales del código:

- `garzon = ceil(effective_demand / 22)`
- `cocinero_linea = ceil((covers + delivery_orders) / 30)`
- `ayudante_cocina = ceil((covers + delivery_orders) / 45)`

### Qué implica

- la productividad se fija manualmente
- no se aprende desde datos históricos
- se asume que esa relación unidades de servicio / personas es estable dentro del dominio

### Por qué se hizo así

Porque en un simulador sintético primero necesitas una regla de negocio razonable para construir la necesidad operacional.

Si no defines eso, no existe una `dotación requerida` coherente contra la cual medir déficit.

---

## 16. Restaurant: Por Qué Esos Roles Y Por Qué Esos Roles Críticos

Los roles modelados representan una operación de salón + cocina + soporte.

### Roles incluidos

- `garzon`: atención directa de mesas
- `host`: recepción y coordinación de ingreso
- `cajero`: cierre/transacción y apoyo front
- `cocinero_linea`: núcleo operativo de cocina
- `ayudante_cocina`: apoyo de ejecución
- `runner`: traslado / apoyo de salón
- `copero`: soporte de limpieza y reposición
- `jefe_turno`: coordinación operativa de la franja

### Por qué críticos son `garzon`, `cocinero_linea`, `jefe_turno`

Porque representan tres cuellos de botella distintos:

- front de atención
- producción de cocina
- coordinación operacional

Si falla alguno:

- cae velocidad de servicio
- aumenta espera
- se degrada experiencia del cliente
- la operación pierde capacidad de reacción

### Por qué el backup es discreto y no libre

Porque en la realidad no cualquier persona puede cubrir cualquier rol con la misma eficacia.

Ejemplo:

- un runner puede apoyar a garzón
- un ayudante puede apoyar cocina de línea en cierto grado
- pero un reemplazo total libre sería poco realista

---

## 17. Restaurant: Contratos Y Capacidad Máxima Por Empleado

Se modelan:

- `full_time`
- `part_time`
- `weekend_only`

### Por qué

Porque la operación gastronómica suele tener mezcla de:

- equipo estable
- apoyos parciales
- refuerzos de fin de semana

### Por qué la capacidad se modela en períodos por día

Porque el dominio está definido por franjas de servicio, no por turnos continuos completos. Entonces el control fino de disponibilidad se aproxima como cuántas franjas puede cubrir una persona en un día.

---

## 18. Restaurant: Demanda, Contexto Y Razón Del Modelo

La demanda se modela por franja porque el negocio necesita responder a bloques de servicio, no a cada minuto del salón.

Variables principales:

- `forecast_covers`: cubiertos esperados
- `actual_covers`: cubiertos realmente ocurridos
- `reservation_count`: reservas
- `walk_in_ratio`: proporción de clientes sin reserva
- `delivery_order_volume`: pedidos a despacho
- `forecast_sales`: ventas proyectadas
- `actual_sales`: ventas reales

### Por qué clima se mete como shock sintético simple

Porque el objetivo no es modelar meteorología real, sino introducir una fuente razonable de variación contextual que pueda mover:

- sala vs delivery
- intensidad de demanda
- comportamiento del cliente

### Por qué no minuto a minuto

Porque para este caso de uso la decisión operativa relevante es de bloque horario. Un modelo minuto a minuto sería mucho más costoso y no necesariamente más útil para el nivel de decisión actual.

---

## 19. Restaurant: Por Qué Se Modela Biometría Y Salud Operativa

Se asume que el riesgo de déficit no depende solo de la demanda, sino también de la capacidad real del equipo para sostenerla.

### Variables y por qué existen

- `fatiga`: para capturar desgaste acumulado
- `estrés`: para reflejar tensión operativa y fisiológica
- `mal sueño`: para reflejar mala recuperación
- `steps`: para representar carga física / movimiento

### Por qué se introducen eventos sintéticos de salud

Porque si el ausentismo fuera totalmente aleatorio, el pipeline perdería coherencia entre salud, carga y cobertura.

Eventos modelados:

- gastrointestinal
- respiratorio
- fatiga
- musculoesquelético
- stress

La idea es que existan patrones explicables, no solo ruido.

---

## 20. Restaurant: ETL, Lags Y Memoria Operativa

### Por qué se agrega a `date + service_period`

Porque esa es la unidad real de decisión del dominio.

### Qué significan `lag1` y `lag7`

- `lag1`: valor del período comparable inmediatamente anterior dentro de la misma franja
- `lag7`: valor del período comparable una semana antes dentro de la misma franja

### Por qué existen

Porque la operación tiene memoria:

- si ayer el almuerzo estuvo tensionado, hoy puede repetir patrón
- si la semana pasada la misma franja tuvo pico, eso puede ser informativo

### Por qué el modelo aprende déficit total y por rol

Porque el negocio necesita dos respuestas distintas:

- ¿la franja quedará corta en total?
- ¿qué rol crítico es el que más probablemente falle?

---

## 21. Clinic: Diccionario De Variables De Capacidad Y Carga

| Campo | Tipo | Descripción | Rol en la regla de dotación |
|---|---|---|---|
| `patient_volume` | int | pacientes esperados o atendidos en la unidad/turno | carga global de atención |
| `high_acuity_cases` | int | casos de mayor complejidad clínica relativa | eleva intensidad de trabajo |
| `scheduled_procedures` | int | procedimientos ambulatorios programados | demanda técnica y de coordinación |
| `active_care_stations` | int | boxes o estaciones activas | capacidad física ocupada |
| `imaging_backlog_cases` | int | rezago pendiente en imagenología | presión acumulada |
| `respiratory_alert_cases` | int | casos asociados a alerta respiratoria | presión epidemiológica-operativa |
| `respiratory_case_ratio` | float | peso relativo de casos respiratorios | multiplicador de presión |
| `is_holiday` | bool | si el día es festivo | modula carga y oferta |
| `is_weekend` | bool | si es fin de semana | cambia la demanda |

---

## 22. Clinic: Por Qué Cada Unidad Tiene Lógica De Capacidad Distinta

No todas las unidades ambulatorias trabajan igual.

Ejemplos:

- `consulta_general` depende más de flujo de pacientes
- `procedimientos_ambulatorios` depende más de procedimientos y boxes
- `imagenologia` depende más de backlog, flujo y tecnólogos
- `toma_muestras` depende más de volumen y demanda espontánea

Por eso la productividad está codificada manualmente por rol y unidad.

### Por qué roles críticos son médico, enfermera y TENS

Porque son los tres perfiles que más directamente sostienen la continuidad clínica del flujo ambulatorio.

### Qué significa asignación primaria y respaldo

- `primary_unit`: unidad principal del trabajador
- `backup_unit`: unidad donde puede apoyar

### Qué significa `can_float`

Que el trabajador puede desplazarse operacionalmente a otra unidad según necesidad. Es una simplificación de flexibilidad funcional interna.

---

## 23. Clinic: Flujo Asistencial Agregado Y Por Qué No Paciente A Paciente

La demanda se resume con baselines por unidad y turno porque el nivel de decisión buscado es de cobertura por unidad, no de agenda individual.

Multiplicadores usados:

- día de semana
- feriado
- campaña de vacunación
- ventana de pago
- presión de bloque electivo
- alerta respiratoria

### Por qué se hace así

Porque un flujo paciente-a-paciente requeriría:

- agenda real
- duraciones por prestación
- secuenciación real
- disponibilidad por profesional

Eso excede el objetivo actual del simulador.

---

## 24. Clinic: Proxies Fisiológicos Y Riesgo De Cobertura

Igual que en restaurant, se asume que el desempeño de cobertura no depende solo de demanda, sino también de fatiga del equipo.

### Variables agregadas relevantes

- `stress_score`
- `fatigue_proxy`
- `sleep_duration_hours`
- `cognitive_load_score`
- `reaction_time_ms`

### Qué representan

- `cognitive_load_score`: presión mental y complejidad operacional acumulada
- `reaction_time_ms`: proxy de enlentecimiento funcional bajo fatiga/carga

### Por qué ausentismo depende de eso

Porque el supuesto del dominio es que el desgaste operativo en ambulatorio influye tanto como la demanda clínica en la estabilidad de cobertura.

---

## 25. Sobre La Consistencia País Del Modelo Industrial

La migración a festivos chilenos en industrial busca mantener consistencia entre:

- contexto país
- comportamiento operativo esperado
- interpretación de resultados por usuarios chilenos

En este documento se deja como recomendación de alineamiento del dominio industrial.

---

## 26. Conclusión General

Los supuestos del proyecto están construidos para maximizar tres cosas:

- interpretabilidad
- reproducibilidad
- utilidad operacional

Se sacrifica parte del realismo fino a cambio de:

- reglas claras
- datasets consistentes
- targets entrenables
- dashboards accionables

La revisión futura más importante para mejorar realismo no está en los modelos primero, sino en:

- las reglas de dotación requerida
- los segmentos temporales
- las escalas de productividad por rol
- el calendario por país
- la granularidad del flujo operativo modelado
