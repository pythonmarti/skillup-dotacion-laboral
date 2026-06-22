# Guia De Lectura De Graficos SHAP

## Objetivo

Este documento explica:

- que muestra cada grafico de la UI
- como interpretar los ejes, colores y tooltips
- que significa cada factor o variable que aparece en los graficos
- como llevar esa lectura a decisiones operativas

La idea no es leer los graficos como un cientifico de datos, sino como un operador o dueño del negocio.

---

## 1. Que es SHAP

SHAP es una forma de explicar por que el modelo predice un cierto riesgo o una cierta dotacion.

Lectura simple:

- un valor SHAP positivo significa: este factor empuja el resultado hacia arriba
- un valor SHAP negativo significa: este factor lo empuja hacia abajo
- un valor absoluto grande significa: este factor pesa mucho en la explicacion

Importante:

- SHAP explica el modelo, no prueba causalidad real por si solo
- sirve para entender que usa el modelo para decidir
- es mas util para priorizar acciones que para sacar conclusiones medicas o legales

---

## 2. Como Leer Los Colores Y Ejes

### Barras SHAP

- eje X: intensidad del impacto del factor
- eje Y: nombre del factor
- barra mas larga: ese factor explica mas
- color rojo: empuja el riesgo o el deficit hacia arriba
- color azul: compensa o empuja hacia abajo

### Heatmap SHAP

- eje X: segmento operativo
- eje Y: factor explicativo
- rojo intenso: ese factor sube el riesgo en ese segmento
- azul intenso: ese factor compensa el riesgo en ese segmento

### Scatter SHAP

- eje X: valor real del factor
- eje Y: impacto SHAP del factor
- cada punto: una observacion del dataset
- sirve para ver desde que umbral el factor empieza a ser peligroso o protector

---

## 3. Graficos Actuales En La UI

## 3.1 Narrativa Explicativa Del Modelo

No es un grafico, pero aparece arriba del resto.

Que hace:

- resume los factores que mas empujan el riesgo
- resume los factores que mas lo compensan
- destaca un segmento donde cierto factor es especialmente importante

Como leerlo:

- sirve como resumen ejecutivo para saber donde mirar primero
- es util cuando el usuario no quiere interpretar cada grafico manualmente

---

## 3.2 Factores Que Mas Explican El Riesgo De Deficit

Este grafico responde:

- por que el modelo cree que hay riesgo de quedar corto de personal

Partes:

- eje Y: factores explicativos
- eje X: `Impacto medio absoluto (SHAP)`
- color: `Empuje neto del riesgo`

Interpretacion:

- si una barra es larga, ese factor explica mucho el riesgo
- si ademas es roja, en promedio sube el riesgo
- si es azul, en promedio lo compensa

Pregunta de negocio que responde:

- "Que esta empujando el riesgo hoy?"

---

## 3.3 Factores Que Mas Explican La Dotacion Disponible

Este grafico responde:

- por que el modelo estima que habra cierta cantidad de personas realmente disponibles

Partes:

- eje Y: factores explicativos
- eje X: `Impacto medio absoluto (SHAP)`
- color: `Empuje neto sobre la dotacion`

Interpretacion:

- rojo: tiende a subir la dotacion disponible estimada
- azul: tiende a bajar la dotacion disponible estimada

Pregunta de negocio que responde:

- "Que esta sosteniendo o erosionando mi capacidad real de operar?"

---

## 3.4 Drivers Del Riesgo Por Segmento

Este grafico es un heatmap.

Responde:

- en que segmento operativo pega mas cada factor

Segmento segun dominio:

- `restaurant`: franja horaria
- `industrial`: area / turno
- `clinic`: unidad / turno

Partes:

- eje X: segmento
- eje Y: factor
- color: `Empuje neto del riesgo`

Interpretacion:

- mismo factor puede no pesar igual en todos los segmentos
- una celda muy roja identifica una combinacion sensible

Pregunta de negocio que responde:

- "Donde exactamente este factor me esta haciendo dano?"

---

## 3.5 Grafico Inferior Derecho Segun Dominio

### Restaurant: Por Que Un Rol Critico Aparece Tensionado

Este grafico reemplaza el scatter generico.

Responde:

- por que `garzon`, `cocinero_linea` o `jefe_turno` aparece como rol con mayor riesgo de deficit

Partes:

- selector: `Rol crítico a explicar`
- eje Y: factores que explican el deficit del rol
- eje X: `Impacto medio absoluto sobre el déficit del rol`
- color: `Empuje neto del déficit del rol`

Interpretacion:

- barra larga: factor clave para ese rol
- rojo: factor que empuja el deficit del rol
- azul: factor que ayuda a contenerlo

Pregunta de negocio que responde:

- "Por que el garzon aparece como el rol mas tensionado?"

### Industrial Y Clinic: Como Cambia El Impacto Segun El Valor Del Factor

Este es un scatter SHAP clasico.

Responde:

- a partir de que valor de una variable el impacto sobre el riesgo se vuelve mas fuerte

Partes:

- selector: `Factor a explorar`
- eje X: valor del factor
- eje Y: impacto SHAP sobre el riesgo
- color: segmento

Pregunta de negocio que responde:

- "Desde que umbral este factor empieza a ser realmente peligroso?"

---

## 3.6 Explicaciones De Segmentos Mas Sensibles

Es una tabla, no un grafico.

Responde:

- que factores explicaron los casos de mayor sensibilidad del modelo

Columnas tipicas:

- `date`: fecha
- `service_period` o `shift`: franja o turno
- `predicted_deficit_probability`: riesgo predicho
- `top_up_driver_1`, `top_up_driver_2`: factores que mas empujan el riesgo
- `top_down_driver_1`, `top_down_driver_2`: factores que mas lo compensan

Pregunta de negocio que responde:

- "Cuales fueron las razones concretas del riesgo en este caso?"

---

## 4. Glosario De Factores Restaurant

Esta es la parte mas importante para negocio en `restaurant`.

### Demanda Y Venta

`Cubiertos proyectados`

- cantidad de clientes esperados para esa franja
- si sube, normalmente aumenta la necesidad de personal

`Cubiertos reales`

- clientes efectivamente atendidos en la franja
- sirve para medir si la demanda real supero lo esperado

`Ventas proyectadas`

- ingresos esperados para la franja
- captura presion comercial esperada

`Ventas reales`

- ingresos realmente generados
- ayuda a entender si hubo mas actividad de la planificada

`Reservas`

- numero de reservas registradas
- suele ser un driver fuerte en almuerzo y cena

`Pedidos delivery`

- volumen de pedidos de delivery
- agrega carga a cocina y operacion aunque no aumente los clientes sentados

`Clientes sin reserva`

- proporcion estimada de walk-ins
- alto valor implica mas incertidumbre y variabilidad operacional

`Promocion activa`

- indica si habia promocion operativa o comercial
- puede elevar demanda y tensionar la dotacion

`Evento local`

- indica si hubo evento cercano o contexto local que empuja la demanda

`Weather impact score`

- puntaje que resume si el clima favorece o frena la demanda
- valores mas altos suelen reflejar condiciones que alteran el mix entre sala y delivery

### Salud Operativa Del Equipo

`Estres promedio del equipo`

- promedio del stress fisiologico estimado del staff programado
- alto valor sugiere mayor desgaste y menor resiliencia

`Horas de sueno promedio`

- promedio de horas dormidas del equipo
- valores bajos suelen empujar riesgo y errores operativos

`Eficiencia de sueno promedio`

- calidad relativa del descanso
- no solo importa cuanto se duerme, sino como se duerme

`Fatiga promedio`

- proxy agregada de fatiga del equipo
- suele combinar descanso, estres y carga de trabajo

`Promedio de steps`

- promedio de pasos del equipo
- funciona como proxy de movimiento fisico y demanda corporal durante la jornada

`Edad promedio del equipo`

- composicion etaria del personal en esa franja
- no implica problema por si sola; solo indica que el modelo encontro patron explicativo

`Carga fisica promedio del equipo`

- etiqueta usada para la variable `avg_bmi`
- representa una proxy agregada de condicion corporal del equipo
- no debe interpretarse como diagnostico individual

### Cobertura Y Ausentismo

`Ausentismo`

- proporcion de personas programadas que no asistieron

`Ausentismo de corto aviso`

- ausencias informadas con poca anticipacion
- suele ser especialmente danino porque impide reaccionar a tiempo

`Dotacion requerida proyectada`

- cantidad estimada de personas necesarias para sostener el servicio esperado

`Dotacion real`

- cantidad de personas efectivamente disponibles en la operacion

`Horas extra programadas`

- overtime acumulado de la franja
- puede ayudar a contener deficit en el corto plazo, pero tambien anticipar desgaste

`Racha promedio de dias trabajados`

- cuantos dias consecutivos viene trabajando el equipo en promedio
- si sube demasiado, suele aumentar riesgo de fatiga y ausentismo

### Memoria De Periodos Anteriores

`Deficit del turno comparable anterior`

- cuanto deficit hubo en el periodo inmediatamente comparable anterior
- ayuda a capturar inercia operativa

`Cubiertos del turno comparable anterior`

- demanda comparable de la franja anterior equivalente

`Cubiertos de la semana anterior`

- referencia historica de demanda de la misma ventana una semana antes

---

## 5. Glosario De Factores Industrial

`Dotacion requerida`

- personal necesario para cubrir area y turno

`Dotacion real`

- personal realmente disponible

`Ausentismo`

- porcentaje de ausencias en esa agregacion

`Estres promedio`

- carga fisiologica agregada del equipo

`Horas de sueno promedio`

- descanso agregado del equipo

Otros nombres generados automaticamente suelen seguir las reglas de prefijos y sufijos explicadas mas abajo.

---

## 6. Glosario De Factores Clinic

`Pacientes proyectados`

- volumen esperado de pacientes para la unidad y turno

`Procedimientos agendados`

- cantidad de procedimientos planificados

`Boxes activos`

- puestos o estaciones de atencion realmente abiertas

`Espera promedio`

- tiempo medio de espera del paciente

`Carga respiratoria`

- peso relativo de casos respiratorios en la demanda

`Ausentismo`

- porcentaje de personas ausentes

`Estres promedio del equipo`

- carga fisiologica agregada del personal

`Carga cognitiva promedio`

- proxy de presion mental del equipo en la operacion

---

## 7. Reglas Generales Para Entender Factores Nuevos

No todos los factores aparecen manualmente nombrados. Si ves uno nuevo, esta regla ayuda:

`avg_...`

- promedio de esa variable

`scheduled_...`

- valor programado o asignado

`actual_...`

- valor real observado

`service_period_...`

- franja horaria codificada como categoria

`shift_...`

- turno codificado como categoria

`plant_area_...`

- area industrial codificada como categoria

`clinical_unit_...`

- unidad clinica codificada como categoria

`season_...`

- temporada del ano

`..._lag1`

- valor del periodo comparable inmediatamente anterior

`..._lag7`

- valor del periodo comparable una semana antes

---

## 8. Como Se Construye Cada Factor Principal En Restaurant

Esta seccion explica de donde sale cada variable del dominio `restaurant` que suele aparecer en los graficos SHAP.

## 8.1 Capas Del Pipeline

En `restaurant`, los factores nacen en cuatro capas:

1. demanda operativa por `date + service_period`
2. registros de trabajo por empleado y franja
3. biometria diaria por empleado
4. agregacion ETL a nivel `date + service_period`

Eso significa que muchos factores del grafico no vienen de una sola columna original, sino de una agregacion sobre varias observaciones individuales.

## 8.2 Factores De Demanda

`Cubiertos proyectados`

- nombre tecnico: `forecast_covers`
- se genera en la simulacion de demanda por franja
- depende de una base por franja, multiplicadores por dia de semana, festivos, promociones, eventos locales y clima

Construccion conceptual:

```text
forecast_covers = base_covers * multiplicador + ruido
```

`Cubiertos reales`

- nombre tecnico: `actual_covers`
- parte de `forecast_covers`
- luego se ajusta con ruido adicional, reservas y eventos locales

`Reservas`

- nombre tecnico: `reservation_count`
- se simula por franja con una base distinta para almuerzo y cena
- luego se ajusta por multiplicadores operativos y ruido

`Pedidos delivery`

- nombre tecnico: `delivery_order_volume`
- se genera por franja segun una base de pedidos y el efecto del clima

Construccion conceptual:

```text
delivery_order_volume = base_delivery * (1 - weather_impact) + ruido
```

`Clientes sin reserva`

- nombre tecnico: `walk_in_ratio`
- representa la proporcion esperada de clientes walk-in
- cambia por franja, con mas variabilidad fuera de los bloques de reserva fuerte

`Weather impact score`

- nombre tecnico: `weather_impact_score`
- se simula como un shock de clima condicionado por la estacion
- luego se recorta a un rango acotado

Construccion conceptual:

```text
weather_impact_score = clip(normal(media_por_estacion, 0.08), -0.25, 0.25)
```

## 8.3 Factores De Cobertura Programada

`Racha promedio de dias trabajados`

- nombre tecnico: `scheduled_avg_consecutive_work_days`
- primero se calcula `consecutive_work_days` por empleado
- esa racha aumenta si el empleado trabajo hoy y tambien habia trabajado el dia anterior
- si no trabajo hoy, la racha vuelve a cero
- despues se promedia entre las personas programadas en esa franja

Construccion conceptual:

```text
scheduled_avg_consecutive_work_days
= promedio(consecutive_work_days del staff programado)
```

`Horas extra programadas`

- nombre tecnico: `scheduled_overtime_hours`
- primero se simula `overtime_hours` por empleado y franja
- suele aumentar en la noche o cuando la presion de demanda es alta
- despues se suman las horas extra de todos los programados

Construccion conceptual:

```text
scheduled_overtime_hours = suma(overtime_hours del staff programado)
```

`Dotacion requerida proyectada`

- nombre tecnico: `forecast_required_headcount_total`
- se calcula con la funcion de staffing requerido usando demanda proyectada
- depende de cubiertos, delivery, franja, festivo, fin de semana y evento local

`Dotacion real`

- nombre tecnico: `actual_headcount_total`
- parte de la programacion por franja
- luego se descuenta quien estaba ausente para obtener la dotacion efectivamente disponible

## 8.4 Factores Biometricos Agregados

Estos factores primero se generan por empleado y despues se agregan por franja.

`Horas de sueno promedio`

- nombre tecnico: `avg_sleep_duration_hours`
- por empleado se calcula a partir de una base de sueno
- baja cuando suben periodos trabajados, horas extra o bloques vespertinos
- luego se promedia sobre el equipo programado

Construccion conceptual:

```text
sleep_duration_hours = sleep_base - periods_worked*0.25 - overtime*0.40 - ruido
avg_sleep_duration_hours = promedio(sleep_duration_hours)
```

`Eficiencia de sueno promedio`

- nombre tecnico: `avg_sleep_efficiency_pct`
- por empleado cae cuando sube el ajuste de estres operativo
- luego se promedia por franja

`Estres promedio del equipo`

- nombre tecnico: `avg_stress_score`
- por empleado depende de:
  - periodos trabajados
  - horas extra
  - dias consecutivos
  - presion de demanda
  - trabajo en bloques de la tarde/noche
  - enfermedad o incubacion
- luego se promedia por franja

`Fatiga promedio`

- nombre tecnico: `avg_fatigue_proxy`
- por empleado se construye mezclando:
  - peor eficiencia de sueno
  - mayor estres
  - exceso de dias consecutivos
- luego se promedia por franja

Construccion conceptual:

```text
fatigue_proxy = clip(
    (100 - sleep_efficiency_pct)*0.35 + stress_score*0.45 + exceso_dias*4,
    0,
    100,
)
avg_fatigue_proxy = promedio(fatigue_proxy)
```

`Promedio de steps`

- nombre tecnico: `avg_steps`
- por empleado depende del rol, cantidad de periodos trabajados y estado de salud
- funciona como proxy de carga fisica y movimiento real
- luego se promedia por franja

`Carga fisica promedio del equipo`

- nombre tecnico: `avg_bmi`
- el BMI se genera una vez por empleado en la tabla de empleados
- luego se promedia entre quienes estan programados en la franja
- en la UI se renombra para que sea mas interpretable en lenguaje de negocio

`Edad promedio del equipo`

- nombre tecnico: `avg_age`
- es el promedio de la edad de los empleados programados en la franja

## 8.5 Factores De Ausentismo

`Ausentismo`

- nombre tecnico: `absentee_rate`
- se calcula como:

```text
absentee_rate = absent_count_total / scheduled_headcount_total
```

`Ausentismo de corto aviso`

- nombre tecnico: `short_notice_absentee_rate`
- se calcula como:

```text
short_notice_absentee_rate = short_notice_absent_count / scheduled_headcount_total
```

Estos dos factores no son simulados directamente en el grafico, sino construidos en el ETL a partir de ausencias reales detectadas para esa franja.

## 8.6 Factores Con Memoria Temporal

Los factores con sufijo `lag1` o `lag7` se crean al final del ETL.

`Pedidos delivery vs periodo comparable anterior`

- nombre tecnico: `delivery_order_volume_lag1`
- toma el valor de `delivery_order_volume` en la observacion anterior dentro de la misma franja

`Cubiertos del turno comparable anterior`

- nombre tecnico: `forecast_covers_lag1`
- toma el valor anterior de `forecast_covers` dentro de la misma franja

`Cubiertos de la semana anterior`

- nombre tecnico: `forecast_covers_lag7`
- toma el valor de siete observaciones antes dentro de la misma franja

`Deficit del turno comparable anterior`

- nombre tecnico: `deficit_count_total_lag1`
- resume si el problema ya venia ocurriendo en el periodo comparable previo

Regla general:

```text
feature_lag1 = feature.shift(1) dentro de la misma franja
feature_lag7 = feature.shift(7) dentro de la misma franja
```

## 8.7 Como Eso Llega Al Grafico

Una vez creado `restaurant_ml_features.csv`, el modelo:

1. toma esas columnas como features
2. predice riesgo de deficit o deficit por rol
3. calcula SHAP sobre esas features
4. promedia la magnitud e impacto de cada una

Por eso en el grafico final no estas viendo datos crudos, sino:

- la importancia promedio de cada variable para el modelo
- y la direccion promedio con que esa variable empuja el riesgo

---

## 9. Como Contarle Esto Al Dueño Del Restaurant

Una lectura simple y accionable seria:

1. este grafico muestra que factores hacen mas probable quedar cortos
2. este otro muestra en que franja pegan mas fuerte
3. este grafico por rol muestra por que `garzon` o cocina aparece tensionado
4. esta tabla muestra ejemplos concretos de turnos sensibles

La decision esperada no es "creerle al modelo porque si", sino:

- reforzar una franja puntual
- mover personal cross-trained
- anticipar reemplazos
- limitar sobrecarga de dias consecutivos
- revisar reservas, delivery y walk-ins como drivers de staffing

---

## 10. Limitaciones De Interpretacion

- SHAP no significa causalidad directa
- una variable puede aparecer importante porque resume varias tensiones operativas
- factores agregados como `fatiga promedio` o `carga fisica promedio del equipo` no deben usarse para juzgar individuos
- la explicacion es tan buena como el dato y el dominio para el que fue entrenado el modelo

---

## 11. Resumen Ejecutivo

Si quieres leer la UI rapido:

1. narrativa explicativa: donde mirar primero
2. riesgo de deficit: que esta empujando el problema
3. dotacion disponible: que sostiene o debilita la capacidad real
4. heatmap: donde pega mas
5. grafico por rol o scatter: por que sucede y desde que umbral
6. tabla final: ejemplos concretos

Con eso puedes responder cinco preguntas clave:

- donde me voy a quedar corto
- por que me voy a quedar corto
- que rol esta mas expuesto
- que factor pesa mas en cada franja
- que accion deberia tomar primero
