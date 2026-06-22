# SkillUp: Plataforma de Predicción de Dotación Laboral

## ¿Qué es SkillUp?

SkillUp es un sistema que ayuda a predecir **cuántas personas necesitará una operación** y **cuál es el riesgo de que falte personal** en un día y horario determinados. Funciona para tres tipos de negocio distintos: una planta industrial, un restaurante y una clínica ambulatoria.

Imagina que eres el gerente de un restaurante. Hoy tienes 60 reservas para el almuerzo, es viernes, hay un evento en el centro y tres de tus garzones llevan 12 días seguidos trabajando. ¿Tienes suficiente gente? ¿Deberías llamar a alguien de refuerzo? ¿Qué probabilidad hay de que justo hoy falle la cocina?

SkillUp responde ese tipo de preguntas combinando datos del personal, su estado físico, la demanda esperada y el historial reciente de la operación.

---

## El Problema que Resuelve

Toda operación que depende de personas enfrenta el mismo dilema:

- Si hay **demasiada gente**, se pagan sueldos innecesarios.
- Si hay **muy poca gente**, la operación se degrada: clientes esperando, pacientes sin atender, producción detenida, accidentes, multas.
- Y el problema es más complejo porque **la disponibilidad real del personal no es la misma que la programada**: la gente se enferma, se cansa, falta por razones personales, y esos factores cambian día a día.

Lo que SkillUp hace es construir una **simulación completa de la operación** para luego entrenar modelos matemáticos que aprenden patrones y pueden anticipar:

1. **Cuántas personas van a estar realmente disponibles** (dotación predicha).
2. **Qué probabilidad hay de que falte personal** (riesgo de déficit).
3. **Qué factores están empujando ese riesgo** (explicación de causas).
4. **Qué rol específico está más expuesto** (garzón, cocinero, médico, enfermera, etc.).

---

## Cómo Funciona el Pipeline: De Principio a Fin

El sistema opera en **5 etapas consecutivas**. Cada etapa produce algo que la siguiente etapa consume.

### Etapa 1: Generación de Datos Simulados (`generate`)

Como el sistema no tiene acceso a datos reales de una empresa específica, **genera datos sintéticos** que simulan una operación verosímil. Piensa en esto como un simulador de vuelo: no es un avión real, pero las reglas de funcionamiento son las mismas.

Esta etapa crea, para cada dominio, los siguientes archivos:

#### Para todos los dominios: Empleados sintéticos

Se generan perfiles de personas con características como:
- Edad, género, índice de masa corporal (BMI).
- Cargo o rol (operario de planta, garzón, médico, enfermera, etc.).
- Antigüedad en la empresa, distancia al trabajo, cantidad de hijos.
- Hábitos como tabaquismo o consumo de alcohol social.
- Tipo de contrato (tiempo completo, medio tiempo, solo fines de semana).
- Patrón de turno o preferencia horaria.

Cada empleado recibe un perfil distinto, con distribuciones estadísticas que intentan parecerse a la realidad (ej. más hombres que mujeres en planta industrial, edades que van de 22 a 62 años con promedio de 38, etc.).

**Código relevante:** `src/generators/employees.py` 

#### Calendario operativo

Se construye un calendario con feriados chilenos, fines de semana, vísperas de feriado, temporada del año y ventanas de pago. Esto es importante porque el comportamiento del personal y la demanda cambian drásticamente en días especiales.

#### Registros de trabajo

Para cada empleado y cada día, se determina:
- Si trabajó o descansó (siguiendo ciclos semanales).
- En qué turno trabajó (diurno, vespertino, nocturno en industrial; morning, evening en clínica; franjas de servicio en restaurante).
- Cuántas horas trabajó, si hizo horas extra.
- Cuántos días seguidos lleva trabajando.
- Qué nivel de carga operativa tuvo ese día.

**Código relevante:** `src/generators/work_records.py`

#### Datos biométricos

Aquí está uno de los aspectos más innovadores del sistema. Para cada empleado y cada día se simulan señales fisiológicas como si vinieran de un reloj inteligente o sensor corporal:

- **Frecuencia cardíaca** (HR): pulsaciones por minuto, con valores mínimos, máximos y promedio.
- **Variabilidad cardíaca** (HRV): qué tan variable es el ritmo del corazón. Valores bajos indican estrés o mala recuperación.
- **Saturación de oxígeno** (SpO2): porcentaje de oxígeno en sangre.
- **Temperatura de la piel**: indicador de calor ambiental o fiebre.
- **Horas de sueño y eficiencia del sueño**: no solo cuántas horas durmió, sino qué porcentaje fue sueño profundo.
- **Estrés fisiológico**: un puntaje que resume la carga del organismo.
- **Pasos diarios**: proxy de movimiento físico.

Estos valores no son aleatorios. Siguen reglas plausibles:

- Una persona con turno nocturno tiende a dormir peor y tener más estrés.
- Trabajar en un área con calor (como cracking en una refinería) eleva la frecuencia cardíaca y la temperatura de la piel.
- Una persona que está incubando una enfermedad (2-3 días antes de ausentarse) muestra señales degradadas: más pulsaciones, menos saturación de oxígeno, peor sueño.
- Una persona fumadora parte con una saturación de oxígeno más baja.
- El BMI alto eleva la frecuencia cardíaca base y reduce la variabilidad cardíaca.
- Los valores tienen autocorrelación: lo que pasó ayer influye en lo que pasa hoy (el cuerpo no resetea cada 24 horas).

Además, el generador **pre-cocina eventos de enfermedad** para cada empleado: determina de antemano en qué días se enfermará, con qué enfermedad (gastrointestinal, respiratoria, fatiga, musculoesquelética, estrés), cuánto durará y cuántos días antes empezará la incubación. Esto crea una cadena causal coherente:

```
señales fisiológicas empeoran → días de incubación → enfermedad → ausencia laboral
```

**Código relevante:** `src/generators/biometrics.py`

#### Ausentismo

El ausentismo **no es aleatorio**. Se activa principalmente cuando los datos biométricos ya indican que la persona está enferma. También existe una probabilidad muy baja (0.3%) de ausencia no médica (trámites, permisos, problemas personales), para que el sistema no sea completamente determinista.

Cada ausencia registra:
- Qué empleado faltó, qué día.
- El motivo (enfermedad, otros).
- Cuántas horas se perdieron.

**Código relevante:** `src/generators/absenteeism.py`

#### Datos de demanda (restaurant y clínica)

En industrial la dotación requerida es fija por área y turno (por seguridad operacional, una planta de cracking siempre necesita cierto mínimo de personas, sin importar la demanda diaria). Pero en restaurant y clínica la demanda sí varía:

**Restaurant** genera:
- Cubiertos proyectados y reales (clientes en sala).
- Reservas.
- Pedidos de delivery.
- Proporción de clientes sin reserva (walk-ins).
- Ventas proyectadas y reales.
- Banderas de promociones activas, eventos locales y condiciones climáticas que afectan la afluencia.

**Clínica** genera:
- Volumen de pacientes por unidad y turno.
- Casos de alta complejidad.
- Procedimientos ambulatorios programados.
- Boxes o estaciones de atención activas.
- Rezago de imagenología.
- Casos de alerta respiratoria.
- Tiempos de espera y porcentaje de inasistencias.
- Multiplicadores de contexto: campaña de vacunación, ventana de pago, presión de bloque electivo.

**Código relevante:** `src/restaurant/generate.py`, `src/clinic/generate.py`

---

### Etapa 2: Transformación y Preparación de Datos (`etl`)

Los datos crudos generados en la Etapa 1 son como ingredientes sin procesar. Esta etapa los limpia, los combina y los convierte en una **tabla de análisis** lista para que los modelos aprendan de ella.

**ETL** significa Extract, Transform, Load (Extraer, Transformar, Cargar).

#### Paso a paso de la transformación

1. **Limpieza de datos biométricos**: se eliminan valores fisiológicamente imposibles (ej. frecuencia cardíaca de 300 bpm, saturación de oxígeno de 120%). También se descartan registros de sensores de mala calidad.

2. **Imputación de valores faltantes**: cuando un sensor falla o un registro tiene mala calidad, el valor se estima usando el historial del mismo empleado. Para huecos pequeños (1-2 días) se copia el último valor conocido. Para huecos más largos se usa el promedio histórico.

3. **Normalización individual**: cada persona tiene su propio "baseline" fisiológico. Una frecuencia cardíaca de 80 bpm puede ser normal para alguien pero elevada para otra persona. El sistema normaliza los valores respecto al historial de cada empleado, para que las comparaciones tengan sentido.

4. **Agregación por unidad de decisión**: los datos individuales se resumen al nivel donde se toma la decisión de negocio:

   - **Industrial**: por `área de planta + turno + día`. Ejemplo: "área de destilación, turno nocturno, día 15 de marzo".
   - **Restaurant**: por `fecha + franja horaria`. Ejemplo: "viernes 15 de marzo, franja 13:00-15:00 (almuerzo peak)".
   - **Clínica**: por `fecha + turno + unidad clínica`. Ejemplo: "lunes 18 de marzo, turno mañana, unidad de imagenología".

5. **Creación de variables con memoria temporal (lags)**: se añaden columnas que miran hacia atrás:
   - `lag1`: qué pasó en el período inmediatamente anterior comparable (ej. el almuerzo de ayer).
   - `lag7`: qué pasó hace exactamente una semana (ej. el almuerzo del viernes pasado).
   
   Esto permite que el modelo aprenda patrones como "cuando el almuerzo del viernes pasado estuvo saturado, este viernes suele repetirse".

6. **Cálculo de la dotación requerida**: aplicando las reglas de negocio de cada dominio, se calcula cuántas personas se necesitan para operar correctamente:
   - **Industrial**: tabla fija por área y turno (ej. cracking nocturno = 8 personas mínimo).
   - **Restaurant**: fórmulas como "1 garzón por cada 22 cubiertos", "1 cocinero de línea por cada 30 cubiertos+delivery".
   - **Clínica**: reglas por unidad y turno según volumen de pacientes, procedimientos, boxes activos y tipo de atención.

7. **Cálculo de targets (lo que el modelo debe aprender a predecir)**:
   - `actual_headcount`: cuántas personas efectivamente estuvieron disponibles.
   - `has_deficit`: ¿faltó personal? (1 = sí, 0 = no).
   - `deficit_count`: ¿cuántas personas faltaron?

Toda esta información se guarda en una **base de datos SQLite** (una base de datos ligera que funciona como un archivo). Esto garantiza que todas las etapas siguientes lean exactamente los mismos datos.

**Código relevante:** `src/etl/transform.py`, `src/restaurant/etl.py`, `src/clinic/etl.py`

---

### Etapa 3: Entrenamiento de Modelos (`train`)

Una vez que existe la tabla de análisis, se entrenan modelos matemáticos para que aprendan a predecir. Un modelo es como un alumno: le muestras muchos ejemplos de "antecedentes → resultado" y él aprende los patrones.

#### División temporal de los datos

Los datos se dividen en tres partes:

```
├── Entrenamiento (64%): el modelo aprende de estos datos
├── Validación    (16%): se usa para elegir el mejor modelo
└── Test ciego    (20%): se usa una sola vez al final para verificar
```

**La división es temporal**, no aleatoria. Los datos más antiguos van a entrenamiento, los siguientes a validación y los más recientes a test. Esto es crítico porque en la vida real el modelo se entrena con el pasado y debe predecir el futuro, no puede "ver el futuro" durante el entrenamiento.

#### Tipos de modelos que se entrenan

**Modelo 1: Regresor de dotación disponible**

Un "regresor" predice un número continuo. En este caso, predice cuántas personas van a estar realmente disponibles. Responde a la pregunta: "¿Con cuánta gente voy a contar?"

Se usa **Random Forest**, que es como un comité de árboles de decisión. Cada árbol mira los datos desde un ángulo distinto y todos votan para dar una predicción final. Es robusto, difícil de engañar y funciona bien con datos tabulares.

**Modelo 2: Clasificador de déficit (XGBoost)**

Un "clasificador" predice una categoría o probabilidad. En este caso, predice si habrá déficit de personal o no, y con qué probabilidad. Responde: "¿Qué tan probable es que me falte gente?"

Se usa **XGBoost**, un algoritmo que aprende de sus propios errores. Primero hace una predicción burda, luego mira en qué se equivocó, construye un segundo modelo que corrige esos errores, luego un tercero que corrige los del segundo, y así sucesivamente. Es uno de los algoritmos más usados en la industria por su precisión y velocidad.

**Modelo 3: Clasificador Calibrado**

Toma el mismo XGBoost del modelo 2 pero lo envuelve en una capa adicional que ajusta las probabilidades para que sean más confiables. Por ejemplo, si el modelo dice "70% de probabilidad de déficit", queremos que aproximadamente 7 de cada 10 veces que diga 70% realmente haya déficit. Sin calibrar, a veces los modelos son "muy seguros" o "muy inseguros" en sus predicciones.

**Modelos por rol crítico (restaurant y clínica)**

Además de predecir el déficit general, se entrena un clasificador por cada rol crítico:

- **Restaurant**: garzón, cocinero de línea, jefe de turno.
- **Clínica**: médico, enfermera, TENS.

Esto responde preguntas más finas como "no es que falte gente en general, es que específicamente van a faltar garzones en el almuerzo".

#### Selección del mejor modelo

El modelo 2 y el modelo 3 compiten. Se evalúan ambos sobre los datos de **validación** (que ninguno ha visto durante entrenamiento) y se elige el que tenga mejor desempeño. Luego **ese ganador se vuelve a entrenar** con los datos de entrenamiento + validación juntos, y se evalúa una sola vez sobre el test ciego.

**Métricas que se usan para comparar:**

- **AUC-ROC**: qué tan bien distingue el modelo entre situaciones con déficit y sin déficit. Un valor de 0.5 es aleatorio, 1.0 es perfecto.
- **Precision-Recall**: qué tan precisas son las alertas de déficit (no queremos demasiadas falsas alarmas).
- **Brier Score**: qué tan calibradas están las probabilidades.
- **F1-Score**: balance entre precisión y cobertura.

Todo esto se guarda en archivos para trazabilidad: los modelos entrenados (archivos `.pkl`), sus métricas, y los umbrales de decisión.

**Código relevante:** `src/models/staffing_models.py`, `src/restaurant/train.py`, `src/clinic/train.py`

---

### Etapa 4: Inferencia y Predicción (`infer`)

Con los modelos ya entrenados, esta etapa genera predicciones para **todos los datos disponibles** y calcula métricas finales solo sobre el período de test (que los modelos nunca vieron antes).

Esta etapa produce:

1. **Archivo de predicciones**: una tabla donde cada fila (un área-turno-día, o franja, o unidad-turno-día) tiene:
   - `predicted_headcount`: dotación predicha (cuánta gente cree el modelo que habrá).
   - `predicted_deficit_probability`: probabilidad de que haya déficit (0% a 100%).
   - `predicted_has_deficit`: predicción final (SÍ o NO) según un umbral.
   - `role_deficit_prob_garzon`, `role_deficit_prob_cocinero_linea`, etc.: probabilidad de déficit por rol específico.
   - **Recomendación operativa textual**: una sugerencia en lenguaje natural como "Reforzar garzón: riesgo 81%. Considere activar backup de runner" o "Dotación estimada 18 personas vs 22 requeridas. Déficit probable. Considere retenes."

2. **Archivo de métricas**: precisión del modelo, tasa de acierto, falsas alarmas, etc.

3. **Análisis SHAP**: explicaciones de por qué el modelo predice lo que predice (ver siguiente sección).

**Código relevante:** `src/models/inference.py`, `src/restaurant/inference.py`, `src/clinic/inference.py`

---

### Etapa 5: Reportes y Dashboard (`report`)

Esta etapa toma los resultados de la inferencia y los convierte en visualizaciones accionables:

- **Dashboard ejecutivo**: un gráfico de una sola página con los indicadores más importantes (dotación requerida vs real vs predicha, riesgo por franja/turno, déficit por rol).
- **Gráficos SHAP interactivos**: para explorar qué factores están empujando el riesgo (ver siguiente sección).
- **Tabla de instancias críticas**: los casos con mayor riesgo, explicados uno por uno.

Toda esta visualización también está disponible en una **aplicación web interactiva** construida con Streamlit, donde el usuario puede:
- Seleccionar el dominio (industrial, restaurant, clínica).
- Ver KPIs ejecutivos.
- Explorar predicciones por fecha, turno, área o unidad.
- Analizar gráficos SHAP con filtros y drill-down por rol.
- Ejecutar pipelines completos o por etapa.

**Código relevante:** `src/ui/dashboard_app.py`, `src/restaurant/reporting.py`, `src/clinic/reporting.py`

---

## Cómo se Explica el Porqué de Cada Predicción (SHAP)

Uno de los mayores problemas de la inteligencia artificial es la "caja negra": el modelo te dice "va a haber déficit con 85% de probabilidad", pero no te dice por qué.

**SHAP** (SHapley Additive exPlanations) resuelve esto. Proviene de la teoría de juegos y responde: de todos los factores que el modelo conoce, ¿cuánto contribuyó cada uno a esta predicción en particular?

### Qué significan los gráficos SHAP

Imagina que el modelo predice que el viernes a las 13:00 hay 81% de riesgo de déficit de garzones. SHAP descompone ese 81% en contribuciones:

- "Hay 120 cubiertos proyectados" → **+22%** de riesgo (empuja hacia arriba).
- "El equipo tiene fatiga promedio alta (68/100)" → **+15%** de riesgo.
- "Es viernes y hay evento local" → **+10%** de riesgo.
- "El ausentismo de corto aviso está bajo (2%)" → **-5%** de riesgo (compensa).

Así, el gerente no solo sabe que hay riesgo, sino que puede actuar sobre las causas.

### Tipos de gráficos SHAP disponibles

1. **Barras de importancia global**: ¿qué factores pesan más en promedio? Responde: "¿qué es lo que más está empujando el déficit en general?"

2. **Mapa de calor por segmento**: ¿en qué turno, franja o unidad impacta más cada factor? Responde: "el estrés del equipo pega más fuerte en la franja de cena que en la de almuerzo".

3. **Gráfico de dependencia**: ¿a partir de qué valor un factor se vuelve peligroso? Responde: "cuando la fatiga promedio pasa de 60, la probabilidad de déficit se dispara".

4. **Explicación por rol**: ¿por qué específicamente va a faltar un garzón y no un cocinero? Responde: "el déficit de garzón lo empuja la cantidad de cubiertos; el de cocinero lo empuja más el delivery".

**Código relevante:** `src/models/shap_analysis.py`, `docs/guia_lectura_graficos_shap.md`

---

## Las Tres Dotaciones: Una Distinción Fundamental

Para entender el sistema es esencial distinguir tres conceptos:

| Concepto | Qué es | Quién lo determina | Pregunta que responde |
|---|---|---|---|
| **Dotación requerida** | Gente que necesito | Reglas de negocio fijas | "¿Cuántas personas debería tener?" |
| **Dotación actual** | Gente que realmente tuve | Dato observado/simulado | "¿Con cuántas personas terminé operando?" |
| **Dotación predicha** | Gente que estimo que tendré | Modelo de regresión | "¿Cuántas personas voy a tener disponibles?" |

La regla práctica más importante:

```
Si dotación predicha < dotación requerida → existe riesgo de déficit
```

---

## Valor para Cada Tipo de Empresa

### Planta Industrial

#### El dolor de negocio que ataca

En una planta industrial (refinería, manufactura pesada, procesamiento), el déficit de personal no solo afecta la productividad: puede ser un **riesgo de seguridad**. Ciertas áreas tienen un mínimo operativo por ley o por protocolo. Si no se alcanza ese mínimo, la operación debe detenerse o ralentizarse.

Además, el personal industrial está expuesto a **desgaste físico acumulativo**: turnos nocturnos, calor extremo, trabajo físico repetitivo. Las señales de fatiga y estrés se acumulan en silencio hasta que explotan en forma de ausentismo, accidentes o errores.

#### Cómo SkillUp ayuda

1. **Predicción de dotación disponible**: el regresor estima cuánta gente va a estar realmente en cada área y turno, considerando el desgaste fisiológico acumulado. No es lo mismo "tenemos 12 personas programadas" que "el modelo estima que solo 9 estarán en condiciones de operar".

2. **Alerta temprana de déficit**: el clasificador avisa con anticipación qué áreas y turnos tienen alta probabilidad de no alcanzar el mínimo operativo.

3. **Visibilidad del desgaste del equipo**: los gráficos SHAP muestran si el problema viene del ausentismo, del estrés acumulado, de la falta de sueño en turnos nocturnos o del calor en ciertas áreas.

4. **Decisiones operativas que habilita**:
   - Reforzar turnos específicos con personal de otras áreas (cross-training).
   - Ajustar rotaciones para reducir días consecutivos en áreas de alto riesgo.
   - Activar protocolos de retén antes de que el déficit ocurra.
   - Identificar áreas donde la exposición al calor está degradando sistemáticamente la disponibilidad.

#### Supuestos específicos del dominio industrial

- La unidad de decisión es `área + turno + día`. Áreas modeladas: destilación, cracking, almacenamiento, mantenimiento, oficinas. Turnos: diurno, vespertino, nocturno.
- La dotación requerida está fija por tabla de negocio (mínimo operativo por seguridad y continuidad), no depende de demanda diaria.
- Cada área tiene un perfil de riesgo fijo (cracking es inherentemente más exigente que oficinas) y algunas tienen exposición a calor.
- El ausentismo está fuertemente ligado a salud y carga operativa, no a causas administrativas complejas.
- **No modela**: cambios diarios de producción, paradas de planta, eventos sindicales, matrices de habilidades detalladas.

### Restaurante (Casual Dining)

#### El dolor de negocio que ataca

Un restaurante de servicio completo vive de la experiencia del cliente. Si un garzón falta en el peak del almuerzo, los tiempos de espera se alargan, los platos salen fríos, las mesas rotan más lento y el cliente no vuelve.

El problema es particularmente agudo porque:

- La demanda **cambia drásticamente por franja horaria**: no es lo mismo las 11:00 que las 13:00.
- Hay **factores externos impredecibles**: un evento en el centro, un día de lluvia que tira todo a delivery, un feriado que triplica las reservas.
- El personal de restaurante tiene **alta rotación**, mezcla de contratos (full-time, part-time, weekend-only) y desgaste físico real (horas de pie, pasos, estrés de servicio).
- Los **roles no son intercambiables libremente**: un runner puede apoyar a garzón hasta cierto punto, pero no puede reemplazar a un cocinero de línea.

#### Cómo SkillUp ayuda

1. **Predicción por franja horaria**: el sistema entiende que el almuerzo del viernes es distinto a la cena del martes. Cada franja (`11-13`, `13-15`, `19-21`, `21-23`) se modela por separado.

2. **Riesgo por rol crítico**: no solo dice "va a faltar gente", dice "específicamente van a faltar garzones, aunque cocina está cubierta". Esto permite acciones precisas: mover un runner a apoyo de salón, activar un garzón de retén, etc.

3. **Sensibilidad a factores de demanda**: el modelo incorpora reservas, walk-ins, delivery, clima, eventos locales, promociones activas, festivos y estacionalidad. Aprende cómo cada uno afecta la necesidad de personal.

4. **Monitoreo fisiológico del equipo**: fatiga promedio, horas de sueño, pasos diarios, estrés. Una cocina agotada un viernes a las 21:00 es una receta para errores y accidentes.

5. **Dashboard ejecutivo**: una vista de una página para el gerente de turno, con alertas de riesgo, recomendaciones de refuerzo y comparación contra lo que realmente pasó.

6. **Decisiones operativas que habilita**:
   - "Reforzar garzón en almuerzo del viernes: riesgo 81%. Reservas altas + evento local. Considere activar backup de runner."
   - "Cocina tensionada en cena del sábado: fatiga acumulada + delivery alto. Evaluar cocinero de retén."
   - "Semana Santa: todas las franjas con riesgo elevado. Revisar dotación completa."

#### Supuestos específicos del dominio restaurant

- Es un restaurante "casual dining" chileno: servicio completo, ticket medio, atención en salón, con delivery, sensible a reservas y walk-ins.
- La unidad de decisión es `fecha + franja horaria`, no turno contractual. Las franjas modeladas son 11-13, 13-15, 19-21, 21-23. La franja 15:00-19:00 no es una unidad analítica propia (no significa que el local esté cerrado, pero no se modela por separado).
- La dotación requerida se calcula con reglas de negocio basadas en cubiertos, delivery y contexto (ej. 1 garzón cada 22 cubiertos).
- Los roles críticos son garzón, cocinero de línea y jefe de turno. Los backups entre roles son discretos, no libres (un runner puede ayudar a garzón, un ayudante puede apoyar a cocina, pero nadie reemplaza a un jefe de turno).
- Se modelan eventos de salud sintéticos: gastrointestinal, respiratorio, fatiga, musculoesquelético, estrés.
- **No modela**: layout de salón por mesa, tiempos de cocción, complejidad de platos, política de propinas, secuencias intra-franja minuto a minuto.

### Clínica Ambulatoria

#### El dolor de negocio que ataca

Una clínica ambulatoria (consultas, especialidades, procedimientos menores, imagenología, toma de muestras) enfrenta un problema de coordinación complejo:

- **Distintas unidades tienen necesidades distintas**: consulta general depende del flujo de pacientes; procedimientos ambulatorios depende de la agenda; imagenología sufre con el backlog acumulado; toma de muestras enfrenta demanda espontánea alta.
- **Los roles clínicos no son intercambiables**: un TENS no puede reemplazar a un médico. Una enfermera puede flotar entre unidades pero no cubrir todas las funciones.
- **La carga cognitiva es real**: un equipo médico con fatiga mental acumulada comete errores, omite detalles, demora más. En una clínica eso significa diagnósticos apurados, esperas más largas y riesgo para el paciente.
- **Hay multiplicadores de presión**: campaña de vacunación que dispara la demanda, alerta respiratoria que congestiona consulta general, ventana de pago que llena toma de muestras.

#### Cómo SkillUp ayuda

1. **Predicción por unidad y turno**: el sistema modela cada unidad clínica por separado (consulta general, especialidades, procedimientos ambulatorios, imagenología, toma de muestras) en turno mañana y tarde. Entiende que las 8:00 en toma de muestras es distinto a las 14:00 en imagenología.

2. **Riesgo por rol crítico**: anticipa si va a faltar médico, enfermera o TENS, que son los tres perfiles que sostienen la continuidad clínica ambulatoria.

3. **Sensibilidad a factores de carga asistencial**: volumen de pacientes, casos de alta complejidad, procedimientos programados, boxes activos, backlog de imagenología, casos respiratorios, tiempo de espera, porcentaje de inasistencias. El modelo aprende cómo cada variable afecta la dotación necesaria.

4. **Variables de carga cognitiva**: el sistema modela `cognitive_load_score` (presión mental acumulada) y `reaction_time_ms` (enlentecimiento por fatiga), capturando el desgaste que no se ve pero que impacta la calidad de atención.

5. **Flexibilidad operacional modelada**: cada empleado tiene una `primary_unit` (donde trabaja normalmente) y una `backup_unit` (donde puede apoyar). Algunos tienen `can_float=True` (pueden desplazarse entre unidades). El modelo entiende esta red de respaldos y la usa en sus predicciones.

6. **Decisiones operativas que habilita**:
   - "Turno mañana en imagenología: déficit probable de tecnólogo médico. Backlog acumulado alto. Considere derivar pacientes a turno tarde o activar flotante."
   - "Toma de muestras colapsada: alerta respiratoria eleva volumen 40%. Evaluar TENS de refuerzo desde procedimientos ambulatorios."
   - "Consulta general del lunes: médico con alta carga cognitiva acumulada. Riesgo de error y demora. Considere redistribuir agenda."

#### Supuestos específicos del dominio clínica

- La unidad de decisión es `fecha + turno + unidad clínica`. Turnos: morning, evening. Unidades: consulta general, especialidades, procedimientos ambulatorios, imagenología, toma de muestras.
- Es una clínica ambulatoria, no hospitalaria. No modela hospitalización, urgencias 24/7 ni pabellón mayor.
- La dotación requerida se calcula con reglas por unidad y rol, considerando volumen de pacientes, alta complejidad, procedimientos y boxes activos.
- Los roles críticos son médico, enfermera y TENS.
- La demanda se resume con baselines por unidad y turno, ajustados por multiplicadores de contexto (día de semana, feriado, campaña de vacunación, ventana de pago, bloque electivo, alerta respiratoria).
- El ausentismo depende de estrés, fatiga, sueño, carga cognitiva e infección respiratoria.
- **No modela**: agenda por médico individual, derivación entre clínicas, reglas contractuales detalladas por gremio, hospitalización prolongada.

---

## Flujo Documental de Fichas Médicas (PDF)

Además del pipeline principal, el sistema incluye una capacidad complementaria de **digitalización de fichas médicas**:

1. **Genera** fichas médicas laborales en formato PDF a partir de los datos de empleados.
2. **Extrae** el contenido de esos PDFs y lo convierte de vuelta a datos estructurados (CSV).
3. **Valida** que la extracción sea exactamente igual al archivo original (prueba de round-trip).
4. **Integra** los datos extraídos al pipeline de ETL y entrenamiento.

Esto es relevante para empresas que manejan fichas médicas en papel y necesitan digitalizarlas para alimentar el sistema de predicción.

**Código relevante:** `src/generators/medical_forms.py`, `src/extraction/medical_forms.py`

---

## Arquitectura Multi-Dominio

El sistema está diseñado como una plataforma que puede expandirse a nuevos tipos de negocio sin reescribir el núcleo:

```
Runner Común (src/domains/)
├── Dominio Industrial   (src/domains/industrial.py)
├── Dominio Restaurant   (src/domains/restaurant.py)
├── Dominio Clinic       (src/domains/clinic.py)
└── [Futuro dominio X]  (src/domains/x.py)
```

Cada dominio implementa las mismas 5 etapas (`generate`, `etl`, `train`, `infer`, `report`), pero con reglas de negocio, generadores y modelos propios. El runner común (`src/domains/base.py`, `src/domains/registry.py`) se encarga de orquestar la ejecución sin saber los detalles de cada dominio.

Esto significa que si mañana se quisiera agregar un dominio de **retail**, **logística** o **hotelería**, solo habría que:
1. Crear un archivo de configuración (`config/retail_settings.py`).
2. Escribir generadores de datos propios del dominio.
3. Escribir ETL, entrenamiento e inferencia propios.
4. Registrar el dominio en `src/domains/registry.py`.

El resto del sistema (runner, UI, persistencia, SHAP, visualización) funciona sin cambios.

---

## Cómo se Ejecuta

### Usando Docker (recomendado para empezar)

```bash
make up                    # Levanta la interfaz web en http://localhost:8501
make pipeline DOMAIN=restaurant STAGE=full ARGS="--employees 72 --days 180 --seed 42"
make pipeline DOMAIN=industrial STAGE=full
make pipeline DOMAIN=clinic STAGE=full ARGS="--employees 96 --days 180 --seed 42"
```

### En local (desarrollo)

```bash
uv run python scripts/run_pipeline.py --domain restaurant --stage full --employees 72 --days 180 --seed 42
uv run python scripts/run_dashboard_ui.py    # Interfaz web
```

### Ejecutar etapas individuales

```bash
uv run python scripts/run_pipeline.py --domain restaurant --stage generate
uv run python scripts/run_pipeline.py --domain restaurant --stage etl
uv run python scripts/run_pipeline.py --domain restaurant --stage train
uv run python scripts/run_pipeline.py --domain restaurant --stage infer
uv run python scripts/run_pipeline.py --domain restaurant --stage report
```

---

## Lo que el Sistema No Hace (Limitaciones Conscientes)

El proyecto está construido sobre **supuestos explícitos y simplificaciones deliberadas**. Es importante entender qué no cubre:

1. **No usa datos reales de una empresa específica.** Los datos son sintéticos, generados con reglas plausibles pero no calibrados contra una operación real. Para usarlo en producción, habría que alimentarlo con datos reales y recalibrar generadores, reglas de negocio y modelos.

2. **No modela causalidad real.** SHAP explica cómo decide el modelo, no demuestra que un factor cause el déficit en el mundo real. Si SHAP dice que "fatiga promedio" es el factor más importante, significa que el modelo lo usa mucho para predecir, no necesariamente que eliminar la fatiga resolvería el problema.

3. **No captura todas las variables del negocio.** Por diseño, se simplifican ciertos aspectos para que el sistema sea interpretable y entrenable. Por ejemplo, no modela paradas de planta en industrial, no modela la secuencia minuto a minuto de un servicio en restaurant, no modela agenda por médico individual en clínica.

4. **Las reglas de dotación requerida son fijas.** No se aprenden de datos históricos sino que se codifican manualmente (ej. "1 garzón cada 22 cubiertos"). En una implementación real, estas reglas deberían calibrarse con datos de la operación.

5. **No es un sistema en tiempo real.** Las predicciones se generan por lotes (batch), no se actualizan automáticamente con nueva información cada minuto.

---

## Resumen: El Valor Central

SkillUp transforma datos operativos y fisiológicos en **decisiones de dotación anticipadas y explicables**. Para un dueño de restaurante, un jefe de planta o un director de clínica, el valor está en:

1. **Anticipar** en lugar de reaccionar: saber el jueves que el viernes va a faltar gente, no enterarse el viernes a las 12:00.
2. **Entender por qué**: no solo recibir una alerta, sino saber qué factor la causó y poder actuar sobre él.
3. **Priorizar acciones**: si hay riesgo en tres turnos, saber cuál es más urgente y qué rol específico reforzar.
4. **Monitorear el desgaste invisible**: ver que el equipo viene acumulando fatiga antes de que explote en ausentismo.
5. **Tomar decisiones basadas en datos**, no en intuición o en "cómo nos fue la semana pasada".

---

## Próximos Pasos

El sistema actual es una base operativa y demostrable. A continuación se describe una hoja de ruta de acciones recomendadas para llevarlo desde su estado de simulador sintético hacia una herramienta de producción real, priorizadas por impacto.

### Fase 1: Calibración con Datos Reales de una Empresa Piloto

Este es el paso más importante y el que más valor destraba. Consiste en:

1. **Seleccionar una empresa piloto** de uno de los tres dominios (idealmente donde ya existan registros de asistencia, turnos programados, demanda diaria y algún tipo de medición de fatiga o ausentismo).

2. **Reemplazar los generadores sintéticos** por conectores a los sistemas reales de la empresa:
   - Reloj control o sistema de asistencia → `work_records.csv`.
   - Sistema de RRHH → `employees.csv`.
   - Sistema de demanda (POS en restaurant, agenda en clínica, planilla de producción en industrial) → datos de demanda.
   - Si existen wearables o sensores corporales → `biometrics.csv`. Si no existen, el sistema puede operar sin ellos, aunque pierde la riqueza de las señales fisiológicas.

3. **Recalibrar las reglas de dotación requerida** con los datos históricos de la empresa:
   - ¿Realmente 1 garzón por cada 22 cubiertos? ¿O en este local la productividad es distinta?
   - ¿El mínimo operativo del área de cracking es realmente 8 personas o depende de la carga de producción?
   - ¿Cuántos pacientes por médico se manejan realmente en consulta general?

4. **Reentrenar los modelos** con datos reales y validar sus predicciones contra lo que efectivamente ocurrió en meses posteriores.

### Fase 2: Enriquecimiento de Variables (Priorizado por Dominio)

Cada dominio tiene puntos ciegos específicos que, al incorporarse, mejorarían la precisión:

**Industrial:**
- Agregar carga de producción diaria por área (throughput, toneladas procesadas, órdenes de trabajo activas).
- Incorporar paradas de planta programadas y correctivas (afectan la disponibilidad y el estrés del equipo).
- Modelar exposición térmica real por turno y estación (temperatura ambiente, no solo bandera binaria de calor).
- Agregar matriz de habilidades y certificaciones por empleado (¿quién puede realmente cubrir a quién?).

**Restaurant:**
- Modelar la franja 15:00-19:00 como unidad analítica propia (hoy no está contemplada).
- Incorporar complejidad del menú y tiempos de cocción (no es lo mismo un plato de 8 minutos que uno de 25).
- Incluir política de propinas (afecta retención y motivación del staff).
- Agregar datos de capacitación cruzada real (¿qué empleados específicos pueden rotar entre qué roles?).

**Clínica:**
- Modelar agenda por médico individual con duraciones reales de consulta.
- Incorporar el flujo de urgencias como presión adicional sobre la dotación ambulatoria.
- Agregar derivaciones entre especialidades (un paciente puede pasar de consulta general a imagenología en la misma visita).
- Incluir reglas contractuales y gremiales por tipo de profesional.

### Fase 3: Mejoras Técnicas del Pipeline

**Modelos más sofisticados:**
- Probar redes neuronales para series temporales (LSTM, Transformers) que capturen patrones estacionales más complejos que los lags actuales.
- Implementar modelos de forecasting de demanda (predecir cuántos cubiertos/pacientes/órdenes habrá, además de la dotación).
- Modelos multi-salida que predigan dotación total y por rol simultáneamente (en lugar de modelos separados).

**Ingesta de datos en tiempo real:**
- Migrar de ejecución batch a un pipeline con ingesta diaria automática.
- Conectar a APIs de clima, calendario de eventos locales y feriados dinámicos (no hardcodeados a 2025).
- Implementar alertas automáticas (email, WhatsApp, Slack) cuando el riesgo supere un umbral configurable.

**Optimización de decisiones:**
- Agregar un módulo de **recomendación prescriptiva** que no solo prediga el déficit, sino que sugiera automáticamente:
  - A qué empleados específicos conviene reprogramar o mover de turno.
  - Cuántas horas extra conviene activar y en qué rol.
  - Si conviene contratar temporal o permanentemente según patrones crónicos de déficit.
- Incorporar costos en las recomendaciones (costo de hora extra vs. costo de déficit vs. costo de contratación).

### Fase 4: Expansión a Nuevos Dominios

La arquitectura multi-dominio ya está preparada. Dominios naturales para expandir:

1. **Retail / Tiendas por departamento**: unidad `tienda + turno + día`, con sensibilidad a temporada de compras, liquidaciones, horario extendido.

2. **Hotelería**: unidad `hotel + turno + fecha`, con demanda por ocupación, eventos, temporada turística, check-in/check-out masivos.

3. **Logística / Centros de distribución**: unidad `centro + turno + día`, con sensibilidad a volumen de pedidos, campañas (CyberDay, Navidad), fatiga por manejo de carga.

4. **Educación / Colegios**: unidad `sede + jornada + día`, con demanda por matrícula, eventos escolares, exámenes, reemplazos por licencia.

Cada nuevo dominio requiere aproximadamente un 20% de código nuevo y reutiliza el 80% del sistema existente (runner, ETL base, entrenamiento, inferencia, SHAP, UI).

### Fase 5: Gobernanza y Confianza

Para que el sistema sea adoptado por una operación real, se necesita:

- **Panel de monitoreo de calidad del modelo**: ¿las predicciones de esta semana fueron precisas? ¿El modelo se está degradando con el tiempo?
- **Bitácora de decisiones**: registrar qué recomendación hizo el sistema, qué decisión tomó el humano y qué resultado se observó (ciclo de mejora continua).
- **Reentrenamiento periódico automático**: cada N meses, reentrenar modelos con los nuevos datos acumulados y comparar contra la versión anterior.
- **Validación de equidad y sesgos**: asegurar que el modelo no discrimine por edad, género u otras variables sensibles al hacer recomendaciones de dotación.

### Criterio de Priorización

La lógica para decidir qué hacer primero debería basarse en tres preguntas:

1. **¿Tenemos los datos?** Si una mejora requiere datos que hoy no existen (ej. wearables), se pospone hasta que estén disponibles.
2. **¿El problema de negocio lo justifica?** Si al dueño del restaurante le duele más el déficit de garzones que el de coperos, priorizar garzones.
3. **¿La mejora es visible?** Una recomendación accionable por WhatsApp vale más que un modelo 2% más preciso que nadie consulta.

---

## Documentación de Referencia

- `docs/supuestos.md` y `docs/supuestos_v2.md`: Supuestos completos del sistema, diccionarios de datos y Q&A extensivo.
- `docs/arquitectura_validada.md` y `docs/pipeline_actualizado.md`: Arquitectura técnica y pipeline actualizado.
- `docs/guia_lectura_graficos_shap.md`: Guía de interpretación de gráficos SHAP con glosario por dominio.
- `docs/pipeline_completo_flujos.drawio` / `.svg`: Diagrama visual de la arquitectura multi-dominio.
- `README.md`: Guía de inicio rápido, comandos y estructura del proyecto.
