# Sistema de mantenimiento predictivo en cascada vs. clasificación multiclase - AI4I 2020

## Resumen

- Sistema de mantenimiento predictivo en dos etapas (detector binario -> diagnosticador multi-label) sobre el dataset AI4I 2020, comparado contra un clasificador multiclase plano.
- Modelo base: Gradient Boosting, seleccionado por MCC sobre Decision Tree, Random Forest y XGBoost. El diagnosticador clasifica 3 de los 4 tipos de falla (HDF, PWF, OSF) con F1 superior al 92% en evaluación aislada; features físicas derivadas (potencia mecánica, gradiente térmico, sobreesfuerzo) alimentan ambas etapas.
- **Hallazgo central:** la arquitectura ganadora depende de la métrica. La cascada iguala al mejor baseline en MCC (87.49% vs 87.64%) pero queda por debajo en F1-macro y balanced accuracy. Ningún enfoque domina; el MCC global recompensa el abandono de la clase minoritaria crítica (TWF).
- **Limitación documentada:** un modo de falla (TWF) es irrecuperable con las features disponibles. Los 12 casos de test caen enteros en los falsos negativos del detector, con probabilidad de falla predicha entre 1e-7 y 1e-3; es estocástico por diseño del dataset. Se documenta el análisis de causa raíz en lugar de ocultarlo.

## Dataset y notebook

**Notebook:** [Mantenimiento predictivo AI4I](https://www.kaggle.com/code/ramif13/mantenimiento-predictivo-ai4i-2020).

**Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).

## El problema y por qué una cascada

Planteé sobre el dataset AI4I 2020 dos preguntas distintas sobre cada máquina: ¿va a fallar?, y si falla ¿de qué tipo es la falla? Un clasificador multiclase plano las responde en un solo paso, asignando a cada máquina una de cinco etiquetas (sin falla + 4 tipos de falla).

La hipótesis de este proyecto fue que separar esas dos preguntas en dos etapas, un detector binario primero y un diagnosticador de tipo de falla después, podría rendir mejor que el enfoque plano, por tres razones:
- **Co-ocurrencia de fallas:** el 7.1% de las fallas presentan más de un tipo simultáneo (por ejemplo, sobreesfuerzo y disipación térmica a la vez). Un multiclase plano fuerza una única etiqueta y pierde esa estructura; un diagnosticador multi-label la conserva.
- **Manejo del desbalance:** las fallas son el 3.4% del dataset. Un detector binario enfrenta un desbalance de 3.4%/96.6%, mientras que un multiclase plano debe clasificar clases que representan tan solo el 0.46% (TWF) o el 0.19% (RNF) del total. Este es un problema de desbalance más extremo y multidimensional.
- **Especialización:** cada etapa optimiza una tarea más simple. El detector distingue falla de no-falla; el diagnosticador clasifica el tipo sobre casos que ya se sabe que fallaron.

El costo de esta arquitectura es la propagación de error: una falla que el detector no captura nunca llega al diagnosticador. Esta tensión entre las ventajas de la separación frente al riesgo de propagación es la que el proyecto pone a prueba comparando ambas arquitecturas sobre el mismo split.

## Decisiones metodológicas

### Arquitectura multi-label en el diagnosticador
El diagnosticador de la etapa 2 clasifica cada tipo de falla con un clasificador binario independiente, en lugar de un único multiclase. La razón es estructural: el 7.1% de las fallas presentan co-ocurrencia de tipos (24 de 339), lo que hace que una etiqueta única sea incorrecta por definición para esos casos. Cuatro clasificadores binarios independientes preservan esa estructura; un multiclase plano la destruye.

RNF (Random Failures, 19 casos, 0.19% del total) fue excluido del diagnosticador. Por diseño del dataset, RNF ocurre independientemente de las condiciones del proceso: es ruido sin señal. Incluirlo en el entrenamiento contaminaría las fronteras de decisión sin aportar capacidad predictiva real.

### Selección de modelo por MCC
El criterio de selección entre modelos fue el MCC (Matthews Correlation Coefficient), no accuracy ni F1. La razón: accuracy sobre datos desbalanceados (96.6% de máquinas sanas) puede ser alta prediciendo siempre "no falla"; F1 depende de qué clase se define como positiva. MCC usa las cuatro celdas de la matriz de confusión y penaliza tanto los falsos positivos como los falsos negativos, lo que lo hace más honesto bajo desbalance.

**Límite conocido y documentado:** el MCC global penaliza poco el abandono de clases muy minoritarias. TWF representa el 0.46% del dataset; ignorarlo casi no mueve el MCC. Este fenómeno se observó empíricamente en los resultados (ver Resultados) y es parte del hallazgo central del proyecto, no un defecto que se ocultó.

### Umbral de decisión: verificado, no ajustado
El umbral del detector se analizó sobre probabilidades out-of-fold (`cross_val_predict`) sobre el conjunto de entrenamiento, nunca sobre test. El análisis mostró que Gradient Boosting concentra las probabilidades en los extremos, con un recall constante: ajustar el umbral con `TunedThresholdClassifierCV` degradó el MCC (88.55% -> 85.79%) y casi duplicó los falsos positivos (8 -> 14). Se retuvo el umbral por defecto (0.5), con el experimento de ajuste preservado en el notebook como evidencia.

### Sin SMOTE
Se descartó sobremuestreo sintético (SMOTE) en favor de pesos de clase (`class_weight='balanced'` para Random Forest y Decision Tree, `sample_weight` para Gradient Boosting, `scale_pos_weight` para XGBoost). Dos razones: SMOTE dentro de validación cruzada introduce leakage si no se encapsula correctamente en el pipeline, y genera muestras sintéticas en el espacio de features sin garantía de que correspondan a condiciones de operación reales. Los pesos de clase preservan la distribución original y son reproducibles sin dependencias adicionales.

### Exclusión de filas sin tipo de falla
El dataset contiene 9 filas donde `Machine failure = 1` pero ningún tipo de falla está marcado. Estas filas son válidas para el detector binario (etiqueta falla/no-falla correcta) pero no para el diagnosticador (no hay etiqueta de tipo que aprender). Se incluyeron en etapa 1 y se excluyeron de etapa 2, preservando toda la información disponible sin contaminar el target del diagnosticador.

## Feature engineering

Se derivaron tres features a partir de la física del proceso: potencia mecánica
(Power = Torque x ω (velocidad angular, rad/s)), gradiente térmico (diff_temp = Process temp - Air temp) y
sobreesfuerzo (overstrain = Torque x Tool wear). Estas combinan variables ya
presentes en el dataset según las relaciones que definen cada modo de falla.

**Impacto medido:** se entrenó el mismo detector (Gradient Boosting, mismo split,
mismo régimen de pesos) sobre las variables crudas únicamente, sin estas tres
derivadas, como control.

| | Recall | Precision | F1-Score | MCC | FN | FP |
|---|---|---|---|---|---|---|
| Gradient Boosting - sin features físicas | 90.20% | 25.70% | 40.00% | 45.29% | 10 | 266 |
| Gradient Boosting - con features físicas | 86.27% | 91.67% | 88.89% | 88.55% | 14 | 8 |

El recall se mantiene similar (90.2% vs 86.3%); la diferencia está casi enteramente
en precision. Sin las features derivadas, el detector generaliza la noción de
falla de forma demasiado amplia y dispara sobre 266 de 2898 máquinas sanas
(9.2%). Las features físicas, al capturar directamente los mecanismos de falla
documentados, permiten al modelo discriminar entre desviaciones normales de
operación y condiciones de falla reales.

## Resultados

Se comparó el sistema en cascada contra dos versiones de un clasificador multiclase plano (Gradient Boosting sin tunear y con gridsearch de hiperparámetros), evaluados sobre el mismo split de test con tres métricas: MCC, F1-macro y balanced accuracy.

| | MCC | F1-macro | Balanced Acc |
|---|---|---|---|
| Cascada | 87.49% | 75.76% | 76.53% |
| Multiclase default | 78.31% | **80.33%** | **85.57%** |
| Multiclase gridsearch | **87.64%** | 77.74% | 78.85% |

**Ningún enfoque domina.** El multiclase con gridsearch tiene el MCC más alto, apenas por encima de la cascada (87.64% vs 87.49%, una diferencia de 0.15 puntos, dentro del margen de ruido esperable entre corridas). El multiclase default lidera F1-macro y balanced accuracy por un margen mayor.

### Recall por clase (detección de cada tipo de falla)

| Clase | Cascada | Multiclase default | Multiclase gridsearch |
|---|---|---|---|
| No_falla | 100% | 99% | 100% |
| HDF | 95% | 100% | 95% |
| OSF | 96% | 100% | 100% |
| PWF | 92% | 96% | 100% |
| **TWF** | **0%** | **33%** | **0%** |

El motivo de la aparente contradicción entre MCC y las otras métricas es el manejo de TWF, la clase minoritaria (0.46% del dataset). El multiclase default es el único de los tres que detecta algo de TWF, lo que le sube F1-macro y balanced accuracy; estas métricas ponderan cada clase por igual. Pero el 33% de recall no es limpio: la precision de TWF en el default es de apenas 0.10, es decir, 9 de cada 10 veces que el modelo predice TWF, se equivoca. La cascada y el multiclase con gridsearch abandonan TWF completamente (recall 0%), lo cual apenas penaliza su MCC porque esta métrica no protege fuerte a clases extremadamente minoritarias. El resultado es contraintuitivo: el modelo con mejor MCC global es el que peor maneja la clase más difícil del problema y el modelo que mejor la maneja, lo hace con una tasa de error muy alta.

## Limitaciones

### TWF: un modo de falla irrecuperable con las features disponibles

TWF (Tool Wear Failure) es la única clase que ningún modelo entrenado en este proyecto logra detectar de forma confiable. El análisis de causa raíz descarta que sea un problema de arquitectura o de threshold, y apunta a un límite estructural del dataset:

1. **El detector no ve señal.** Las probabilidades de falla que Gradient Boosting asigna a los 12 TWF reales de test están entre 1e-7 y 1e-3, por lo que el modelo los clasifica como sanos con altísima confianza. Ningún umbral de decisión razonable los rescata: capturarlos exigiría bajar el umbral hasta aproximadamente 0.003, lo que dispararía falsos positivos.

2. **La zona de desgaste crítico no discrimina.** TWF se produce, por diseño del dataset, en un rango de desgaste de herramienta entre 200 y 240 minutos. Ese rango contiene 43 de los 46 casos de TWF del dataset (93%), pero también 671 máquinas sanas — un 94% de falsos positivos si se usara como regla de decisión. La condición es necesaria en la gran mayoría de los casos pero no suficiente: casi todos los TWF caen ahí, pero la inmensa mayoría de lo que cae ahí son máquinas sanas.

3. **La variable relevante ya está disponible, sin efecto.** `Tool wear` crudo (en minutos) ya forma parte del conjunto de features en las dos etapas. El detector no lo aprovecha para separar TWF de máquinas sanas. Esto no es un problema de representación de datos: es evidencia de que dentro del rango crítico, no existe una combinación de las variables disponibles que distinga un caso del otro.

4. **Es estocástico por diseño.** La documentación del dataset señala que TWF ocurre en un punto aleatorio dentro de la ventana de desgaste crítico, a diferencia de OSF (determinístico: torque x desgaste > umbral), que el sistema clasifica con precisión y recall perfectos. AI4I introdujo ruido en este modo de falla, y ningún feature engineering aplicado en este proyecto pudo recuperar una señal que el generador del dataset no incluyó.

En la cascada, esta limitación se manifiesta como abandono completo de clase: los 12 TWF de test caen enteros en los falsos negativos del detector y nunca llegan al diagnosticador. Pese a que este, evaluado de forma aislada sobre fallas reales, clasifica TWF con 80% de precisión y 86% de recall. El diagnosticador sabe reconocer TWF; el problema es que este nunca lo ve.

### Alarm fatigue residual
Aun con las features físicas, el detector genera 8 falsos positivos sobre 2898 máquinas sanas de test. Es una mejora sustancial respecto al detector sin features (266 FP), pero en un entorno de producción con miles de máquinas, un 0.3% de tasa de falsa alarma sigue representando un volumen de alertas a gestionar.

### Dataset sintético
AI4I 2020 es un dataset generado sintéticamente, no telemetría real. Las relaciones entre features y fallas son más limpias de lo esperable en un entorno real, lo que probablemente infla el desempeño de los modelos de este proyecto respecto a lo que se observaría en producción.

### Cómo se llevaría esto a producción
- **Recalibración periódica del umbral**: el punto de operación (0.5) fue elegido sobre la distribución de este dataset; en producción requeriría reevaluación basada en sensores y desgaste de equipos reales.
- **Monitoreo de la tasa de falsos positivos en el tiempo**: una tasa que crece indicaría degradación del modelo o cambios en las condiciones de operación no capturados en el entrenamiento.
- **Reentrenamiento con fallas etiquetadas nuevas**: especialmente relevante para TWF, donde más datos históricos podrían revelar una señal que este dataset, por su tamaño y diseño, no contiene.
- **Etiquetado de causa por parte de mantenimiento**: la calidad del diagnosticador depende de que las fallas reales se registren con su tipo correcto. Este es un proceso humano que en producción no es automático como en el dataset utilizado.
