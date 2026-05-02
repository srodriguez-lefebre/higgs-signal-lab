# Plan De Mejora De XGBoost Por Jets

> **Para ejecutar después:** usar este plan como guía de trabajo secuencial. Mantener el preprocesado fijo y cambiar una sola cosa importante por fase.

**Objetivo:** mejorar el modelo por jets actual de `11.Mezcla/03` y `11.Mezcla/04` sin mezclar causas, separando claramente preprocesado, entrenamiento y política final de decisión.

**Baseline actual:** `11.Mezcla/04_train_package_xgboost_by_jets.ipynb`

- Preprocesado fijo por jet: `i3_tree_safe_mix_by_jets`
- Un modelo `XGBoost` por `PRI_jet_num = 0, 1, 2, 3`
- Comparación principal contra `10.try/06`
- Resultado actual del ensemble por jets:
  - `AUC = 0.908090`
  - `AMS = 1.083018`
  - `precision = 0.918535`
  - `recall = 0.444847`

**Hipótesis principal:** el resultado actual puede estar limitado por dos factores distintos:

1. Los hiperparámetros no están adaptados a cada grupo de jets.
2. La política final de decisión usa un umbral global único y puede estar descartando demasiados positivos.

## Preprocesado Fijo

Durante este plan no se cambia el preprocesado base.

- Mantener fijo `11.Mezcla/03_xgboost_i3_tree_safe_mix_por_jets.ipynb`
- Mantener fijo `11.Mezcla/i3_tree_safe_mix_by_jets.py`
- Mantener fijo el split global `train/valid`
- Mantener el ruteo por `PRI_jet_num` crudo

La idea es que cualquier mejora observada se pueda atribuir a:

- mejor tuning del modelo por jet;
- mejor política de thresholds;
- o ambas.

## Criterio De Éxito

**Métrica principal:**

- `AMS` global sobre validación

**Métricas secundarias:**

- `AUC` global
- `recall` global
- `precision` global
- estabilidad de resultados por jet

**Comparaciones obligatorias:**

1. Contra el baseline de `11.Mezcla/04`
2. Contra el mejor árbol anterior de `10.try/06` (`v2_physics_by_jet`)
3. Si el candidato final parece fuerte, contra leaderboard mediante submission

## Reglas De Trabajo

1. No cambiar preprocesado y tuning al mismo tiempo.
2. No cambiar tuning y thresholds al mismo tiempo en una misma comparación principal.
3. Registrar cada corrida con parámetros, métricas y observaciones.
4. Mantener un baseline reproducible y siempre comparable.
5. Si una mejora es mínima, tratarla como incierta y no como victoria automática.

## Formato De Implementación

No hace falta que cada fase quede en notebook.

- Las fases intermedias se pueden resolver con Python normal.
- Está bien usar módulos `.py`, scripts auxiliares y tablas de resultados para iterar más rápido.
- El cierre sí debe quedar en un notebook final.

**Notebook final obligatorio:**

- entrenar los mejores 4 modelos por jet ya seleccionados;
- combinarlos con la política final correcta de ruteo por `PRI_jet_num`;
- aplicar la política final de thresholds que haya ganado;
- comparar el resultado final contra el modelo anterior que teníamos;
- dejar listo el candidato final de submission.

## Fase 1: Diagnóstico De Thresholds Sobre Los Modelos Actuales

**Objetivo:** entender cuánto del problema actual viene de la política de decisión final y no del entrenamiento.

**Idea:** usar exactamente los 4 modelos ya entrenados en `11.Mezcla/04`, pero cambiar la forma de decidir `s` o `b`.

**Experimentos:**

1. Medir el baseline actual con `threshold` global único.
2. Probar `threshold` independiente por jet usando el mejor AMS de cada jet por separado.
3. Probar optimización conjunta de 4 thresholds para maximizar el `AMS` global final.
4. Comparar las tres políticas:
   - umbral global único;
   - umbral por jet independiente;
   - optimización conjunta de los 4 umbrales.

**Entregable sugerido:**

- `11.Mezcla/05_thresholds_por_jet.py`

**Resultado esperado de esta fase:**

- saber si el cuello de botella principal está en la decisión final;
- saber si vale la pena mantener una política por jet aun antes de retunear el modelo.

## Fase 2: Tuning De XGBoost Por Jet

**Objetivo:** encontrar mejores hiperparámetros para cada uno de los 4 subconjuntos de jets, manteniendo fijo el preprocesado.

**Idea:** cada jet puede necesitar un perfil distinto de complejidad, regularización, subsampling y peso de clases.

**Hiperparámetros candidatos a revisar:**

- `max_depth`
- `learning_rate`
- `n_estimators`
- `min_child_weight`
- `gamma`
- `subsample`
- `colsample_bytree`
- `reg_lambda`
- `reg_alpha`
- `scale_pos_weight`

**Entregable sugerido:**

- `11.Mezcla/06_tuning_xgboost_por_jets.py`

### Metodología Cautelosa Para El Tuning

Esta fase debe hacerse con cuidado. La prioridad no es probar mucho, sino probar bien.

**Reglas:**

1. Empezar desde el baseline ganador actual y no desde una grilla arbitraria.
2. Cambiar pocos hiperparámetros por vez.
3. Empezar con búsquedas chicas y locales alrededor del baseline.
4. Anotar cada corrida aunque salga peor.
5. No declarar ganador un experimento por diferencias muy pequeñas sin contexto.

**Secuencia sugerida:**

1. Tomar los hiperparámetros actuales como punto cero.
2. Ajustar primero los parámetros estructurales:
   - `max_depth`
   - `min_child_weight`
   - `gamma`
3. Luego ajustar capacidad y ritmo:
   - `learning_rate`
   - `n_estimators`
4. Luego ajustar robustez:
   - `subsample`
   - `colsample_bytree`
5. Luego ajustar regularización:
   - `reg_lambda`
   - `reg_alpha`
6. Revisar al final `scale_pos_weight` si sigue haciendo falta.

**Forma de explorar:**

- Primero por un jet a la vez.
- No abrir 4 búsquedas grandes en paralelo sin haber entendido el patrón.
- Si un jet mejora con una dirección clara, recién ahí profundizar en ese jet.
- Si un jet no mejora, volver al baseline y probar otra familia de cambios.

**Registro mínimo por corrida:**

- `jet_num`
- fecha/hora
- notebook o bloque que la produjo
- hiperparámetros usados
- `AUC`
- `AMS`
- `precision`
- `recall`
- `fit_seconds`
- observación corta:
  - “mejora AMS pero cae recall”
  - “mejora AUC sin impacto real en AMS”
  - “resultado inestable o marginal”

**Criterio para avanzar:**

- si una configuración mejora de forma clara al baseline de ese jet, guardarla como candidata;
- si no mejora claro, no reemplazar el baseline;
- al final de la fase, cerrar un “mejor modelo por jet” provisional.

## Fase 3: Calibración Final De Thresholds Sobre Los Modelos Tuneados

**Objetivo:** una vez fijados los 4 modelos tuneados, encontrar la mejor política final de decisión.

**Experimentos:**

1. Recalcular el mejor threshold por jet de manera individual.
2. Optimizar en conjunto los 4 thresholds para maximizar el `AMS` global.
3. Comparar contra:
   - umbral global único;
   - umbral por jet individual;
   - umbral por jet optimizado en conjunto.

**Entregable sugerido:**

- `11.Mezcla/07_calibracion_final_xgboost_por_jets.py`

**Resultado esperado de esta fase:**

- cerrar una política final de inferencia por jet que sea coherente con los modelos tuneados;
- dejar listo el candidato final para submission.

## Evaluación Final

**Cierre en notebook:**

- `11.Mezcla/08_entrenamiento_final_xgboost_por_jets.ipynb`

Ese notebook final debe:

1. tomar los mejores hiperparámetros elegidos para cada jet;
2. entrenar de nuevo los 4 modelos finales;
3. enrutar correctamente la inferencia según `PRI_jet_num`;
4. aplicar la política final de thresholds ganadora;
5. comparar contra el modelo anterior que teníamos como referencia;
6. dejar exportados resultados, bundle y artefactos finales.

Al terminar las tres fases, comparar al menos estos candidatos:

1. `10.try/06` mejor árbol histórico
2. `11.Mezcla/04` baseline por jets actual
3. mejor variante de Fase 1
4. mejor variante de Fase 2 sin recalibración final
5. candidato final de Fase 3

**Pregunta final a responder:**

- ¿el enfoque por jets realmente supera al mejor árbol global previo, o solo mejora localmente en validación?

## Artefactos A Mantener

Crear y mantener, dentro de `11.Mezcla/results`, tablas simples y acumulativas con:

- comparación global por experimento;
- métricas por jet;
- thresholds usados;
- observaciones cortas por corrida.

También conviene mantener un resumen manual corto en esta carpeta `12.plan` si el número de corridas empieza a crecer.

## Riesgos Metodológicos

1. Sobreajuste al mismo split de validación.
2. Conclusiones apresuradas por diferencias pequeñas.
3. Mezclar en un mismo paso cambios de modelo y cambios de thresholds.
4. Perder trazabilidad de qué combinación produjo cada resultado.

## Cierre Esperado

Si el plan sale bien, el resultado final debería dejar:

- 4 modelos por jet con hiperparámetros justificados;
- una política final de thresholds por jet también justificada;
- una comparación limpia contra `10.try/06`;
- y un candidato serio para submission final.
