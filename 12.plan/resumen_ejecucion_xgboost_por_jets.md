# Resumen De Ejecucion: XGBoost Por Jets

## Contexto

Partimos del experimento `11.Mezcla/04`, que entrenaba un `XGBoost` separado para cada valor de `PRI_jet_num` usando el preprocesado final `i3_tree_safe_mix_by_jets`.

La idea era probar si separar el problema en 4 modelos especializados podia mejorar el resultado:

- `jet 0`
- `jet 1`
- `jet 2`
- `jet 3`

El baseline local anterior mas fuerte era el arbol de `10.try/06`:

- modelo: `v2_physics_by_jet`
- `AMS = 1.107879`
- `AUC = 0.909888`

En leaderboard, el mejor resultado que teniamos hasta ahora era:

- archivo: `higgs_submission_i3_tree_safe_mix_xgboost_best_20260501_225111.csv`
- `publicScore = 3.47823`
- `privateScore = 3.57621`

## Plan Ejecutado

Se creo y ejecuto el plan en `12.plan`.

Archivos principales:

- `12.plan/plan_mejora_xgboost_por_jets.md`
- `12.plan/xgboost_by_jets_plan_utils.py`
- `12.plan/05_thresholds_por_jet.py`
- `12.plan/06_tuning_xgboost_por_jets.py`
- `12.plan/07_calibracion_final_xgboost_por_jets.py`
- `12.plan/08_entrenamiento_final_xgboost_por_jets.ipynb`

La regla de trabajo fue mantener fijo el preprocesado y cambiar solo:

- los thresholds;
- los hiperparametros de XGBoost por jet;
- la politica final de combinacion.

## Fase 1: Thresholds Por Jet Sobre Modelos Actuales

Primero se tomaron los modelos por jet actuales y se compararon varias politicas de decision:

- umbral global unico;
- umbral independiente por jet;
- optimizacion conjunta de los 4 umbrales contra `AMS` global.

Resultado principal:

- baseline con umbral global: `AMS = 1.083018`
- mejor politica por thresholds: `AMS = 1.104753`

Conclusion de fase:

- cambiar solo la politica de thresholds mejora claramente el modelo por jets;
- aun asi, no alcanza para superar al mejor arbol local anterior de `10.try/06`.

Artefactos:

- `12.plan/results/05_thresholds_policy_comparison.csv`
- `12.plan/results/05_thresholds_policy_details.json`

## Fase 2: Tuning Cauteloso De XGBoost Por Jet

Despues se busco un set de hiperparametros especifico para cada jet.

La busqueda fue local y por etapas, no una grilla grande:

- `scale_pos_weight`
- parametros estructurales;
- capacidad;
- sampling;
- regularizacion.

Mejores elecciones por jet:

- `jet 0`: `reg_alpha = 0.1`
- `jet 1`: `scale_pos_weight = 1.0`
- `jet 2`: `learning_rate = 0.1`, `n_estimators = 240`
- `jet 3`: `subsample = 0.8`, `colsample_bytree = 0.8`

Metricas por jet del modelo final:

| jet | AUC | AMS | threshold |
| --- | ---: | ---: | ---: |
| 0 | 0.893698 | 0.773110 | 0.880455 |
| 1 | 0.890174 | 0.563862 | 0.849449 |
| 2 | 0.931770 | 0.650208 | 0.918633 |
| 3 | 0.903803 | 0.255729 | 0.729651 |

Artefactos:

- `12.plan/results/06_tuning_xgboost_por_jets_runs.csv`
- `12.plan/results/06_tuning_xgboost_por_jets_best_rows.csv`
- `12.plan/results/06_tuning_xgboost_por_jets_best_params.json`

## Fase 3: Calibracion Final De Thresholds

Con los 4 modelos tuneados fijos, se recalibraron los thresholds.

Politicas comparadas:

- umbral global unico;
- umbral independiente por jet;
- optimizacion conjunta desde umbral global;
- optimizacion conjunta desde umbrales independientes.

La mejor politica local fue:

- `joint_thresholds_from_independent`

Thresholds finales:

- `jet 0`: `0.922631`
- `jet 1`: `0.849449`
- `jet 2`: `0.886892`
- `jet 3`: `0.729651`

Resultado local final:

- `AMS = 1.147119`
- `AUC = 0.903617`
- `precision = 0.928863`
- `recall = 0.397806`
- `F1 = 0.557045`

Comparacion local contra el mejor arbol anterior:

- `delta AMS = +0.039240`
- `delta AUC = -0.006271`

Lectura:

- mejoro fuerte en `AMS` local;
- bajo en `AUC`;
- quedo muy conservador, con precision alta y recall bajo.

Artefactos:

- `12.plan/results/07_calibracion_final_policy_comparison.csv`
- `12.plan/results/07_calibracion_final_threshold_details.json`

## Notebook Final

El cierre quedo en:

- `12.plan/08_entrenamiento_final_xgboost_por_jets.ipynb`

Ese notebook:

- carga los mejores hiperparametros por jet;
- reentrena los 4 modelos finales;
- calibra y aplica la politica final de thresholds;
- compara contra el mejor modelo anterior;
- exporta bundle, metricas y submission.

Artefactos finales:

- `12.plan/results/08_final_comparison.csv`
- `12.plan/results/08_final_group_metrics.csv`
- `12.plan/results/08_final_threshold_policy_comparison.csv`
- `12.plan/results/08_final_summary.txt`
- `12.plan/models/xgboost_i3_tree_safe_mix_by_jets_final_bundle`
- `12.plan/results/higgs_submission_xgboost_i3_tree_safe_mix_by_jets_final.csv`

## Submission A Kaggle

Se envio el archivo:

- `12.plan/results/higgs_submission_xgboost_i3_tree_safe_mix_by_jets_final.csv`

Mensaje de submission:

- `XGBoost por jets final 12.plan tuning thresholds`

Kaggle respondio correctamente:

- `Successfully submitted to Higgs Boson Machine Learning Challenge`

Score obtenido:

- `publicScore = 3.41715`
- `privateScore = 3.53878`

## Conclusion

El plan logro mejorar la metrica local `AMS`:

- anterior local: `1.107879`
- final local: `1.147119`

Pero no logro mejorar el leaderboard:

- mejor leaderboard previo: `privateScore = 3.57621`
- modelo final `12.plan`: `privateScore = 3.53878`

La conclusion principal es que el enfoque por jets con tuning y thresholds especificos tiene senial local, pero probablemente sobreajusto el split de validacion usado.

## Lectura Metodologica

El resultado muestra una brecha entre validacion local y Kaggle.

Esto sugiere que seguir optimizando sobre un unico split puede llevar a elegir modelos demasiado adaptados a ese split.

El siguiente paso mas razonable no es seguir agregando complejidad al modelo por jets, sino revisar la robustez de validacion:

- multi-seed validation;
- K-fold o varios splits estratificados;
- calibracion de thresholds estable entre splits;
- comparacion robusta contra `i3_tree_safe_mix` global.

## Recomendacion Para Seguir

Crear una nueva etapa, por ejemplo `13.robustez`, enfocada en:

1. tomar el mejor leaderboard actual como baseline real;
2. evaluar `i3_tree_safe_mix` global y el modelo por jets en varios splits;
3. medir estabilidad de `AMS`, no solo el mejor valor en un split;
4. calibrar thresholds de forma mas robusta;
5. enviar a Kaggle solo modelos que mejoren de forma consistente.
