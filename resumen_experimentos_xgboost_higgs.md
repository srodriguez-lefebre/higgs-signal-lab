# Resumen De Experimentos XGBoost Higgs

Este documento resume las pruebas realizadas sobre modelos XGBoost, redes y ensembles durante las etapas `10.try` a `17.cv_jets`.

## Mejor Resultado Actual

El mejor resultado hasta ahora es el ensemble de `13.comb`:

- archivo: `13.comb/results/higgs_submission_i3_tree_safe_mix_plus_v2_physics_by_jet.csv`
- descripcion: `i3_tree_safe_mix + v2_physics_by_jet`, mezcla por percentile rank
- pesos: `30% i3_tree_safe_mix`, `70% v2_physics_by_jet`
- `publicScore = 3.46156`
- `privateScore = 3.60593`

## 10.try

### 10.try/04: XGBoost Sobre V2

Se compararon nuevos preprocessings contra los resultados previos de `7.Arboles`.

Mejor modelo local:

- dataset: `v2_physics_by_jet`
- modelo: XGBoost con hiperparametros heredados del mejor arbol previo
- `AUC valid = 0.909888`
- `AMS valid = 1.107879`
- `accuracy = 0.836120`
- `precision = 0.743623`
- `recall = 0.796311`

Submission:

- archivo: `10.try/results/higgs_submission_v2_physics_by_jet_xgboost_best_20260501_221359.csv`
- `publicScore = 3.39520`
- `privateScore = 3.55958`

### 10.try/05: NN Best Optuna Sobre V2

Se entreno la mejor arquitectura de NN encontrada previamente sobre `v2_nn_dense`.

Resultado local:

- `AUC valid = 0.908083`
- `AMS valid = 1.092437`
- `accuracy = 0.825840`
- `precision = 0.711093`
- `recall = 0.828295`

Submission:

- archivo: `10.try/results/higgs_submission_v2_nn_dense_best_optuna_20260501_222604.csv`
- `publicScore = 3.29235`
- `privateScore = 3.46147`

### 10.try/06: Comparacion NN vs Arbol

Resultado:

- mejor arbol: `v2_physics_by_jet`
- mejor NN: `v2_nn_dense_best_optuna`
- el arbol gano en `AMS`, `AUC`, `accuracy` y `precision`
- la NN solo gano en `recall`

## 11.Mezcla

### 11.Mezcla/02: i3_tree_safe_mix Global

Se entreno XGBoost global sobre el preprocesado `i3_tree_safe_mix`.

Resultado local:

- `AUC valid = 0.909116`
- `AMS valid = 1.098125`
- `accuracy = 0.834920`
- `precision = 0.741252`
- `recall = 0.796195`

Submission:

- archivo: `higgs_submission_i3_tree_safe_mix_xgboost_best_20260501_225111.csv`
- `publicScore = 3.47823`
- `privateScore = 3.57621`

Este fue durante un tiempo el mejor leaderboard.

### 11.Mezcla/04: XGBoost Por Jets Baseline

Se entreno un modelo separado para cada `PRI_jet_num`.

Resultado local:

- `AUC valid = 0.908090`
- `AMS valid = 1.083018`
- `precision = 0.918535`
- `recall = 0.444847`

Lectura:

- muy conservador;
- precision alta;
- recall bajo;
- no mejora al arbol global.

## 12.plan

Objetivo: mejorar el modelo por jets con tuning y thresholds.

### Fase 1: Thresholds Sobre Modelos Por Jets Actuales

Se compararon:

- umbral global unico;
- umbral independiente por jet;
- thresholds conjuntos por jet.

Resultado:

- baseline por jets con umbral global: `AMS = 1.083018`
- mejor politica de thresholds: `AMS = 1.104753`

Mejoro localmente, pero no supero al mejor arbol previo.

### Fase 2: Tuning XGBoost Por Jet

Mejores cambios por jet:

- `jet 0`: `reg_alpha = 0.1`
- `jet 1`: `scale_pos_weight = 1.0`
- `jet 2`: `learning_rate = 0.1`, `n_estimators = 240`
- `jet 3`: `subsample = 0.8`, `colsample_bytree = 0.8`

### Fase 3: Calibracion Final

Resultado local final:

- `AUC valid = 0.903617`
- `AMS valid = 1.147119`
- `precision = 0.928863`
- `recall = 0.397806`

Submission:

- archivo: `12.plan/results/higgs_submission_xgboost_i3_tree_safe_mix_by_jets_final.csv`
- `publicScore = 3.41715`
- `privateScore = 3.53878`

Conclusion:

- mejoro mucho el AMS local;
- no mejoro Kaggle;
- hubo sobreajuste al split de validacion.

## 13.comb

Objetivo: combinar `i3_tree_safe_mix` global con `v2_physics_by_jet`.

Se entrenaron ambos XGBoost y se barrieron mezclas de scores:

- probabilidad cruda;
- percentile rank.

Mejor mezcla local:

- tipo: `percentile_rank`
- pesos: `30% i3_tree_safe_mix`, `70% v2_physics_by_jet`
- `AUC valid = 0.910106`
- `AMS valid = 1.118636`

Submission:

- archivo: `13.comb/results/higgs_submission_i3_tree_safe_mix_plus_v2_physics_by_jet.csv`
- `publicScore = 3.46156`
- `privateScore = 3.60593`

Conclusion:

- mejor resultado global hasta ahora;
- el ensemble generaliza mejor que los modelos por separado.

## 14.invest

Objetivo: investigar si el modelo por jets fallaba por no usar `sample_weight`.

Se probaron variantes por jet:

- sin `sample_weight`;
- con `sample_weight`;
- `scale_pos_weight` por conteo;
- `scale_pos_weight` por suma de pesos;
- `max_delta_step = 1`.

Resultado local del mejor weighted por jets:

- `AUC valid = 0.819138`
- `AMS valid = 1.130993`
- `precision = 0.917442`
- `recall = 0.368390`

Submission:

- archivo: `14.invest/results/higgs_submission_weighted_xgboost_by_jets.csv`
- `publicScore = 3.39848`
- `privateScore = 3.52492`

Conclusion:

- usar `sample_weight` no mejoro el leaderboard;
- el ranking del modelo weighted por jets fue peor;
- no conviene como modelo final aislado.

## 15.tune_global

Objetivo: tunear por separado los dos modelos que formaban el mejor ensemble:

- `i3_tree_safe_mix`
- `v2_physics_by_jet`

### i3_tree_safe_mix Tuneado

Mejor cambio:

- `reg_lambda = 10.0`

Resultado local:

- `AUC valid = 0.909336`
- `AMS valid = 1.113891`

### v2_physics_by_jet Tuneado

Mejor cambio:

- `learning_rate = 0.07`
- `n_estimators = 330`
- `max_depth = 5`
- `max_delta_step = 1`

Resultado local:

- `AUC valid = 0.908828`
- `AMS valid = 1.157615`

El mejor blend local fue en realidad:

- `100% v2_physics_by_jet tuneado`

Submission:

- archivo: `15.tune_global/results/higgs_submission_tuned_i3_plus_tuned_v2.csv`
- `publicScore = 3.38314`
- `privateScore = 3.57271`

Conclusion:

- gran mejora local;
- no supero Kaggle;
- senal clara de sobreajuste al split.

## 16.cv

Objetivo: comprobar si el tuning global estaba sobreajustando.

Se hizo validacion cruzada de 5 folds, refiteando el preprocesador dentro de cada fold.

Modelos evaluados:

- `i3_tree_safe_mix` baseline
- `i3_tree_safe_mix` tuneado
- `v2_physics_by_jet` baseline
- `v2_physics_by_jet` tuneado

Resumen CV:

| modelo | variante | AMS mean | AUC mean |
| --- | --- | ---: | ---: |
| `v2_physics_by_jet` | baseline | 1.631106 | 0.912448 |
| `i3_tree_safe_mix` | baseline | 1.626776 | 0.911967 |
| `v2_physics_by_jet` | tuned | 1.622239 | 0.911795 |
| `i3_tree_safe_mix` | tuned | 1.620329 | 0.912069 |

Conclusion:

- el tuning de `15.tune_global` sobreajusto;
- los baseline eran mas robustos en CV.

Tambien se probo un ensemble elegido por CV:

- `20% i3_tree_safe_mix tuneado`
- `80% v2_physics_by_jet tuneado`
- percentile rank

Submission:

- archivo: `16.cv/results/higgs_submission_cv_selected_tuned_rank_20_80.csv`
- `publicScore = 3.41424`
- `privateScore = 3.60259`

Quedo cerca del mejor, pero no lo supero.

## 17.cv_jets

Objetivo: mejorar el modelo por jets usando validacion cruzada especifica.

Se evaluaron 3 variantes por jets en 5 folds:

- `baseline_shared`
- `plan12_tuned`
- `weighted14`

Resumen CV:

| variante | mejor politica | AMS mean | AUC mean |
| --- | --- | ---: | ---: |
| `weighted14` | thresholds conjuntos | 1.670198 | 0.900278 |
| `plan12_tuned` | thresholds conjuntos | 1.641765 | 0.906927 |
| `baseline_shared` | thresholds conjuntos | 1.625486 | 0.910841 |

Conclusion CV:

- `weighted14` parecia el mejor por jets en CV;
- pero tenia AUC bastante menor;
- los thresholds eran inestables entre folds.

### Por Jets Como Tercer Score Del Ensemble

Se agrego `weighted14` al mejor ensemble de `13.comb`.

Mejor mezcla local:

- `15.8% i3_tree_safe_mix`
- `77.2% v2_physics_by_jet`
- `7.0% weighted jets`
- `AMS local = 1.129409`

Submission:

- archivo: `17.cv_jets/results/higgs_submission_best_plus_weighted_jets.csv`
- `publicScore = 3.45492`
- `privateScore = 3.57706`

No mejoro el ensemble original.

### Por Jets Solo Con Full Train

Se entreno `weighted14` por jets sobre todo el training y se probaron thresholds derivados de CV.

Submission:

- archivo: `17.cv_jets/results/higgs_submission_weighted_jets_full_cv_median_count.csv`
- `publicScore = 1.66314`
- `privateScore = 1.68697`

Conclusion:

- el por-jets solo generalizo muy mal;
- la transferencia de thresholds de CV a full train/test fue mala;
- los scores por jet no estaban calibrados de forma estable.

## Tabla Kaggle Principal

| archivo | descripcion | public | private |
| --- | --- | ---: | ---: |
| `higgs_submission_i3_tree_safe_mix_plus_v2_physics_by_jet.csv` | ensemble `13.comb` | 3.46156 | 3.60593 |
| `higgs_submission_cv_selected_tuned_rank_20_80.csv` | ensemble tuneado elegido por CV | 3.41424 | 3.60259 |
| `higgs_submission_i3_tree_safe_mix_xgboost_best_20260501_225111.csv` | `i3_tree_safe_mix` global | 3.47823 | 3.57621 |
| `higgs_submission_best_plus_weighted_jets.csv` | ensemble + jets weighted | 3.45492 | 3.57706 |
| `higgs_submission_tuned_i3_plus_tuned_v2.csv` | tuning global `15.tune_global` | 3.38314 | 3.57271 |
| `higgs_submission_v2_physics_by_jet_xgboost_best_20260501_221359.csv` | `v2_physics_by_jet` baseline | 3.39520 | 3.55958 |
| `higgs_submission_xgboost_i3_tree_safe_mix_by_jets_final.csv` | por-jets `12.plan` | 3.41715 | 3.53878 |
| `higgs_submission_weighted_xgboost_by_jets.csv` | por-jets weighted `14.invest` | 3.39848 | 3.52492 |
| `higgs_submission_v2_nn_dense_best_optuna_20260501_222604.csv` | NN best optuna sobre V2 | 3.29235 | 3.46147 |
| `higgs_submission_weighted_jets_full_cv_median_count.csv` | por-jets full train con thresholds CV | 1.66314 | 1.68697 |

## Conclusiones

1. El mejor resultado actual es el ensemble de `13.comb`.
2. Los modelos por jets mejoran localmente con tuning y CV, pero no generalizan bien en Kaggle.
3. El tuning sobre un unico split fue enganoso: `15.tune_global` subio localmente pero bajo en Kaggle.
4. La validacion cruzada mostro que los baseline globales eran mas robustos que los modelos tuneados.
5. El score por-jets tiene calibracion inestable y AUC menor, por eso no funciono bien solo ni como tercer componente fuerte del ensemble.
6. La mejor estrategia hasta ahora es combinar modelos globales robustos usando percentile ranks.

## Siguiente Linea Recomendada

No seguir empujando el por-jets como modelo independiente.

Las opciones mas prometedoras son:

- agregar otros modelos globales buenos al ensemble, por ejemplo `i2_arboles`, `i3_arboles` y `v2_tree_safe`;
- hacer busqueda de pesos por CV, no solo por un split;
- mantener mezcla por ranks;
- enviar solo candidatos que sean estables en CV.
