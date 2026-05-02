# 20.Arbol_i3

Busqueda de hiperparametros para el modelo XGBoost sobre los datos `i3_arboles`, tomando como punto de partida el mejor arbol historico de `7.Arboles`.

## Datos

Se usan los artefactos procesados en:

`data/higgs-boson/processed/i3_arboles`

Archivos usados:

- `X_train_processed.csv`
- `X_test_processed.csv`
- `y_train.csv`
- `train_weight.csv`
- `train_event_id.csv`
- `test_event_id.csv`

La validacion local usa split estratificado 90/10 con `random_state=42`, igualando el criterio usado para reproducir el baseline historico.

## Comet

La corrida usa `comet_ml.Experiment` con:

- `project_name`: `Primer Proyecto TAA`
- `workspace`: `equipo-a`

La API key se lee desde `keys/.comet.config` y no se imprime ni se guarda en resultados.

Cada trial registra en Comet:

- hiperparametros completos de XGBoost;
- metadata del split y del dataset;
- `best_ams`, `ams`, `ams_threshold`, `best_threshold`;
- `roc_auc`;
- precision, recall, F1 y accuracy en el umbral AMS;
- metricas default con umbral interno del clasificador;
- pesos acumulados de senial y background;
- tiempo de entrenamiento.

## Resultado local

Baseline reproducido:

- `trial_id`: 0
- `best_ams`: 1.086077
- `roc_auc`: 0.909278
- `best_threshold`: 0.861100

Mejor trial de la busqueda:

- `trial_id`: 4
- `best_ams`: 1.131997
- `roc_auc`: 0.909088
- `best_threshold`: 0.857992
- `learning_rate`: 0.085
- `n_estimators`: 500
- `max_depth`: 4
- `min_child_weight`: 8
- `gamma`: 0.1
- `subsample`: 0.75
- `colsample_bytree`: 0.9
- `reg_lambda`: 5.0
- `reg_alpha`: 0.01
- `scale_pos_weight`: 1.25
- `max_delta_step`: 1

Archivos generados:

- `01_search_xgboost_i3_arboles_comet.py`: script reproducible de busqueda.
- `results/01_search_results.csv`: resultados completos.
- `results/01_search_results_sorted.csv`: resultados ordenados por `best_ams`.
- `results/01_best_params.json`: mejor configuracion local.
- `models/best_xgboost_i3_arboles.joblib`: mejor modelo del split local.

## Nota metodologica

La mejora es local sobre un unico split. Como ya vimos en experimentos anteriores, optimizar AMS en un split puede sobreajustar la politica de umbral o los hiperparametros. Antes de usarlo como candidato final fuerte conviene validar con CV o, si se busca comparacion Kaggle directa, reentrenar con la configuracion elegida y hacer submission controlada.
