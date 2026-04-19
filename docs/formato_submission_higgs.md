# Formato de submission para Kaggle: Higgs Boson

Este documento deja registrado el formato del archivo de submission que usamos en el notebook [NN_extendido.ipynb](/home/santig14/fing/taa/1_TallerAA/5.%20Experimentacion%20NN/NN_extendido.ipynb) para la competencia `higgs-boson`.

## Estructura del CSV

El archivo final debe tener exactamente estas columnas:

| Columna | Tipo esperado | Descripción |
| --- | --- | --- |
| `EventId` | entero | Identificador del evento del conjunto de test. |
| `RankOrder` | entero | Ranking único de cada fila. |
| `Class` | string | Predicción final, con valores `s` o `b`. |

## Reglas prácticas

- `EventId` sale directo del set de test provisto por la competencia.
- `RankOrder` debe ser un ranking sin repetidos.
- `Class` usa `s` para señal y `b` para background.
- En el archivo de referencia de la competencia, `RankOrder` recorre `1..550000`.

## Ejemplo mínimo

```csv
EventId,RankOrder,Class
350000,1,b
350001,2,b
350002,3,s
```

## Criterio usado en `NN_extendido.ipynb`

La sección final del notebook hace lo siguiente:

1. Toma el mejor modelo de `comparison_summary`.
2. Calcula probabilidades sobre validación y test.
3. Ajusta un umbral sobre validación maximizando AMS ponderado con los pesos de `train_weight.csv`.
4. Genera `Class` con `s` si el score es mayor o igual al umbral óptimo, y `b` en caso contrario.
5. Genera `RankOrder` a partir del score del modelo usando un ranking único.
6. Guarda el CSV en `results/` con timestamp.
7. Intenta enviarlo a Kaggle usando `keys/kaggle.json`.

## Convención actual del notebook

La generación del submission hoy sigue esta lógica:

```python
submission_df = pd.DataFrame(
    {
        "EventId": test_event_id.astype(int),
        "RankOrder": pd.Series(test_scores).rank(
            method="first",
            ascending=True,
        ).astype(int),
        "Class": np.where(test_scores >= best_threshold, "s", "b"),
    }
)
```

Notas:

- `method="first"` asegura que `RankOrder` no tenga empates.
- `ascending=True` deja rango `1` para el score más bajo y el valor máximo para el score más alto.
- El umbral de corte no se fija manualmente: se calibra en validación con AMS ponderado.

## Ubicaciones relevantes

- Notebook que genera el submission: [NN_extendido.ipynb](/home/santig14/fing/taa/1_TallerAA/5.%20Experimentacion%20NN/NN_extendido.ipynb)
- Archivo de ejemplo de la competencia: `data/higgs-boson/random_submission.csv`
- Credenciales locales de Kaggle: `keys/kaggle.json`
