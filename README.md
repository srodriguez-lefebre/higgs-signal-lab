# Higgs Boson Machine Learning Challenge

Proyecto académico de clasificación desarrollado para **Taller de Aprendizaje Automático**, Facultad de Ingeniería, Universidad de la República (2026).

El objetivo es distinguir eventos de señal asociados al bosón de Higgs de eventos de fondo a partir de datos simulados del detector ATLAS. El trabajo recorre el proceso completo: exploración de datos, preprocesamiento, entrenamiento, validación, selección del umbral de decisión e interpretación del modelo.

## Metodología

- Tratamiento del marcador de datos faltantes `-999` e incorporación de indicadores de ausencia.
- Ingeniería de características y comparación de distintos pipelines de preprocesamiento.
- Evaluación de regresión logística, Random Forest, redes neuronales y XGBoost.
- Validación cruzada estratificada de 5 folds y predicciones *out-of-fold*.
- Selección del punto de operación con la métrica AMS (*Approximate Median Significance*).
- Análisis de sobreajuste, ajuste de hiperparámetros y combinación de modelos.
- Interpretación de las predicciones mediante SHAP.

## Solución final

La solución seleccionada utiliza el pipeline de preprocesamiento `P04`, un ensamble de cinco modelos XGBoost entrenados por validación cruzada y una política de decisión obtenida a partir de predicciones *out-of-fold*. La entrega final alcanzó un **AMS público de 3.61216 en Kaggle**.

El notebook principal parte de los archivos originales, reconstruye el preprocesamiento, entrena el ensamble y genera la entrega sin depender de artefactos intermedios:

- [Pipeline final](entrega/pipeline_final_taller_1.ipynb)
- [Pipeline con registro opcional en Comet](entrega/pipeline_final_taller_1_experimentos_comet.ipynb)
- [Informe académico](informe/informe_final.pdf)
- [Presentación](presentacion/presentacion_boson_higgs.pdf)

## Estructura del repositorio

```text
.
├── entrega/          # Notebooks reproducibles de la solución final
├── informe/          # Informe, fuente LaTeX, figuras y notebooks auxiliares
├── presentacion/     # Presentación en PDF y versión HTML
├── experimentos/     # Evolución del trabajo y pruebas intermedias
├── docs/             # Consigna, roadmap y resumen de resultados
└── requirements.txt  # Dependencias principales
```

La carpeta `experimentos/` conserva la evolución del proyecto para mantener la trazabilidad de las decisiones. Su numeración refleja el orden de las etapas de trabajo; hay un [índice resumido](experimentos/README.md).

## Reproducción

Se requiere Python 3.10 o posterior y una cuenta de Kaggle con acceso a la competencia `higgs-boson`.

```bash
python -m venv .venv
pip install -r requirements.txt
jupyter lab
```

Luego se debe abrir `entrega/pipeline_final_taller_1.ipynb` y ejecutar sus celdas. El notebook busca `training.csv` y `test.csv` en su directorio de ejecución; si no existen, puede descargarlos mediante la API de Kaggle. Los datos, modelos y resultados generados no se incluyen en el repositorio.

## Referencias

- C. Adam-Bourdarios et al., *Learning to Discover: the Higgs Boson Machine Learning Challenge*, 2014.
- [Higgs Boson Machine Learning Challenge — Kaggle](https://www.kaggle.com/c/higgs-boson)
