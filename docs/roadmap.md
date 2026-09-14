# Roadmap de Trabajo

## Objetivo

Este documento define la forma de trabajo para abordar el problema de Higgs de manera academica, iterativa y experimental.

La idea no es implementar rapidamente un modelo final, sino construir una solucion paso a paso, discutiendo cada decision antes de codificarla, validando hipotesis y comparando alternativas de forma metodologicamente correcta.

## Principios de trabajo

- Leer y entender la consigna completa antes de implementar.
- No introducir codigo ni modelos sin acuerdo previo sobre el objetivo de la iteracion.
- Avanzar a partir de hipotesis pequenas y experimentos controlados.
- Separar claramente entendimiento del problema, preparacion de datos, modelado y evaluacion.
- Priorizar decisiones justificadas por evidencia y no por intuicion aislada.
- Documentar supuestos, riesgos metodologicos y conclusiones parciales.

## Roadmap

### 1. Entender la consigna y los datos

- Leer la consigna completa antes de tocar codigo.
- Identificar que se quiere predecir o resolver.
- Determinar si el problema es de:
  - clasificacion
  - regresion
  - clustering / no supervisado
  - otro
- Ver que representa cada fila del dataset.
- Ver que variable es la objetivo.
- Ver si hay restricciones explicitas:
  - metrica obligatoria
  - modelo obligatorio
  - conjunto de datos fijo
  - formato de entrega
- Anotar supuestos si la consigna tiene ambiguedades.

### 2. Cargar e inspeccionar los datos

- Cargar el dataset de forma reproducible.
- Ver dimensiones del dataset.
- Ver nombres de columnas.
- Ver tipos de datos.
- Ver primeras filas.
- Ver estadisticas descriptivas basicas.
- Ver cantidad de valores faltantes por columna.
- Ver categorias unicas si hay variables categoricas.
- Detectar columnas sospechosas:
  - IDs
  - columnas constantes
  - columnas casi vacias
  - columnas que puedan filtrar la respuesta

### 3. Hacer una exploracion inicial

- Graficar distribuciones de variables numericas.
- Ver histogramas.
- Ver boxplots si sirve.
- Ver distribucion de la variable objetivo.
- Si es clasificacion:
  - ver balance o desbalance de clases
- Si es regresion:
  - ver rango, sesgo, outliers y posibles topes o caps
- Hacer scatterplots entre variables importantes.
- Mirar correlaciones entre variables numericas.
- Buscar relaciones obvias entre features y target.
- Detectar outliers visualmente.
- Detectar escalas muy distintas entre variables.

### 4. Pensar el problema desde ML

- Definir que entra como input y que sale como output.
- Pensar si el problema requiere:
  - baseline simple
  - modelo lineal
  - modelo no lineal
  - regularizacion
  - transformacion de variables
- Pensar si hay pocos datos o muchos datos.
- Pensar si los datos parecen representativos.
- Revisar si puede haber data leakage.
- Revisar si la variable objetivo esta disponible solo despues del evento.

### 5. Separar correctamente los conjuntos

- Definir train, validation y test.
- Asegurar que test quede apartado desde el principio.
- Si corresponde, usar particion estratificada.
- Si hay validacion cruzada, definirla antes de entrenar.
- Verificar que ninguna transformacion use informacion de test.

### 6. Preparar los datos

- Tratar valores faltantes:
  - eliminar filas
  - eliminar columnas
  - imputar
- Tratar variables categoricas:
  - ordinal encoding
  - one-hot encoding
- Escalar variables si el modelo lo necesita.
- Considerar transformaciones:
  - log
  - normalizacion
  - estandarizacion
- Crear nuevas variables si aportan informacion.
- Eliminar variables irrelevantes o peligrosas.
- Dejar el preprocesamiento dentro de un pipeline cuando sea posible.

### 7. Elegir metricas correctas

- Si es clasificacion, decidir entre:
  - accuracy
  - precision
  - recall
  - F1
  - ROC-AUC
- Si es regresion, decidir entre:
  - MSE
  - RMSE
  - MAE
- Justificar la metrica segun el problema.
- Si hay desbalance, evitar depender solo de accuracy.

### 8. Entrenar un baseline

- Empezar por un modelo simple.
- Entrenar sin sobrecomplicar al inicio.
- Obtener una referencia inicial de desempeno.
- Guardar resultados para comparar despues.

### 9. Evaluar correctamente

- Evaluar en validation o con cross-validation.
- No tocar test hasta el final.
- Comparar train vs validation.
- Si es clasificacion:
  - matriz de confusion
  - precision y recall
  - curva ROC si aplica
- Si es regresion:
  - mirar errores promedio
  - mirar distribucion de residuos si aporta

### 10. Analizar errores

- Ver donde se equivoca el modelo.
- Identificar patrones de error.
- Revisar ejemplos concretos mal clasificados o mal predichos.
- Ver si ciertos grupos o rangos fallan mas.
- Ver si los errores sugieren:
  - falta de features
  - mala limpieza
  - mal modelo
  - desbalance
  - overfitting
  - underfitting

### 11. Mejorar iterativamente

- Probar mejor preprocesamiento.
- Probar nuevas features.
- Probar scaling si faltaba.
- Probar otro modelo.
- Ajustar hiperparametros.
- En clasificacion, ajustar umbral si tiene sentido.
- Repetir evaluacion y comparar de forma justa.

### 12. Diagnosticar overfitting y underfitting

- Comparar desempeno en train y validation.
- Usar curvas de aprendizaje si corresponde.
- Si train muy bien y validation mal:
  - sospechar overfitting
- Si ambos mal:
  - sospechar underfitting
- Decidir acciones:
  - simplificar modelo
  - complejizar modelo
  - regularizar
  - conseguir mas datos
  - mejorar features

### 13. Regularizar cuando haga falta

- Probar Ridge.
- Probar Lasso.
- Probar Elastic Net.
- Justificar por que regularizar ayuda en este caso.
- No usar regularizacion porque si.

### 14. Elegir modelo final

- Elegir el mejor modelo segun validation.
- Reentrenar con todos los datos de entrenamiento disponibles.
- Evaluar una sola vez en test.
- Reportar la metrica final con claridad.

### 15. Redactar conclusiones

- Explicar que se hizo.
- Explicar por que se eligio ese pipeline.
- Explicar que modelo rindio mejor.
- Explicar limitaciones.
- Explicar posibles mejoras futuras.
- Aclarar riesgos metodologicos si los hubo.

## Como usar este roadmap

Cada iteracion del proyecto deberia apoyarse en este esquema:

1. Definir la pregunta concreta a responder.
2. Acordar que parte del roadmap corresponde trabajar.
3. Discutir opciones y tradeoffs antes de implementar.
4. Ejecutar un experimento acotado.
5. Registrar resultados y decidir el siguiente paso.

## Nota para este proyecto

En Higgs no alcanza con reutilizar mecanicamente un pipeline de un taller anterior. Antes de decidir modelos hay que estudiar con cuidado:

- la consigna especifica del proyecto;
- la metrica de evaluacion obligatoria;
- el rol de columnas especiales del dataset;
- la estrategia de validacion mas adecuada;
- los riesgos de leakage y de comparaciones injustas entre experimentos.

Este documento se toma como guia de trabajo viva y puede refinarse a medida que avancemos.
