## Consigna del problema

El proyecto se basa en el problema del Higgs Boson Machine Learning Challenge, que utiliza datos simulados del experimento ATLAS en CERN. El objetivo general es construir un modelo de aprendizaje automatico capaz de distinguir entre eventos asociados a la senal de interes, vinculada al decaimiento del boson de Higgs en dos particulas tau, y eventos de fondo (`background`) que no corresponden a esa senal.

Desde el punto de vista de aprendizaje automatico, se trata de un problema de clasificacion supervisada. A partir de un conjunto de caracteristicas calculadas para cada evento, el modelo debe aprender a separar los casos `signal` de los casos `background`, con el fin de mejorar la capacidad de deteccion de eventos relevantes dentro de una gran cantidad de observaciones.

## Que se quiere predecir o resolver

Cada fila del dataset representa un evento observado o simulado, descrito a partir de 30 variables extraidas de ese evento. A partir de esas caracteristicas, el modelo debe decidir si el evento pertenece a la clase `signal`, es decir, si esta asociado al proceso de interes del challenge, o a la clase `background`, que corresponde a eventos que no pertenecen a esa senal aunque puedan presentar caracteristicas similares.

## Tipo de problema

Desde el punto de vista de aprendizaje automatico, este problema corresponde a una tarea de clasificacion binaria supervisada. Esto se debe a que, durante el entrenamiento, cada evento tiene asociada una etiqueta conocida y el objetivo del modelo es decidir a cual de las dos clases pertenece: `signal` o `background`.

## Estructura general de los datos

Cada fila del dataset corresponde a un evento del experimento. En el conjunto de entrenamiento, cada evento incluye un identificador (`EventId`), 30 variables descriptivas, un peso asociado (`Weight`) y una etiqueta de clase (`Label`). En el conjunto de test, en cambio, no se incluye ni el peso ni la etiqueta. Las 30 variables se dividen en dos grandes grupos: las variables con prefijo `PRI`, que representan cantidades primitivas medidas por el detector, y las variables con prefijo `DER`, que son cantidades derivadas calculadas a partir de las anteriores. Segun la documentacion del challenge, `EventId`, `Weight` y `Label` tienen un rol especial y no deben usarse como variables de entrada del clasificador.

La mayoria de las variables son numericas continuas. Los angulos azimutales `phi` estan expresados en radianes, mientras que cantidades como energia, masa y momento estan dadas en GeV. Ademas, algunas variables pueden quedar indefinidas para ciertos eventos; en esos casos se utiliza el valor `-999.0`, que actua como un marcador especial y no como un valor fisico real. Esto ocurre, por ejemplo, en variables relacionadas con jets cuando el evento no contiene suficientes jets para calcularlas.

## Variable objetivo

La variable objetivo del problema es la columna `Label` presente en el conjunto de entrenamiento. Esta variable indica la clase de cada evento y toma dos posibles valores: `s` para `signal` y `b` para `background`. Por lo tanto, el objetivo del modelo es predecir esa etiqueta a partir de las variables descriptivas del evento.

## Restricciones explicitas y particularidades de la evaluacion

La consigna fija tanto el conjunto de datos como la forma general de evaluar la solucion. El archivo `training.csv` contiene 250000 eventos e incluye `EventId`, 30 variables descriptivas, `Weight` y `Label`. El archivo `test.csv` contiene 550000 eventos e incluye `EventId` y las mismas 30 variables, pero no contiene ni `Weight` ni `Label`. Por lo tanto, el problema debe resolverse respetando esa estructura fija y distinguiendo con claridad entre columnas auxiliares y variables de entrada.

La documentacion del challenge indica de forma explicita que `EventId`, `Weight` y `Label` no deben utilizarse como variables de entrada del clasificador. `EventId` funciona solo como identificador del evento, `Label` es la variable objetivo del entrenamiento y `Weight` es un peso asociado a cada evento que interviene en la evaluacion, pero no forma parte de las features. Ademas, algunas variables pueden tomar el valor `-999.0`, que no representa una medicion fisica real sino una variable indefinida o no computable en ese evento, por lo que ese valor requiere un tratamiento especial durante el analisis y el preprocesamiento.

Otra restriccion importante es que la evaluacion del challenge no esta centrada simplemente en `accuracy`, sino en la metrica AMS (`Approximate Median Significance`). En la documentacion tecnica, esta se define como:

`AMS = sqrt(2 * ((s + b + b_reg) * ln(1 + s / (b + b_reg)) - s))`

donde `s` representa la suma de pesos de los eventos `signal` seleccionados correctamente, `b` representa la suma de pesos de los eventos `background` seleccionados, y en el challenge se utiliza `b_reg = 10`. Esto implica que no alcanza con clasificar bien en promedio: la solucion debe priorizar una seleccion de eventos que mejore la significancia estadistica de la senal.

Finalmente, el formato de entrega tambien impone una condicion especifica. La submission no consiste solamente en asignar una clase final a cada evento, sino en generar un archivo con columnas `EventId`, `RankOrder` y `Class`. La columna `Class` indica la prediccion final (`s` o `b`), mientras que `RankOrder` ordena los eventos segun el score o nivel de confianza producido por el modelo. Esto refuerza la idea de que, ademas de clasificar, el modelo debe producir un criterio de ordenamiento util para la evaluacion del challenge.

## Ambiguedades de la consigna

Una ambiguedad importante es que el enunciado resumido del curso simplifica el problema al hablar de detectar eventos donde aparece la particula tau, mientras que la documentacion tecnica describe con mas precision una separacion entre eventos asociados a la senal de interes del canal Higgs -> tau tau y eventos de fondo. Por este motivo, para el resto del trabajo se tomara como referencia principal la formulacion tecnica del challenge y no una interpretacion simplificada del enunciado.
