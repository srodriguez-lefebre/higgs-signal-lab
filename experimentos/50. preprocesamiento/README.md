# 50. preprocesamiento

Nuevo punto de arranque para rehacer el preprocesamiento del proyecto con el conocimiento acumulado de los experimentos anteriores.

Objetivo de esta carpeta:
- diseñar nuevas variantes de preprocesamiento;
- documentar decisiones metodológicas;
- centralizar notebooks y notas de esta nueva etapa.

Convención inicial:
- los notebooks y notas de trabajo viven acá;
- los artefactos procesados se guardan en `data/preprocesor/`.



PLAN:

Te dejo una **primera versión de especificación en Markdown**, pensada para que el equipo implemente sin tomar decisiones libres. Separé:

* `C00`: procedimiento común obligatorio para todos.
* `P00` a `P04`: variantes de preprocesamiento experimental.

La idea metodológica sigue el enfoque del curso/libro: separar preparación de datos, limpieza, manejo de faltantes, escalado, pipelines y validación; además, el Taller 5 pide explícitamente repensar el pipeline de preprocesamiento usado previamente.  

---

# Especificación de preprocesamiento para Higgs Boson Challenge

## 0. Principios generales

El problema es de **clasificación binaria supervisada**. Cada evento debe clasificarse como:

```text
s = signal
b = background
```

El dataset proviene de simulaciones oficiales de ATLAS para eventos `Higgs -> tau tau` mezclados con fondos físicos que pueden imitar esa señal. CERN describe que los eventos tienen pesos asociados a la simulación, pero esos pesos **no forman parte de la entrada del clasificador**. Se usan para evaluación, especialmente para AMS. ([CERN Open Data Portal][1])

La métrica principal del desafío es **AMS**, que depende de la suma ponderada de verdaderos positivos `s` y falsos positivos `b`, con regularización `b_r = 10`. Por eso, `Weight` debe usarse para evaluar AMS, pero no como feature. ([CERN Open Data Portal][1])

---

# C00 — Procedimiento común obligatorio para todos los pipelines

## Objetivo

Construir una representación canónica del dataset, común a todos los experimentos, sin aprender estadísticas del dataset completo y sin introducir data leakage.

Este bloque **no imputa**, **no escala**, **no aplica log-transform**, **no hace one-hot encoding** y **no selecciona features**. Solo separa roles de columnas, convierte el código artificial de faltantes y deja los datos listos para que cada pipeline se ajuste dentro de cada fold.

---

## Columnas especiales

Estas columnas nunca deben usarse como features:

```text
EventId
Weight
Label
KaggleSet
KaggleWeight
```

Reglas:

| Columna        | Uso permitido                                | Uso prohibido       |
| -------------- | -------------------------------------------- | ------------------- |
| `EventId`      | identificar y recombinar predicciones        | entrenar modelos    |
| `Label`        | construir `y`                                | entrenar como input |
| `Weight`       | calcular AMS y opcionalmente `sample_weight` | usar como feature   |
| `KaggleSet`    | análisis del origen del dato, si existe      | usar como feature   |
| `KaggleWeight` | reproducir scores Kaggle, si corresponde     | usar como feature   |

CERN/Open Data indica explícitamente que `Weight` es un artefacto de simulación y no es input del clasificador; también aclara que en la versión Open Data labels y pesos están disponibles para todo el dataset. ([CERN Open Data Portal][1])

---

## Features originales permitidas

Todas las columnas siguientes son candidatas a features:

```text
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
DER_pt_h
DER_deltaeta_jet_jet
DER_mass_jet_jet
DER_prodeta_jet_jet
DER_deltar_tau_lep
DER_pt_tot
DER_sum_pt
DER_pt_ratio_lep_tau
DER_met_phi_centrality
DER_lep_eta_centrality
PRI_tau_pt
PRI_tau_eta
PRI_tau_phi
PRI_lep_pt
PRI_lep_eta
PRI_lep_phi
PRI_met
PRI_met_phi
PRI_met_sumet
PRI_jet_num
PRI_jet_leading_pt
PRI_jet_leading_eta
PRI_jet_leading_phi
PRI_jet_subleading_pt
PRI_jet_subleading_eta
PRI_jet_subleading_phi
PRI_jet_all_pt
```

La documentación del desafío define variables `DER_*` como derivadas físicas y variables `PRI_*` como cantidades primitivas del evento; también especifica que `PRI_jet_num` toma valores `0`, `1`, `2`, `3`, con valores mayores capados en `3`. ([CERN Open Data Portal][1])

---

## Conversión obligatoria de faltantes

En las columnas de features, todo valor:

```text
-999.0
```

debe convertirse a:

```text
NaN
```

Razón: `-999.0` no es un valor físico real. La documentación indica que cuando una variable no tiene sentido o no puede computarse, se asigna `-999.0`, fuera del rango normal de las variables. 

Implementación obligatoria:

```python
SPECIAL_COLUMNS = ["EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"]

feature_cols = [c for c in df.columns if c not in SPECIAL_COLUMNS]

df[feature_cols] = df[feature_cols].replace(-999.0, np.nan)
```

Esta conversión puede hacerse antes de los folds porque es una transformación determinista que no aprende estadísticas.

---

## Target

Si existe `Label`, mapear:

```python
y = df["Label"].map({"b": 0, "s": 1}).astype(np.int8)
```

Convención obligatoria:

```text
0 = background
1 = signal
```

---

## Pesos

Si existe `Weight`:

```python
w = df["Weight"].astype(np.float64)
```

Regla obligatoria:

```text
Weight no se usa como feature.
Weight se conserva para AMS.
Weight puede pasarse como sample_weight solo si el experimento lo declara.
```

---

## Validaciones obligatorias

Antes de entregar `X`, `y`, `w`, el bloque común debe verificar:

```text
1. No hay columnas especiales dentro de X.
2. PRI_jet_num existe.
3. PRI_jet_num solo contiene valores 0, 1, 2, 3.
4. Todas las features son numéricas.
5. Label, si existe, solo contiene "s" o "b".
6. EventId, si existe, se conserva fuera de X.
7. No se eliminaron filas.
```

---

## Salida del procedimiento común

El procedimiento común debe devolver:

```text
X_raw
y
w
event_id
metadata
```

Donde:

```text
X_raw:
    DataFrame solo con features, con -999 convertido a NaN.

y:
    vector binario 0/1, si Label existe.

w:
    pesos del evento, si Weight existe.

event_id:
    identificador del evento, si EventId existe.

metadata:
    diccionario con:
        - feature_cols
        - special_cols_detected
        - n_rows
        - n_features
        - missing_counts_by_column
        - pri_jet_num_distribution
```

---

# Reglas comunes dentro de cada fold

Todo pipeline `P00` a `P04` debe respetar estas reglas.

## Orden obligatorio por fold

Para cada fold:

```text
1. Recibir X_train_raw, X_valid_raw.
2. Ajustar el preprocesamiento solo con X_train_raw.
3. Transformar X_train_raw.
4. Transformar X_valid_raw usando los objetos ajustados en train.
5. Entrenar el modelo solo con X_train_processed.
6. Evaluar en X_valid_processed.
```

No se permite calcular medianas, medias, desvíos, cuantiles, scalers, selección de columnas ni thresholds mirando todo el dataset.

---

## Indicadores de faltantes

Los indicadores `is_missing_<columna>` se crean mirando `X_train_raw` después de `-999 -> NaN` y antes de imputar.

Regla:

```python
missing_indicator_cols = [
    col for col in X_train_raw.columns
    if X_train_raw[col].isna().any()
]
```

Para cada columna de esa lista:

```python
X_train[f"is_missing_{col}"] = X_train_raw[col].isna().astype(np.float32)
X_valid[f"is_missing_{col}"] = X_valid_raw[col].isna().astype(np.float32)
```

Si una columna no tiene faltantes en train pero sí en valid, **no se crea indicador nuevo en valid**. La estructura de columnas la define train.

---

## Columnas con log-transform

La transformación logarítmica debe ser:

```python
np.log1p(x)
```

No se usa `np.log(x)`.

Columnas base con log-transform:

```text
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
DER_pt_h
DER_deltaeta_jet_jet
DER_mass_jet_jet
DER_deltar_tau_lep
DER_pt_tot
DER_sum_pt
DER_pt_ratio_lep_tau
PRI_tau_pt
PRI_lep_pt
PRI_met
PRI_met_sumet
PRI_jet_leading_pt
PRI_jet_subleading_pt
PRI_jet_all_pt
```

No aplicar log a:

```text
PRI_tau_eta
PRI_tau_phi
PRI_lep_eta
PRI_lep_phi
PRI_met_phi
PRI_jet_leading_eta
PRI_jet_leading_phi
PRI_jet_subleading_eta
PRI_jet_subleading_phi
DER_prodeta_jet_jet
DER_met_phi_centrality
DER_lep_eta_centrality
PRI_jet_num
indicadores is_missing_*
one-hot encodings
```

Si una columna seleccionada para `log1p` tiene un valor no faltante menor a `0`, el pipeline debe fallar con error explícito. No se permite corregir silenciosamente.

---

# P00 — Baseline global: log + mediana + StandardScaler + missing flags

## Objetivo

Construir el baseline principal. Este pipeline representa una solución simple, global y comparable.

---

## Pasos

Dentro de cada fold:

```text
1. Copiar X_train_raw y X_valid_raw.
2. Detectar columnas con faltantes en X_train_raw.
3. Crear indicadores is_missing_<columna>.
4. Aplicar log1p a las columnas definidas.
5. Imputar todas las features numéricas con mediana.
6. Escalar features numéricas continuas con StandardScaler.
7. No escalar indicadores de faltantes.
8. Castear salida final a float32.
```

---

## Tratamiento de `PRI_jet_num`

En `P00`, `PRI_jet_num` se mantiene como variable numérica.

Regla:

```text
PRI_jet_num se conserva.
PRI_jet_num se escala con StandardScaler.
No se hace one-hot.
```

Esto es intencional: `P00` es el baseline simple.

---

## Imputación

Usar mediana:

```python
median_imputer = SimpleImputer(strategy="median")
```

La mediana se calcula después del `log1p`, solo con `X_train`.

---

## Escalado

Usar:

```python
StandardScaler()
```

Se ajusta solo con train.

Columnas escaladas:

```text
todas las columnas numéricas continuas,
incluyendo PRI_jet_num,
excluyendo is_missing_*.
```

---

## Salida

```text
X_train_P00
X_valid_P00
metadata_P00
```

`metadata_P00` debe guardar:

```text
missing_indicator_cols
log_cols
median_values
standard_scaler_mean
standard_scaler_scale
final_feature_order
```

---

# P01 — Global + `PRI_jet_num` como categórica

## Objetivo

Probar si mejora el rendimiento al tratar `PRI_jet_num` como variable estructural/categórica en lugar de numérica ordinal.

Esto se justifica porque `PRI_jet_num` toma valores `0`, `1`, `2`, `3`, y `3` significa “3 o más jets”, no exactamente tres. Además, varias variables de jets están definidas o indefinidas según esa columna. ([CERN Open Data Portal][1])

---

## Diferencias contra P00

`P01` es igual a `P00`, excepto por `PRI_jet_num`.

---

## Tratamiento de `PRI_jet_num`

Eliminar la columna original:

```python
X = X.drop(columns=["PRI_jet_num"])
```

Crear siempre estas cuatro columnas, en este orden:

```text
PRI_jet_num_0
PRI_jet_num_1
PRI_jet_num_2
PRI_jet_num_3
```

Reglas:

```text
PRI_jet_num_0 = 1 si PRI_jet_num == 0, si no 0
PRI_jet_num_1 = 1 si PRI_jet_num == 1, si no 0
PRI_jet_num_2 = 1 si PRI_jet_num == 2, si no 0
PRI_jet_num_3 = 1 si PRI_jet_num == 3, si no 0
```

Estas columnas:

```text
no se imputan
no se escalan
no se transforman con log
se casteen a float32
```

---

## Pasos completos

```text
1. Copiar X_train_raw y X_valid_raw.
2. Guardar PRI_jet_num original.
3. Detectar columnas con faltantes en X_train_raw, excluyendo PRI_jet_num.
4. Crear indicadores is_missing_<columna>.
5. Eliminar PRI_jet_num.
6. Crear one-hot fijo de PRI_jet_num.
7. Aplicar log1p a columnas logarítmicas.
8. Imputar columnas continuas con mediana.
9. Escalar columnas continuas con StandardScaler.
10. Concatenar continuas escaladas + indicadores + one-hot.
11. Castear salida final a float32.
```

---

## Salida

`metadata_P01` debe guardar:

```text
missing_indicator_cols
log_cols
one_hot_cols = [PRI_jet_num_0, PRI_jet_num_1, PRI_jet_num_2, PRI_jet_num_3]
median_values
standard_scaler_mean
standard_scaler_scale
final_feature_order
```

---

# P02 — Global robusto: one-hot + mediana + RobustScaler

## Objetivo

Probar una variante más robusta frente a outliers y colas pesadas.

La diferencia principal con `P01` es que el escalado no usa media y desvío estándar, sino mediana e IQR. Esto puede ser útil porque muchas variables físicas de energía, momento y masa pueden tener distribuciones asimétricas o con colas largas.

---

## Base

`P02` es igual a `P01`, salvo por el escalado.

---

## Escalado

Usar:

```python
RobustScaler(
    with_centering=True,
    with_scaling=True,
    quantile_range=(25.0, 75.0)
)
```

Reglas:

```text
Se ajusta solo con X_train.
Se aplica a X_train y X_valid.
No se aplica a indicadores is_missing_*.
No se aplica a one-hot de PRI_jet_num.
```

---

## Pasos completos

```text
1. Copiar X_train_raw y X_valid_raw.
2. Guardar PRI_jet_num original.
3. Detectar columnas con faltantes en X_train_raw, excluyendo PRI_jet_num.
4. Crear indicadores is_missing_<columna>.
5. Eliminar PRI_jet_num.
6. Crear one-hot fijo de PRI_jet_num.
7. Aplicar log1p a columnas logarítmicas.
8. Imputar columnas continuas con mediana.
9. Escalar columnas continuas con RobustScaler.
10. Concatenar continuas escaladas + indicadores + one-hot.
11. Castear salida final a float32.
```

---

## Salida

`metadata_P02` debe guardar:

```text
missing_indicator_cols
log_cols
one_hot_cols
median_values
robust_scaler_center
robust_scaler_scale
final_feature_order
```

---

# P03 — Modelos separados por grupo de jets

## Objetivo

Evitar que un único modelo global tenga que aprender artificialmente subespacios con variables estructuralmente indefinidas.

La documentación del desafío indica que más del 70% de los eventos carecen de al menos una feature, que esos faltantes no son fallas de simulación sino ausencias estructurales, y que todos los faltantes están relacionados con `PRI_jet_num`, excepto `DER_mass_MMC`. 

También se advierte que imputar faltantes estructurales puede producir completaciones sin significado y perjudicar la clasificación. 

---

## Grupos

Se deben construir tres grupos:

```text
G0: PRI_jet_num == 0
G1: PRI_jet_num == 1
G2: PRI_jet_num >= 2
```

Cada grupo tiene:

```text
su propio preprocesador
su propio modelo
sus propias medianas
su propio scaler
su propio orden de columnas
```

---

## G0 — Eventos sin jets

Filtro:

```python
X_group = X_raw[X_raw["PRI_jet_num"] == 0]
```

Columnas a eliminar obligatoriamente:

```text
PRI_jet_num
DER_deltaeta_jet_jet
DER_mass_jet_jet
DER_prodeta_jet_jet
DER_lep_eta_centrality
PRI_jet_leading_pt
PRI_jet_leading_eta
PRI_jet_leading_phi
PRI_jet_subleading_pt
PRI_jet_subleading_eta
PRI_jet_subleading_phi
```

Justificación: para `PRI_jet_num = 0` no existe leading jet ni subleading jet, y las variables de dos jets no están definidas. La documentación define explícitamente que las variables de leading jet están indefinidas si `PRI_jet_num = 0`, y las de subleading jet o dos jets si `PRI_jet_num <= 1`. ([CERN Open Data Portal][1])

---

## G1 — Eventos con un jet

Filtro:

```python
X_group = X_raw[X_raw["PRI_jet_num"] == 1]
```

Columnas a eliminar obligatoriamente:

```text
PRI_jet_num
DER_deltaeta_jet_jet
DER_mass_jet_jet
DER_prodeta_jet_jet
DER_lep_eta_centrality
PRI_jet_subleading_pt
PRI_jet_subleading_eta
PRI_jet_subleading_phi
```

Se conservan:

```text
PRI_jet_leading_pt
PRI_jet_leading_eta
PRI_jet_leading_phi
PRI_jet_all_pt
```

---

## G2 — Eventos con dos o más jets

Filtro:

```python
X_group = X_raw[X_raw["PRI_jet_num"] >= 2]
```

Tratamiento de `PRI_jet_num`:

```text
Eliminar PRI_jet_num.
Crear PRI_jet_num_is_3.
```

Regla:

```python
PRI_jet_num_is_3 = 1 si PRI_jet_num == 3, si no 0
```

Justificación: `3` significa “3 o más jets”, porque valores mayores fueron capados en `3`; por tanto, dentro del grupo `>=2`, distinguir `2` contra `3+` puede aportar información estructural. ([CERN Open Data Portal][1])

---

## Preprocesamiento dentro de cada grupo

Para cada grupo `G0`, `G1`, `G2`, aplicar:

```text
1. Detectar faltantes en X_train_group.
2. Crear indicadores is_missing_<columna>.
3. Aplicar log1p a las columnas logarítmicas que existan en ese grupo.
4. Imputar con mediana calculada solo en train del grupo.
5. Escalar columnas continuas con StandardScaler.
6. No escalar indicadores.
7. No escalar PRI_jet_num_is_3.
8. Castear a float32.
```

---

## Recombinar predicciones

Durante validación:

```text
1. Separar X_valid_raw en G0, G1, G2.
2. Aplicar el preprocesador correspondiente.
3. Obtener scores del modelo correspondiente.
4. Recombinar scores usando el índice original o EventId.
5. Calcular métricas sobre todo el fold recombinado.
```

Regla obligatoria:

```text
El threshold AMS se calcula globalmente sobre los scores recombinados, no por grupo.
```

La optimización de thresholds por grupo se deja para otro experimento posterior, no para `P03`.

---

# P04 — Pipeline cargado con feature engineering explícito

## Objetivo

Agregar variables físicas derivadas simples, calculadas fila a fila, sin usar labels ni estadísticas del dataset.

Este pipeline busca capturar relaciones entre momentos, masas, ángulos, energía faltante y jets. Las variables originales ya incluyen cantidades físicas relevantes como masas, momentos transversos, pseudorapidez, ángulos, energía faltante y jets. ([CERN Open Data Portal][1])

---

## Base de P04

`P04` parte de `P02`:

```text
- -999 ya convertido a NaN por C00
- PRI_jet_num con one-hot
- indicadores de missing
- log1p en magnitudes positivas
- imputación por mediana
- RobustScaler
```

Además agrega las features de esta sección.

---

## Regla general para features nuevas

Todas las features nuevas se calculan:

```text
después de C00
antes de log-transform
antes de imputación
antes de escalado
```

Si una feature usa una columna `NaN`, el resultado queda `NaN` y luego se maneja con indicadores/imputación.

No se permite calcular features usando:

```text
Label
Weight
KaggleWeight
KaggleSet
EventId
```

---

## Constantes

Usar:

```python
EPS = 1e-6
PI = np.pi
```

Para diferencia angular circular:

```python
def abs_delta_phi(phi_a, phi_b):
    return np.abs((phi_a - phi_b + np.pi) % (2 * np.pi) - np.pi)
```

---

## Features nuevas obligatorias

### 1. Diferencias angulares entre tau, lepton y MET

```python
FE_abs_dphi_tau_lep = abs_delta_phi(PRI_tau_phi, PRI_lep_phi)
FE_abs_dphi_tau_met = abs_delta_phi(PRI_tau_phi, PRI_met_phi)
FE_abs_dphi_lep_met = abs_delta_phi(PRI_lep_phi, PRI_met_phi)
```

Justificación: los ángulos azimutales de tau, lepton y energía faltante son variables primitivas disponibles; su relación angular puede aportar información sobre la topología del evento. CERN lista estas variables como `PRI_tau_phi`, `PRI_lep_phi` y `PRI_met_phi`. ([CERN Open Data Portal][1])

---

### 2. Separación en pseudorapidez tau-lepton

```python
FE_abs_deta_tau_lep = abs(PRI_tau_eta - PRI_lep_eta)
```

Justificación: `PRI_tau_eta` y `PRI_lep_eta` describen pseudorapidez de tau y lepton. Aunque existe `DER_deltar_tau_lep`, separar el componente de pseudorapidez permite que el modelo use esta dimensión de forma explícita. ([CERN Open Data Portal][1])

---

### 3. Suma de momento transversal tau-lepton

```python
FE_tau_lep_pt_sum = PRI_tau_pt + PRI_lep_pt
```

Justificación: `PRI_tau_pt` y `PRI_lep_pt` son momentos transversos de las partículas visibles principales. La suma representa una escala simple de actividad visible del par tau-lepton. ([CERN Open Data Portal][1])

---

### 4. Fracción de momento transversal del tau

```python
FE_tau_pt_fraction = PRI_tau_pt / (PRI_tau_pt + PRI_lep_pt + EPS)
```

Justificación: captura el reparto relativo de momento transversal entre tau y lepton. Es una versión acotada y estable de una relación entre ambas partículas.

---

### 5. MET relativo al sistema tau-lepton

```python
FE_met_over_tau_lep_pt = PRI_met / (PRI_tau_pt + PRI_lep_pt + EPS)
```

Justificación: `PRI_met` representa energía transversa faltante. Relacionarla con la escala visible tau-lepton puede ayudar a distinguir eventos con distinta cantidad relativa de energía no observada. ([CERN Open Data Portal][1])

---

### 6. MET relativo a la energía transversa total

```python
FE_met_over_sumet = PRI_met / (PRI_met_sumet + EPS)
```

Justificación: `PRI_met_sumet` es la energía transversa total del detector. Esta feature mide qué fracción de la actividad total aparece como energía faltante. ([CERN Open Data Portal][1])

---

### 7. Balance vectorial relativo

```python
FE_pt_balance = DER_pt_tot / (DER_sum_pt + EPS)
```

Justificación: `DER_pt_tot` es el módulo de una suma vectorial, mientras que `DER_sum_pt` es una suma escalar de momentos. El cociente aproxima cuán balanceado o desbalanceado está el evento. CERN define ambas variables dentro de las variables derivadas. 

---

### 8. Masa MMC relativa a masa visible

```python
FE_mmc_over_visible_mass = DER_mass_MMC / (DER_mass_vis + EPS)
```

Justificación: `DER_mass_MMC` estima la masa del candidato Higgs y `DER_mass_vis` mide la masa visible del sistema tau-lepton. Su relación puede capturar discrepancias entre masa reconstruida y masa visible. 

---

### 9. Masa transversa relativa a masa visible

```python
FE_transverse_mass_over_visible_mass = DER_mass_transverse_met_lep / (DER_mass_vis + EPS)
```

Justificación: combina una masa transversa que involucra MET y lepton con la masa visible tau-lepton. Puede aportar una escala relativa entre información visible y faltante.

---

### 10. Fracción de actividad asociada a jets

```python
FE_jet_all_pt_fraction = PRI_jet_all_pt / (DER_sum_pt + EPS)
```

Justificación: `PRI_jet_all_pt` es la suma escalar del momento transversal de todos los jets, mientras que `DER_sum_pt` resume momentos de tau, lepton y jets. El cociente mide cuánto peso tienen los jets en la actividad total del evento. ([CERN Open Data Portal][1])

---

### 11. Fracción del leading jet

```python
FE_leading_jet_pt_fraction = PRI_jet_leading_pt / (PRI_jet_all_pt + EPS)
```

Justificación: cuando hay jets, mide cuánto domina el jet principal dentro de la actividad total de jets. Queda `NaN` si no hay leading jet.

---

### 12. Fracción del subleading jet

```python
FE_subleading_jet_pt_fraction = PRI_jet_subleading_pt / (PRI_jet_all_pt + EPS)
```

Justificación: complementa la anterior para eventos con al menos dos jets. Queda `NaN` si no hay subleading jet.

---

### 13. Balance entre leading y subleading jet

```python
FE_jet_pt_balance = (
    PRI_jet_leading_pt - PRI_jet_subleading_pt
) / (
    PRI_jet_leading_pt + PRI_jet_subleading_pt + EPS
)
```

Justificación: para eventos con dos o más jets, mide si ambos jets tienen momentos similares o si uno domina claramente.

---

### 14. Diferencia angular entre leading y subleading jet

```python
FE_abs_dphi_leading_subleading_jet = abs_delta_phi(
    PRI_jet_leading_phi,
    PRI_jet_subleading_phi
)
```

Justificación: agrega una relación angular entre jets que no aparece directamente como columna primitiva simple. Queda `NaN` cuando no existen dos jets.

---

### 15. Masa de dos jets relativa a actividad total

```python
FE_mass_jet_over_sum_pt = DER_mass_jet_jet / (DER_sum_pt + EPS)
```

Justificación: `DER_mass_jet_jet` solo está definida si hay al menos dos jets. Relacionarla con la actividad total puede estabilizar su escala y aportar información relativa del sistema de jets. 

---

## Log-transform en P04

Además de las columnas base con `log1p`, aplicar `log1p` a:

```text
FE_tau_lep_pt_sum
```

No aplicar log a:

```text
features angulares
features de diferencia eta
fracciones
ratios
balances
```

Razón: muchas de esas features ya están acotadas, pueden ser negativas o tienen interpretación relativa.

---

## Preprocesamiento posterior en P04

Después de crear features:

```text
1. Detectar columnas con faltantes en X_train.
2. Crear indicadores is_missing_<columna>.
3. Aplicar log1p a columnas definidas.
4. Imputar continuas con mediana.
5. One-hot fijo de PRI_jet_num.
6. Escalar continuas con RobustScaler.
7. No escalar indicadores.
8. No escalar one-hot.
9. Castear salida final a float32.
```

---

## No hacer en P04

Para no mezclar decisiones, `P04` no debe hacer:

```text
selección automática de features
PCA
eliminación de outliers
oversampling
undersampling
SMOTE
uso de Weight como feature
thresholds por grupo de jets
```

Si se quiere probar cualquiera de esas decisiones, debe abrirse otro pipeline o ablation separado.

---

# Reglas de evaluación comunes a todos los pipelines

Aunque esta especificación se concentra en preprocesamiento, cada experimento debe reportar como mínimo:

```text
AUC ROC
Precision
Recall
F1
AMS máximo en validación
Threshold que maximiza AMS
Porcentaje de eventos clasificados como signal
s ponderado
b ponderado
```

AMS debe calcularse usando `Weight`, porque la definición oficial usa pesos para sumar verdaderos positivos y falsos positivos ponderados. ([CERN Open Data Portal][1])

---

# Registro obligatorio por experimento

Cada corrida debe registrar:

```text
Experiment ID:
Fecha:
Responsable:
Pipeline: C00 + PXX
Modelo:
Hiperparámetros:
Seed:
Fold strategy:
Usa sample_weight en entrenamiento: sí/no
Usa Weight como feature: NO
Features finales:
Cantidad de features finales:
AUC valid:
AMS valid:
Threshold AMS:
Precision:
Recall:
F1:
Observaciones de overfitting:
Errores o advertencias:
Conclusión:
```

---

# Resumen operativo

La especificación queda así:

| ID    | Nombre                          | Idea principal                                    |
| ----- | ------------------------------- | ------------------------------------------------- |
| `C00` | común obligatorio               | separar columnas, mapear target, `-999 -> NaN`    |
| `P00` | baseline global                 | log + mediana + StandardScaler + missing flags    |
| `P01` | baseline + jets categóricos     | P00 + one-hot de `PRI_jet_num`                    |
| `P02` | robusto                         | P01 + RobustScaler                                |
| `P03` | modelos por grupo de jets       | un preprocesador/modelo para `0`, `1`, `>=2` jets |
| `P04` | cargado con feature engineering | P02 + features físicas explícitas                 |

Mi recomendación es implementar en este orden:

```text
C00 -> P00 -> P01 -> P02 -> P03 -> P04
```

Así primero queda una base confiable y luego cada mejora tiene una comparación clara.

[1]: https://opendata.cern.ch/record/328 "Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014 | CERN Open Data Portal"
