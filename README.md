# Pipeline de Preprocesamiento Robusto con Scikit-Learn

Pre-entrega: construcción de un pipeline reutilizable que automatiza la limpieza y transformación de datos y los deja listos para un modelo de Machine Learning, evitando la fuga de datos (*data leakage*) y garantizando que el preprocesamiento sea idéntico en entrenamiento y en inferencia.

## Dataset

**California Housing** (versión de *Hands-On Machine Learning*, A. Géron), basada en el censo de California de 1990. Cada fila representa un distrito (block group) y la tarea es de **regresión**: predecir el valor mediano de las viviendas.

| Columna | Tipo | Descripción |
|---|---|---|
| `longitude` | numérica | Longitud del distrito |
| `latitude` | numérica | Latitud del distrito |
| `housing_median_age` | numérica | Antigüedad mediana de las viviendas (años) |
| `total_rooms` | numérica | Total de habitaciones del distrito |
| `total_bedrooms` | numérica | Total de dormitorios (**contiene valores nulos**) |
| `population` | numérica | Población del distrito |
| `households` | numérica | Cantidad de hogares |
| `median_income` | numérica | Ingreso mediano (en decenas de miles de USD) |
| `ocean_proximity` | **categórica** | Cercanía al océano: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND` |
| `median_house_value` | **objetivo** | Valor mediano de la vivienda (USD) |

- ~20.640 registros, 10 columnas.
- Fuente: https://github.com/ageron/handson-ml2/tree/master/datasets/housing
- Se eligió esta versión (y no `fetch_california_housing` de sklearn) porque incluye una variable categórica y valores nulos reales, lo que permite ejercitar todo el `ColumnTransformer`.

El CSV se descarga automáticamente en `data/housing.csv` la primera vez que se ejecuta el código.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                # punto de entrada: entrena, evalúa, exporta y predice
├── data/                  # housing.csv (se descarga al ejecutar)
├── models/                # pipeline entrenado (.joblib)
└── src/
    ├── __init__.py
    ├── pipeline.py        # carga, split, ColumnTransformer, Pipeline, métricas y joblib
    └── predict.py         # carga del pipeline y predicción sobre datos nuevos
```

## Diseño del pipeline

```
Pipeline
├── preprocessor: ColumnTransformer
│   ├── num → SimpleImputer(median) → StandardScaler
│   └── cat → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
└── model: LinearRegression
```

**Decisiones**
- **Split antes de ajustar:** `train_test_split` se ejecuta antes de cualquier `fit`. El imputador, el scaler y el encoder aprenden sus parámetros solo con `X_train`; el test se transforma con esos parámetros. Así no hay *data leakage*.
- **Mediana para numéricas:** variables como `total_rooms` o `population` tienen outliers, y la mediana es más robusta que la media.
- **`handle_unknown="ignore"`:** si en producción llega una categoría no vista en entrenamiento, se codifica como un vector de ceros en lugar de lanzar un error.
- **Un único objeto:** preprocesamiento + modelo se entrenan con un solo `fit` y se usan con un solo `predict`, lo que asegura consistencia entre entrenamiento e inferencia.

## Cómo ejecutarlo paso a paso

Las instrucciones sirven para **Windows**, **macOS** y **Linux**. Cuando un comando cambia según el sistema operativo, se indican las dos variantes.

### Paso 0: Requisitos previos

Necesitás tener instalado:

| Herramienta | Versión | Para qué | Dónde conseguirla |
|---|---|---|---|
| **Python** | 3.10 o superior | Ejecutar el código | https://www.python.org/downloads/ |
| **Git** | cualquiera | Clonar el repositorio (opcional, ver Paso 1) | https://git-scm.com/downloads |
| **Conexión a internet** | — | Instalar librerías y descargar el dataset la primera vez | — |

> **Windows:** al instalar Python, marcá la casilla **"Add Python to PATH"** en la primera pantalla del instalador. Si no lo hacés, la terminal no va a reconocer el comando `python`.

Para verificar que quedaron instalados, abrí una terminal y ejecutá:

```bash
python --version
git --version
```

Deberías ver algo como `Python 3.12.x` y `git version 2.x`.

> **¿Cómo abro una terminal?**
> - **Windows:** tecla Windows → escribí `PowerShell` → Enter.
> - **macOS:** `Cmd + Espacio` → escribí `Terminal` → Enter.
> - **Linux:** `Ctrl + Alt + T`.
>
> En macOS/Linux, si `python` no funciona, probá con `python3` (y `pip3` en lugar de `pip`). En Windows, si no funciona `python`, probá con `py`.

### Paso 1: Descargar el proyecto

**Opción A: con Git (recomendada)**

```bash
git clone https://github.com/TU-USUARIO/pipeline-preprocesamiento.git
cd pipeline-preprocesamiento
```

**Opción B: sin Git**

1. En la página del repositorio en GitHub, hacé clic en el botón verde **Code** → **Download ZIP**.
2. Descomprimí el archivo.
3. En la terminal, entrá a la carpeta descomprimida:
   ```bash
   cd ruta/a/pipeline-preprocesamiento
   ```

> A partir de acá, **todos los comandos se ejecutan desde la carpeta raíz del proyecto** (la que contiene `main.py`). Podés comprobarlo con `ls` (macOS/Linux) o `dir` (Windows): tenés que ver `main.py`, `src/`, `README.md`, etc.

### Paso 2: Crear un entorno virtual

El entorno virtual aísla las librerías de este proyecto para que no interfieran con otras instalaciones de Python.

```bash
python -m venv .venv
```

Esto crea una carpeta `.venv/` dentro del proyecto (ya está excluida de Git en `.gitignore`).

### Paso 3: Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Si se activó bien, vas a ver `(.venv)` al principio de la línea de la terminal.

> **Error en PowerShell** del tipo *"la ejecución de scripts está deshabilitada en este sistema"*: ejecutá una sola vez
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> confirmá con `S` (o `Y`) y volvé a intentar el Paso 3.

### Paso 4: Instalar las dependencias

Con el entorno activado:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se instalan `pandas`, `numpy`, `scikit-learn` y `joblib`. Puede tardar uno o dos minutos.

### Paso 5: Ejecutar el proyecto

```bash
python main.py
```

El script hace todo el flujo automáticamente:

1. **Descarga el dataset** a `data/housing.csv` (solo la primera vez; las siguientes lo lee del disco).
2. **Explora los datos:** muestra la forma, los tipos de cada columna y los valores nulos.
3. **Divide en Train/Test** (80 % / 20 %) antes de ajustar cualquier transformación.
4. **Entrena el pipeline** (preprocesamiento + regresión lineal) y **muestra las métricas** MAE, RMSE y R² sobre el conjunto de test.
5. **Exporta el pipeline** entrenado a `models/pipeline_housing.joblib`.
6. **Vuelve a cargarlo desde disco y predice** sobre 3 registros nuevos, uno con un valor nulo y otro con una categoría que no existe en el dataset.

### Paso 6: Verificar la salida

La salida en la terminal tiene esta forma (los números exactos dependen de los datos):

```
============================================================
1. ENTRENAMIENTO Y EVALUACIÓN
============================================================
Forma del dataset: (20640, 10)
...
Valores nulos por columna:
total_bedrooms    207

Train: (16512, 9) | Test: (4128, 9)
Numéricas (8): ['longitude', 'latitude', ..., 'median_income']
Categóricas (1): ['ocean_proximity']

Métricas en test:
   MAE: ...
  RMSE: ...
    R2: ...

Pipeline guardado en: models/pipeline_housing.joblib

============================================================
2. PREDICCIÓN SOBRE DATOS NUEVOS (pipeline cargado con joblib)
============================================================
 median_income ocean_proximity  prediccion_usd
        8.3252        NEAR BAY        ...
        3.5000       <1H OCEAN        ...
        4.2000          DESERT        ...
```

Además, se generan dos archivos:
- `data/housing.csv`: el dataset descargado.
- `models/pipeline_housing.joblib`: el pipeline entrenado, listo para reutilizar.

### Paso 7 (opcional): Usar el pipeline guardado desde otro código

Una vez ejecutado `main.py`, el pipeline se puede cargar y usar sin volver a entrenar:

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/pipeline_housing.joblib")

nuevo = pd.DataFrame([{
    "longitude": -122.25, "latitude": 37.85, "housing_median_age": 30.0,
    "total_rooms": 1500.0, "total_bedrooms": 300.0, "population": 800.0,
    "households": 280.0, "median_income": 5.0, "ocean_proximity": "NEAR BAY",
}])

print(pipeline.predict(nuevo))
```

Los datos se pasan **crudos**, con las mismas columnas que el CSV original (sin `median_house_value`); el pipeline se encarga de imputar, escalar y codificar.

### Paso 8: Desactivar el entorno

Cuando termines:

```bash
deactivate
```

La próxima vez que quieras ejecutarlo, solo repetí el **Paso 3** (activar) y el **Paso 5** (ejecutar); no hace falta reinstalar nada.

### Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| `'python' no se reconoce como un comando...` | Python no está en el PATH | Reinstalá Python marcando **"Add Python to PATH"**, o usá `py` (Windows) / `python3` (macOS/Linux) |
| `ModuleNotFoundError: No module named 'sklearn'` (o `pandas`) | El entorno virtual no está activado o no se instalaron las dependencias | Activá el entorno (Paso 3) y ejecutá `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Se ejecutó `main.py` desde otra carpeta | Entrá a la carpeta raíz del proyecto con `cd` y ejecutá `python main.py` |
| Error de conexión / `URLError` al descargar el dataset | Sin internet o red con restricciones | Descargá manualmente [housing.csv](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv) (clic derecho → *Guardar como*) y guardalo en `data/housing.csv` |
| `FileNotFoundError: models/pipeline_housing.joblib` | Se intentó cargar el pipeline sin haberlo entrenado | Ejecutá primero `python main.py` |
| PowerShell no deja activar el entorno | Política de ejecución de scripts | Ver la nota del Paso 3 |

## Resultados

Métricas sobre el conjunto de test (20 % de los datos, `random_state=42`):

| Métrica | Valor |
|---|---|
| MAE | _completar_ |
| RMSE | _completar_ |
| R² | _completar_ |

La segunda parte de `main.py` demuestra que el pipeline exportado predice sin errores sobre datos crudos que incluyen un valor nulo y una categoría desconocida (`"DESERT"`).

## Cumplimiento de la consigna

| Criterio | Dónde |
|---|---|
| README con descripción del dataset | Este archivo |
| `Pipeline` con al menos dos pasos de transformación | `build_pipeline()` en `src/pipeline.py` (imputer → scaler / imputer → onehot, y preprocesador → modelo) |
| `ColumnTransformer` que diferencia numéricas y categóricas | `preprocessor` en `build_pipeline()` (`src/pipeline.py`) |
| División Train/Test antes de ajustar el preprocesamiento | `train()` en `src/pipeline.py`: `train_test_split` antes de `fit` |
| Pipeline funcional con `.predict()` sobre datos nuevos | `predict()` en `src/predict.py`, ejecutado desde `main.py` |
| Exportación con `joblib` | `joblib.dump` en `train()` y `joblib.load` en `src/predict.py` |
