"""
Pipeline de preprocesamiento robusto con Scikit-Learn.

Dataset: California Housing (versión con la columna categórica `ocean_proximity`).
Objetivo: predecir `median_house_value` (regresión).

Flujo:
    1. Carga y exploración de los datos.
    2. División Train/Test ANTES de ajustar cualquier transformación (evita data leakage).
       EDA visual (src/eda.py) sobre el conjunto de train.
    3. ColumnTransformer: tratamiento separado de variables numéricas y categóricas,
       más un flag de trazabilidad (1 = el valor original era nulo, 0 = estaba presente).
    4. Pipeline final = preprocesador + estimador (LinearRegression).
    5. Entrenamiento, métricas sobre test y exportación del pipeline con joblib.

Se ejecuta desde main.py (raíz del repositorio).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda import run_eda

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_PATH = ROOT / "models" / "pipeline_housing.joblib"
DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.csv"
)

TARGET = "median_house_value"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# --------------------------------------------------------------------------- #
# 1. Carga y exploración
# --------------------------------------------------------------------------- #
def load_data(path: Path = DATA_PATH, url: str = DATA_URL) -> pd.DataFrame:
    """Lee el CSV local; si no existe, lo descarga y lo guarda en data/."""
    if path.exists():
        return pd.read_csv(path)
    print(f"No se encontró {path.name}, descargando desde {url} ...")
    df = pd.read_csv(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def explore(df: pd.DataFrame) -> None:
    """Exploración mínima: forma, tipos y valores nulos."""
    print(f"Forma del dataset: {df.shape}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isna().sum()[df.isna().sum() > 0])


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identifica columnas numéricas y categóricas según su dtype."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols


# --------------------------------------------------------------------------- #
# 2 y 3. Definición de transformers y ensamblaje del pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Construye el pipeline completo: preprocesador + modelo."""

    # Numéricas: imputación por mediana (robusta a outliers) + estandarización
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categóricas: imputación por valor más frecuente + One-Hot Encoding.
    # handle_unknown="ignore" evita errores si en inferencia aparece una categoría nueva.
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Flag de trazabilidad: por cada columna que tenía nulos en train, agrega
    # una columna binaria (1 = el valor original era nulo, 0 = estaba presente).
    # Va en una rama aparte para que el flag NO pase por el StandardScaler y
    # se mantenga como 0/1.
    # - features="missing-only": solo genera flags para columnas con nulos en train.
    # - error_on_new=False: si en inferencia aparece un nulo en otra columna,
    #   no falla (esa columna la cubre igual el imputer).
    missing_flags = MissingIndicator(features="missing-only", error_on_new=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
        ("missing_flag", missing_flags, num_cols + cat_cols),
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])


def get_missing_flags(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Devuelve los flags de trazabilidad (0/1) que el pipeline genera para X.

    Las columnas se llaman `<columna>_era_null`. Sirve para auditar qué valores
    fueron imputados en un conjunto de datos dado.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    indicator = preprocessor.named_transformers_["missing_flag"]
    input_cols = next(cols for name, _, cols in preprocessor.transformers_
                      if name == "missing_flag")
    flagged_cols = [input_cols[i] for i in indicator.features_]
    flags = indicator.transform(X[input_cols]).astype(int)
    return pd.DataFrame(flags, columns=[f"{c}_era_null" for c in flagged_cols],
                        index=X.index)


# --------------------------------------------------------------------------- #
# 4. Validación
# --------------------------------------------------------------------------- #
def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de regresión sobre el conjunto de test."""
    y_pred = pipeline.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": r2_score(y_test, y_pred),
    }
    print("\nMétricas en test:")
    for name, value in metrics.items():
        print(f"  {name:>4}: {value:,.4f}")
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def train(make_plots: bool = True) -> Pipeline:
    """Carga los datos, hace el EDA, entrena y evalúa el pipeline, y lo exporta con joblib.

    make_plots=False omite la generación de gráficos (ejecución más rápida).
    """
    df = load_data()
    explore(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split ANTES de cualquier fit: el imputador, el scaler y el encoder
    # aprenden sus parámetros SOLO con X_train.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    num_cols, cat_cols = split_columns(X_train)
    print(f"Numéricas ({len(num_cols)}): {num_cols}")
    print(f"Categóricas ({len(cat_cols)}): {cat_cols}")

    # EDA visual SOLO con train: ninguna decisión se toma mirando el test.
    if make_plots:
        run_eda(X_train, y_train, num_cols, cat_cols)

    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    flags_train = get_missing_flags(pipeline, X_train)
    print(f"\nFlags de trazabilidad creados: {flags_train.columns.tolist()}")
    print("Valores imputados en train (cantidad de 1 por flag):")
    print(flags_train.sum().to_string())

    evaluate(pipeline, X_test, y_test)

    # 5. Persistencia
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nPipeline guardado en: {MODEL_PATH.relative_to(ROOT)}")
    return pipeline
