"""
Inferencia con el pipeline entrenado.

Carga el pipeline exportado con joblib y predice sobre datos nuevos "crudos"
(sin preprocesar). Los ejemplos incluyen a propósito un valor nulo y una
categoría que no existía en entrenamiento, para mostrar que el pipeline
los maneja sin errores.

Se ejecuta desde main.py (raíz del repositorio).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline import get_missing_flags

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pipeline_housing.joblib"


def load_pipeline(path: Path = MODEL_PATH):
    """Carga el pipeline entrenado desde disco."""
    return joblib.load(path)


def example_data() -> pd.DataFrame:
    """Datos nuevos de ejemplo, en el mismo formato crudo que el CSV original."""
    return pd.DataFrame([
        {   # Caso normal
            "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
            "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0,
            "households": 126.0, "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY",
        },
        {   # Valor nulo en una numérica -> lo imputa con la mediana de train
            "longitude": -118.25, "latitude": 34.05, "housing_median_age": 25.0,
            "total_rooms": 2000.0, "total_bedrooms": np.nan, "population": 1100.0,
            "households": 400.0, "median_income": 3.5,
            "ocean_proximity": "<1H OCEAN",
        },
        {   # Categoría nunca vista -> OneHotEncoder(handle_unknown="ignore")
            "longitude": -120.0, "latitude": 36.5, "housing_median_age": 15.0,
            "total_rooms": 1500.0, "total_bedrooms": 300.0, "population": 900.0,
            "households": 280.0, "median_income": 4.2,
            "ocean_proximity": "DESERT",
        },
    ])


def predict(new_data: pd.DataFrame, path: Path = MODEL_PATH) -> pd.DataFrame:
    """Carga el pipeline y devuelve los datos con los flags de trazabilidad
    y una columna de predicción."""
    pipeline = load_pipeline(path)
    result = new_data.copy()
    result = result.join(get_missing_flags(pipeline, new_data))
    result["prediccion_usd"] = pipeline.predict(new_data).round(0)
    return result
