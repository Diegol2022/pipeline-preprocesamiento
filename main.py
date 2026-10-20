"""
Punto de entrada del proyecto.

Ejecuta el flujo completo:
    0. Genera el análisis exploratorio (gráficos en reports/figures/).
    1. Entrena el pipeline (preprocesamiento + modelo), muestra métricas
       y lo exporta con joblib en models/pipeline_housing.joblib.
    2. Carga el pipeline exportado y predice sobre datos nuevos.

Uso:
    python main.py                 # flujo completo con gráficos
    python main.py --sin-graficos  # omite el EDA visual (más rápido)
"""

import argparse

from src.pipeline import train
from src.predict import example_data, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de preprocesamiento - California Housing")
    parser.add_argument("--sin-graficos", action="store_true",
                        help="no generar los gráficos del análisis exploratorio")
    args = parser.parse_args()

    print("=" * 60)
    print("1. EXPLORACIÓN, ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 60)
    train(make_plots=not args.sin_graficos)

    print("\n" + "=" * 60)
    print("2. PREDICCIÓN SOBRE DATOS NUEVOS (pipeline cargado con joblib)")
    print("=" * 60)
    resultado = predict(example_data())
    flag_cols = [c for c in resultado.columns if c.endswith("_era_null")]
    cols = ["total_bedrooms", "median_income", "ocean_proximity", *flag_cols,
            "prediccion_usd"]
    print(resultado[cols].to_string(index=False))


if __name__ == "__main__":
    main()
