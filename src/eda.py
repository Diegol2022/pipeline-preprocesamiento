"""
Análisis exploratorio (EDA) con visualizaciones.

Genera y guarda en reports/figures/:
    01_distribuciones.png           Histogramas de todas las variables numéricas.
    02_heatmap_correlaciones.png    Heatmap de correlación de Pearson.
    03_correlacion_con_objetivo.png Correlación (Pearson y Spearman) de cada variable con el objetivo.
    04_pairplot.png                 Pairplot de las variables más relevantes, por ocean_proximity.
    05_boxplots_outliers.png        Boxplots de cada variable numérica (outliers según IQR).
    06_outliers_iqr.png             Porcentaje de outliers por variable (regla 1.5 × IQR).
    07_valor_por_proximidad.png     Distribución del objetivo según ocean_proximity.
    08_mapa_geografico.png          Ubicación de los distritos coloreada por valor de vivienda.

Y en reports/:
    correlaciones.csv               Matriz de correlación de Pearson.
    outliers_iqr.csv                Límites IQR, cantidad y porcentaje de outliers por variable.

IMPORTANTE: el EDA se hace SOLO sobre el conjunto de entrenamiento, para que
ninguna decisión de análisis se base en datos de test (evita data snooping).

Se ejecuta desde src/pipeline.py -> train().
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: guarda archivos, funciona en cualquier PC

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# Estilo: paleta fija y apta para daltonismo, ejes y grillas discretos
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"     # color principal (una sola serie)
ORANGE = "#eb6834"   # segunda serie

# Cada categoría tiene SIEMPRE el mismo color en todos los gráficos.
CATEGORY_COLORS = {
    "<1H OCEAN": "#2a78d6",
    "INLAND": "#eb6834",
    "NEAR OCEAN": "#1baf7a",
    "NEAR BAY": "#eda100",
    "ISLAND": "#e87ba4",
}

# Secuencial (magnitud): un solo tono, de claro a oscuro.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# Divergente (correlación -1..1): azul <-> gris neutro <-> rojo.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ee8a87", "#b8302f"]
)

TARGET = "median_house_value"


def _set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 110,
    })


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _category_order(series: pd.Series) -> list[str]:
    """Categorías presentes, en el orden fijo de CATEGORY_COLORS."""
    present = set(series.dropna().unique())
    known = [c for c in CATEGORY_COLORS if c in present]
    return known + sorted(present - set(known))


def _palette(order: list[str]) -> dict:
    return {c: CATEGORY_COLORS.get(c, MUTED) for c in order}


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_distributions(df: pd.DataFrame, num_cols: list[str]) -> Path:
    cols = num_cols + [TARGET]
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna(), bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
        ax.set_title(col)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Distribución de las variables numéricas (train)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_distribuciones.png")


def plot_correlation_heatmap(corr: pd.DataFrame) -> Path:
    # Triángulo inferior sin la diagonal (que siempre vale 1 y no aporta información)
    corr = corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8, "color": INK},
        linewidths=1.5, linecolor=SURFACE, square=True,
        cbar_kws={"shrink": 0.75, "label": "Correlación de Pearson"}, ax=ax,
    )
    ax.set_title("Matriz de correlación (Pearson, train)")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, "02_heatmap_correlaciones.png")


def plot_target_correlation(df: pd.DataFrame, num_cols: list[str]) -> tuple[Path, pd.DataFrame]:
    corr_t = pd.DataFrame({
        "Pearson": df[num_cols].corrwith(df[TARGET], method="pearson"),
        "Spearman": df[num_cols].corrwith(df[TARGET], method="spearman"),
    }).sort_values("Pearson")

    y = np.arange(len(corr_t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y - h / 2, corr_t["Pearson"], height=h, color=BLUE, label="Pearson (lineal)")
    ax.barh(y + h / 2, corr_t["Spearman"], height=h, color=ORANGE, label="Spearman (monótona)")
    ax.set_yticks(y, corr_t.index)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(f"Correlación con {TARGET}")
    ax.set_title(f"Correlación de cada variable con {TARGET} (train)")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, corr_t["Pearson"]):
        v = 0.0 if abs(v) < 0.005 else v  # evita mostrar "-0.00"
        ax.text(v + (0.02 if v >= 0 else -0.02), yi - h / 2, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK_2)
    ax.legend(loc="lower right")
    return _save(fig, "03_correlacion_con_objetivo.png"), corr_t


def plot_pairplot(df: pd.DataFrame, cat_col: str, sample_size: int = 2000,
                  random_state: int = 42) -> Path:
    cols = ["median_income", "housing_median_age", "total_rooms", TARGET]
    cols = [c for c in cols if c in df.columns]
    data = df[cols + [cat_col]].dropna()
    if len(data) > sample_size:  # muestra para que el gráfico sea legible y rápido
        data = data.sample(sample_size, random_state=random_state)
    order = _category_order(data[cat_col])
    g = sns.pairplot(
        data, vars=cols, hue=cat_col, hue_order=order, palette=_palette(order),
        corner=True, diag_kind="kde",
        plot_kws={"s": 9, "alpha": 0.5, "linewidth": 0},
        diag_kws={"common_norm": False, "fill": False, "linewidth": 1.5},
        height=2.4,
    )
    g.figure.suptitle(f"Pairplot de variables clave por {cat_col} "
                      f"(muestra de {len(data):,} distritos de train)",
                      fontweight="bold", y=1.02)
    return _save(g.figure, "04_pairplot.png")


def iqr_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cantidad y porcentaje de outliers por variable según la regla 1.5 × IQR."""
    rows = []
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({
            "variable": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "limite_inferior": low, "limite_superior": high,
            "outliers": n_out, "porcentaje": 100 * n_out / len(s),
        })
    return pd.DataFrame(rows).set_index("variable").sort_values("porcentaje", ascending=False)


def plot_boxplots(df: pd.DataFrame, num_cols: list[str]) -> Path:
    cols = num_cols + [TARGET]
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.3 * nrows))
    for ax, col in zip(axes.flat, cols):
        sns.boxplot(
            x=df[col], ax=ax, color="#b7d3f6", width=0.5, linewidth=1,
            boxprops={"edgecolor": "#256abf"}, medianprops={"color": "#0d366b", "linewidth": 1.8},
            whiskerprops={"color": "#256abf"}, capprops={"color": "#256abf"},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": ORANGE,
                        "markeredgecolor": "none", "alpha": 0.5},
        )
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Boxplots por variable: puntos naranjas = outliers (fuera de 1.5 × IQR)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_boxplots_outliers.png")


def plot_outlier_share(outliers: pd.DataFrame) -> Path:
    data = outliers.sort_values("porcentaje")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(data.index, data["porcentaje"], color=BLUE, height=0.6)
    for i, (pct, n) in enumerate(zip(data["porcentaje"], data["outliers"])):
        ax.text(pct + 0.1, i, f"{pct:.1f}%  ({n:,})", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("% de registros fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR]")
    ax.set_title("Proporción de outliers por variable (regla IQR, train)")
    ax.set_xlim(0, max(data["porcentaje"].max() * 1.25, 1))
    ax.grid(axis="y", visible=False)
    return _save(fig, "06_outliers_iqr.png")


def plot_target_by_category(df: pd.DataFrame, cat_col: str) -> Path:
    order = _category_order(df[cat_col])
    counts = df[cat_col].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(
        data=df, x=cat_col, y=TARGET, order=order, hue=cat_col, hue_order=order,
        palette=_palette(order), legend=False, width=0.55, linewidth=1,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        ax=ax,
    )
    ax.set_xticks(range(len(order)), [f"{c}\n(n={counts[c]:,})" for c in order])
    ax.set_xlabel("")
    ax.set_ylabel(TARGET)
    ax.set_title(f"{TARGET} según {cat_col} (train)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.grid(axis="x", visible=False)
    return _save(fig, "07_valor_por_proximidad.png")


def plot_geo(df: pd.DataFrame) -> Path | None:
    if not {"longitude", "latitude"}.issubset(df.columns):
        return None
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["longitude"], df["latitude"], c=df[TARGET], cmap=SEQUENTIAL,
                    s=4, alpha=0.6, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.solids.set_alpha(1)  # la barra de color sin transparencia
    cbar.set_label(TARGET)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Ubicación de los distritos y valor de la vivienda (train)")
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, "08_mapa_geografico.png")


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #
def run_eda(X_train: pd.DataFrame, y_train: pd.Series,
            num_cols: list[str], cat_cols: list[str]) -> None:
    """Genera todas las visualizaciones y reportes del EDA sobre train."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    df = X_train.copy()
    df[TARGET] = y_train

    corr = df[num_cols + [TARGET]].corr(method="pearson")
    corr.round(4).to_csv(REPORTS_DIR / "correlaciones.csv")

    outliers = iqr_outliers(df, num_cols + [TARGET])
    outliers.round(4).to_csv(REPORTS_DIR / "outliers_iqr.csv")

    generated = [
        plot_distributions(df, num_cols),
        plot_correlation_heatmap(corr),
    ]
    path, corr_t = plot_target_correlation(df, num_cols)
    generated.append(path)
    if cat_cols:
        generated.append(plot_pairplot(df, cat_cols[0]))
    generated += [plot_boxplots(df, num_cols), plot_outlier_share(outliers)]
    if cat_cols:
        generated.append(plot_target_by_category(df, cat_cols[0]))
    geo = plot_geo(df)
    if geo:
        generated.append(geo)

    # Resumen en consola
    print(f"\nCorrelación con {TARGET} (Pearson / Spearman):")
    print(corr_t.sort_values("Pearson", ascending=False).round(3).to_string())
    print("\nOutliers por variable (regla 1.5 × IQR):")
    print(outliers[["outliers", "porcentaje"]].round(2).to_string())
    print(f"\nGráficos guardados en {FIGURES_DIR.relative_to(ROOT)}/:")
    for p in generated:
        print(f"  - {p.name}")
