"""
Gráficos del modelo ARIMA/SARIMA de materias primas.
Autor: Angel Marchán Collazos (UNMSM) - 4 de noviembre de 2026
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from . import AUTOR

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
FUENTE = f"Fuente: Banco Mundial (Pink Sheet). Elaboración: {AUTOR}."
EPISODIOS = [("2008-07", "2009-03", "Crisis financiera"), ("2020-02", "2020-05", "COVID-19")]


def _pie(fig):
    fig.text(0.01, 0.005, FUENTE, fontsize=7, color="gray")


def _slug(nombre: str) -> str:
    t = nombre.lower().replace(" ", "_")
    for a, b in zip("áéíóúñ", "aeioun"):
        t = t.replace(a, b)
    return t


def panel_series(precios: pd.DataFrame, unidades: dict, ruta: Path):
    n = precios.shape[1]
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(13, 3 * np.ceil(n / 3)))
    for ax, col in zip(axes.flat, precios.columns):
        ax.plot(precios.index, precios[col], color="#1f4e79", lw=1.1)
        for a, b, _ in EPISODIOS:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="gray", alpha=.2)
        ax.set_title(f"{col} ({unidades[col]})", fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Precios mensuales de las principales materias primas exportadas por el Perú, 2000-2026",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, .02, 1, .97)); _pie(fig)
    fig.savefig(ruta); plt.close(fig)


def indice_base100(precios: pd.DataFrame, ruta: Path):
    idx = precios / precios.loc["2000"].mean() * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in idx.columns:
        ax.plot(idx.index, idx[col], lw=1.1, label=col)
    ax.set_yscale("log"); ax.set_ylabel("Índice (promedio 2000 = 100, escala log)")
    ax.set_title("Evolución relativa de precios (promedio 2000 = 100)", weight="bold")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(rect=(0, .03, 1, 1)); _pie(fig)
    fig.savefig(ruta); plt.close(fig)


def acf_pacf(y: pd.Series, ruta: Path):
    dy = y.diff().dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes[0, 0].plot(y.index, y, color="#1f4e79"); axes[0, 0].set_title(f"log({y.name}) en niveles")
    axes[0, 1].plot(dy.index, dy, color="#c55a11", lw=.8); axes[0, 1].set_title("Primera diferencia del log (retorno mensual)")
    plot_acf(dy, lags=36, ax=axes[1, 0], title="FAS de Δlog"); plot_pacf(dy, lags=36, ax=axes[1, 1], method="ywm", title="FAP de Δlog")
    fig.suptitle(f"{y.name}: identificación Box-Jenkins", weight="bold")
    fig.tight_layout(rect=(0, .03, 1, .96)); _pie(fig); fig.savefig(ruta); plt.close(fig)


def descomposicion(y: pd.Series, ruta: Path):
    r = STL(y, period=12, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, (s, t) in zip(axes, [(y, "log precio"), (r.trend, "Tendencia"), (r.seasonal, "Estacionalidad"), (r.resid, "Irregular")]):
        ax.plot(s.index, s, lw=1, color="#1f4e79"); ax.set_ylabel(t)
    fig.suptitle(f"{y.name}: descomposición STL", weight="bold")
    fig.tight_layout(rect=(0, .03, 1, .97)); _pie(fig); fig.savefig(ruta); plt.close(fig)


def diagnostico(res, nombre: str, ruta: Path):
    fig = res.plot_diagnostics(figsize=(11, 7), lags=24)
    fig.suptitle(f"{nombre}: diagnóstico de residuos del modelo final", weight="bold")
    fig.tight_layout(rect=(0, .03, 1, .96)); _pie(fig); fig.savefig(ruta); plt.close(fig)


def pronostico(precio: pd.Series, pron: pd.DataFrame, val_pred: pd.Series, titulo: str, unidad: str, ruta: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    h = precio.loc["2015":]
    ax.plot(h.index, h, color="#1f4e79", lw=1.3, label="Observado")
    ax.plot(val_pred.index, val_pred, color="#7f7f7f", ls=":", lw=1.4, label="Validación (pronóstico fuera de muestra)")
    ult = precio.iloc[[-1]]
    med = pd.concat([ult, pron["Pronóstico (mediana)"]])
    ax.plot(med.index, med, color="#c00000", lw=1.8, label="Pronóstico 2026-2027")
    ax.fill_between(pron.index, pron["LI 95%"], pron["LS 95%"], color="#c00000", alpha=.12, label="IC 95%")
    ax.fill_between(pron.index, pron["LI 80%"], pron["LS 80%"], color="#c00000", alpha=.22, label="IC 80%")
    ax.axvline(precio.index[-1], color="k", lw=.7, ls="--")
    ax.set_title(titulo, weight="bold"); ax.set_ylabel(unidad); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, .03, 1, 1)); _pie(fig); fig.savefig(ruta); plt.close(fig)


def panel_pronosticos(resultados: dict, unidades: dict, ruta: Path):
    n = len(resultados)
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(13, 3 * np.ceil(n / 3)))
    for ax, (nom, r) in zip(axes.flat, resultados.items()):
        h = r["precio"].loc["2020":]; p = r["pronostico"]
        ax.plot(h.index, h, color="#1f4e79", lw=1)
        ax.plot(pd.concat([h.iloc[[-1]], p["Pronóstico (mediana)"]]), color="#c00000", lw=1.5)
        ax.fill_between(p.index, p["LI 95%"], p["LS 95%"], color="#c00000", alpha=.12)
        ax.fill_between(p.index, p["LI 80%"], p["LS 80%"], color="#c00000", alpha=.22)
        ax.set_title(f"{nom} ({unidades[nom]}) – {r['candidatos'][r['final']]['etiqueta']}", fontsize=8.5)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Proyección de precios a diciembre de 2027 (mediana e intervalos 80% y 95%)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, .02, 1, .97)); _pie(fig); fig.savefig(ruta); plt.close(fig)


def barras_variacion(tabla: pd.DataFrame, ruta: Path):
    t = tabla.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = ["#2e7d32" if v >= 0 else "#c62828" for v in t]
    ax.barh(t.index, t.values, color=cols)
    for i, v in enumerate(t.values):
        ax.text(v, i, f" {v:+.1f}%", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=.7)
    ax.set_xlabel("Variación % del precio promedio 2027 (proyectado) vs 2026")
    ax.set_title("Variación esperada del precio promedio anual 2027/2026", weight="bold")
    fig.tight_layout(rect=(0, .03, 1, 1)); _pie(fig); fig.savefig(ruta); plt.close(fig)


def mapa_estacional(factores: pd.DataFrame, ruta: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    lim = np.abs(factores.values).max()
    im = ax.imshow(factores.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(factores))); ax.set_yticklabels(factores.index)
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
    ax.set_xticks(range(12)); ax.set_xticklabels(meses)
    for i in range(factores.shape[0]):
        for j in range(12):
            ax.text(j, i, f"{factores.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Factor estacional STL (% sobre tendencia)")
    ax.set_title("Patrón estacional promedio de los precios (STL, 2000-2026)", weight="bold")
    fig.tight_layout(rect=(0, .03, 1, 1)); _pie(fig); fig.savefig(ruta); plt.close(fig)
