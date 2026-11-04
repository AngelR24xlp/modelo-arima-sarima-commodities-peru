"""
Carga y limpieza de precios mensuales de materias primas (World Bank Pink Sheet).
Autor: Angel Marchán Collazos (UNMSM) - 4 de noviembre de 2026
"""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "data" / "raw" / "CMO-Historical-Data-Monthly.xlsx"
URL = ("https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
       "related/CMO-Historical-Data-Monthly.xlsx")

# Principales productos de exportación del Perú disponibles en el Pink Sheet
PRODUCTOS = {
    "Cobre":            ("Copper", "US$/tm"),
    "Oro":              ("Gold", "US$/oz troy"),
    "Zinc":             ("Zinc", "US$/tm"),
    "Plata":            ("Silver", "US$/oz troy"),
    "Plomo":            ("Lead", "US$/tm"),
    "Estaño":           ("Tin", "US$/tm"),
    "Hierro":           ("Iron ore, cfr spot", "US$/tms"),
    "Harina de pescado": ("Fish meal", "US$/tm"),
    "Café arábica":     ("Coffee, Arabica", "US$/kg"),
}
INICIO = "2000-01"


def cargar_precios(archivo: Path = ARCHIVO) -> pd.DataFrame:
    if not archivo.exists():
        archivo.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(URL, archivo)
    bruto = pd.read_excel(archivo, sheet_name="Monthly Prices", header=None)
    fila_nombres = bruto.index[bruto.iloc[:, 1].astype(str).str.contains("Crude oil, average")][0]
    nombres = bruto.iloc[fila_nombres].tolist()
    datos = bruto.iloc[fila_nombres + 2:].copy()
    datos.columns = ["periodo"] + nombres[1:]
    datos = datos[datos["periodo"].astype(str).str.match(r"^\d{4}M\d{2}$")]
    datos.index = pd.PeriodIndex(datos["periodo"].str.replace("M", "-"), freq="M").to_timestamp()
    salida = pd.DataFrame(index=datos.index)
    for es, (en, _) in PRODUCTOS.items():
        salida[es] = pd.to_numeric(datos[en].astype(str), errors="coerce")
    salida = salida.loc[INICIO:].astype(float)
    salida.index.name = "fecha"
    salida.index.freq = "MS"
    return salida


def ultima_actualizacion(archivo: Path = ARCHIVO) -> str:
    bruto = pd.read_excel(archivo, sheet_name="Monthly Prices", header=None, nrows=6)
    for v in bruto.iloc[:, 0].astype(str):
        if v.startswith("Updated"):
            return v
    return ""
