"""
Modelo econométrico ARIMA / SARIMA de los precios de las principales materias primas
que exporta el Perú (datos mensuales 2000-2026) y proyección a diciembre de 2027.

Autor: Angel Marchán Collazos (UNMSM) - 4 de noviembre de 2026
Uso:   python main.py
"""
import json
import numpy as np
import pandas as pd

from src import AUTOR, FECHA
from src.datos import RAIZ, PRODUCTOS, cargar_precios, ultima_actualizacion
from src.modelo import analizar_producto, MESES_VALIDACION
from src import graficos as g
from src.reporte import generar_informe

FIG = RAIZ / "output" / "figuras"
TAB = RAIZ / "output" / "tablas"
PERIODOS = {
    "2000-2002 (pre-auge)": ("2000", "2002"),
    "2003-2008 (superciclo)": ("2003", "2008"),
    "2009-2011 (rebote post-crisis)": ("2009", "2011"),
    "2012-2015 (fin del superciclo)": ("2012", "2015"),
    "2016-2019 (recuperación)": ("2016", "2019"),
    "2020-2022 (COVID y rebote)": ("2020", "2022"),
    "2023-2026 (nuevo auge)": ("2023", "2026"),
}


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    precios = cargar_precios()
    unidades = {k: v[1] for k, v in PRODUCTOS.items()}
    precios.to_csv(RAIZ / "data" / "processed" / "precios_mensuales_2000_2026.csv", float_format="%.4f")
    print(f"Datos: {precios.index[0]:%Y-%m} a {precios.index[-1]:%Y-%m} ({len(precios)} meses). {ultima_actualizacion()}")

    # --- Análisis descriptivo ---
    g.panel_series(precios, unidades, FIG / "01_series_precios_2000_2026.png")
    g.indice_base100(precios, FIG / "02_indice_base100.png")
    ret = np.log(precios).diff() * 100
    desc = pd.DataFrame({
        "Unidad": pd.Series(unidades),
        "Promedio": precios.mean(), "Desv. estándar": precios.std(),
        "Mínimo": precios.min(), "Fecha mínimo": precios.idxmin().dt.strftime("%Y-%m"),
        "Máximo": precios.max(), "Fecha máximo": precios.idxmax().dt.strftime("%Y-%m"),
        "Precio ene-2000": precios.iloc[0], "Precio último": precios.iloc[-1],
        "Var. acumulada (%)": (precios.iloc[-1] / precios.iloc[0] - 1) * 100,
        "Crec. anual compuesto (%)": ((precios.iloc[-1] / precios.iloc[0]) ** (12 / (len(precios) - 1)) - 1) * 100,
        "Volatilidad anualizada (%)": ret.std() * np.sqrt(12),
    })
    desc.index.name = "Producto"
    crec = pd.DataFrame({p: ((precios.loc[b].mean() / precios.loc[a].mean()) ** (1 / max(int(b) - int(a), 1)) - 1) * 100
                         for p, (a, b) in PERIODOS.items()}).T
    crec.index.name = "Periodo (crec. anual compuesto del precio promedio, %)"

    # --- Modelación ---
    resultados = {}
    for nom in precios.columns:
        print(f"  Estimando {nom}...")
        r = analizar_producto(precios[nom])
        resultados[nom] = r
        s = g._slug(nom)
        g.acf_pacf(r["y"], FIG / f"acf_pacf_{s}.png")
        g.descomposicion(r["y"], FIG / f"stl_{s}.png")
        g.diagnostico(r["res"], nom, FIG / f"diagnostico_{s}.png")
        c = r["candidatos"][r["final"]]
        g.pronostico(r["precio"], r["pronostico"], c["val_pred"],
                     f"{nom}: {c['etiqueta']} – pronóstico a dic-2027", unidades[nom], FIG / f"pronostico_{s}.png")

    est = pd.DataFrame({n: r["estacionariedad"] for n, r in resultados.items()}).T
    filas = []
    for n, r in resultados.items():
        for k, c in r["candidatos"].items():
            filas.append({"Producto": n, "Familia": k, "Especificación": c["etiqueta"],
                          "AIC": c["aic"], "BIC": c["bic"],
                          f"MAPE validación {MESES_VALIDACION}m (%)": c["val"]["MAPE (%)"],
                          "RMSE validación": c["val"]["RMSE"],
                          "MAPE paseo aleatorio (%)": c["val_ingenuo"]["MAPE (%)"],
                          "Modelo final": "Sí" if k == r["final"] else ""})
    sel = pd.DataFrame(filas)
    coefs = []
    for n, r in resultados.items():
        res = r["res"]
        for p in res.params.index:
            coefs.append({"Producto": n, "Modelo": r["candidatos"][r["final"]]["etiqueta"],
                          "Parámetro": "deriva (intercept)" if p == "intercept" else p,
                          "Coeficiente": res.params[p], "Error estándar": res.bse[p],
                          "z": res.tvalues[p], "p-valor": res.pvalues[p]})
    coefs = pd.DataFrame(coefs)
    diag = pd.DataFrame({n: r["diagnostico"] for n, r in resultados.items()}).T
    factores = pd.DataFrame({n: r["factores"] for n, r in resultados.items()}).T
    factores.columns = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
    factores.insert(0, "Fuerza estacional (0-1)", pd.Series({n: r["fuerza_estacional"] for n, r in resultados.items()}))

    pron_largo = pd.concat({n: r["pronostico"] for n, r in resultados.items()}, names=["Producto", "Mes"]).reset_index()
    pron_largo["Mes"] = pron_largo["Mes"].dt.strftime("%Y-%m")
    pron_ancho = pd.DataFrame({n: r["pronostico"]["Pronóstico (mediana)"] for n, r in resultados.items()})
    pron_ancho.index = pron_ancho.index.strftime("%Y-%m"); pron_ancho.index.name = "Mes (pronóstico)"

    anual = {}
    for n, r in resultados.items():
        serie = pd.concat([r["precio"], r["pronostico"]["Pronóstico (mediana)"]])
        p27 = r["pronostico"].loc["2027"]
        anual[n] = {"Unidad": unidades[n],
                    "2024 (obs.)": r["precio"].loc["2024"].mean(),
                    "2025 (obs.)": r["precio"].loc["2025"].mean(),
                    "2026 (ene-oct obs. + nov-dic proy.)": serie.loc["2026"].mean(),
                    "Último dato (oct-2026)": r["precio"].iloc[-1],
                    "2027 proyectado (mediana)": p27["Pronóstico (mediana)"].mean(),
                    "2027 proyectado (media)": p27["Pronóstico (media)"].mean(),
                    "Dic-2027 LI 95%": p27["LI 95%"].iloc[-1], "Dic-2027 LI 80%": p27["LI 80%"].iloc[-1],
                    "Dic-2027 mediana": p27["Pronóstico (mediana)"].iloc[-1],
                    "Dic-2027 LS 80%": p27["LS 80%"].iloc[-1], "Dic-2027 LS 95%": p27["LS 95%"].iloc[-1]}
    anual = pd.DataFrame(anual).T
    anual["Var. % 2027/2026"] = (anual["2027 proyectado (mediana)"].astype(float) /
                                 anual["2026 (ene-oct obs. + nov-dic proy.)"].astype(float) - 1) * 100
    anual.index.name = "Producto"

    # --- Gráficos resumen ---
    g.panel_pronosticos(resultados, unidades, FIG / "03_panel_pronosticos_2027.png")
    g.barras_variacion(anual["Var. % 2027/2026"].astype(float), FIG / "04_variacion_2027_vs_2026.png")
    g.mapa_estacional(factores.drop(columns="Fuerza estacional (0-1)"), FIG / "05_patron_estacional.png")

    # --- Exportar tablas ---
    tablas = {"01_estadisticos_descriptivos": desc, "02_crecimiento_por_periodo": crec,
              "03_pruebas_estacionariedad": est, "04_seleccion_modelos": sel.set_index("Producto"),
              "05_coeficientes_modelos": coefs.set_index("Producto"), "06_diagnostico_residuos": diag,
              "07_factores_estacionales": factores, "08_pronostico_mensual_detalle": pron_largo.set_index("Producto"),
              "09_pronostico_mensual_mediana": pron_ancho, "10_promedios_anuales_y_proyeccion_2027": anual}
    with pd.ExcelWriter(TAB / "resultados_arima_sarima.xlsx") as xl:
        for nombre, t in tablas.items():
            t.to_csv(TAB / f"{nombre}.csv", float_format="%.4f", encoding="utf-8-sig")
            t.to_excel(xl, sheet_name=nombre[:31])
    with open(TAB / "resumen_modelos_finales.txt", "w", encoding="utf-8") as f:
        f.write(f"Modelos ARIMA/SARIMA finales - {AUTOR} - {FECHA}\n\n")
        for n, r in resultados.items():
            f.write(f"{'=' * 78}\n{n}: {r['candidatos'][r['final']]['etiqueta']} sobre log(precio)\n")
            f.write(str(r["res"].summary()) + "\n\n")
    with open(TAB / "resumen.json", "w", encoding="utf-8") as f:
        json.dump({n: {"modelo": r["candidatos"][r["final"]]["etiqueta"], "familia": r["final"]}
                   for n, r in resultados.items()}, f, ensure_ascii=False, indent=2)
    print("\nModelos finales:")
    print(sel[sel["Modelo final"] == "Sí"][["Producto", "Especificación", "BIC", f"MAPE validación {MESES_VALIDACION}m (%)", "MAPE paseo aleatorio (%)"]].to_string(index=False))
    print("\nPromedios anuales y proyección:")
    print(anual[["2025 (obs.)", "2026 (ene-oct obs. + nov-dic proy.)", "2027 proyectado (mediana)", "Var. % 2027/2026"]].to_string())
    generar_informe()
    return resultados, tablas


if __name__ == "__main__":
    main()
