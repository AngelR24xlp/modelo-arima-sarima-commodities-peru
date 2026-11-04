"""
Genera el informe en Word (.docx) con los resultados e interpretación del modelo ARIMA/SARIMA.
Lee las tablas exportadas por main.py (output/tablas) y las figuras (output/figuras).
Autor: Angel Marchán Collazos (UNMSM) - 4 de noviembre de 2026
"""
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm, RGBColor

from . import AUTOR, INSTITUCION, FECHA
from .datos import RAIZ
from .graficos import _slug

TAB = RAIZ / "output" / "tablas"
FIG = RAIZ / "output" / "figuras"
SALIDA = RAIZ / "docs" / "Informe_ARIMA_SARIMA_materias_primas_Peru.docx"


def _leer(nombre, **kw):
    return pd.read_csv(TAB / f"{nombre}.csv", encoding="utf-8-sig", **kw)


def _fmt(v, dec=2):
    if isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:,.{dec}f}"
    return "" if pd.isna(v) else str(v)


def _tabla(doc, df: pd.DataFrame, titulo: str, dec=2, fuente=True, ancho=None):
    p = doc.add_paragraph(); r = p.add_run(titulo); r.bold = True; r.font.size = Pt(10)
    t = doc.add_table(rows=1, cols=df.shape[1], style="Light Grid Accent 1")
    for j, c in enumerate(df.columns):
        t.rows[0].cells[j].text = str(c)
    for _, fila in df.iterrows():
        celdas = t.add_row().cells
        for j, v in enumerate(fila):
            celdas[j].text = _fmt(v, dec)
    for fila in t.rows:
        for c in fila.cells:
            for par in c.paragraphs:
                for run in par.runs:
                    run.font.size = Pt(ancho or 8)
    if fuente:
        f = doc.add_paragraph(f"Fuente: Banco Mundial (Pink Sheet). Elaboración: {AUTOR}.")
        f.runs[0].font.size = Pt(7.5); f.runs[0].italic = True


def _figura(doc, archivo, pie, ancho=16):
    doc.add_picture(str(FIG / archivo), width=Cm(ancho))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(pie); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.size = Pt(8.5); p.runs[0].italic = True


def _par(doc, texto):
    p = doc.add_paragraph(texto); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def generar_informe():
    desc = _leer("01_estadisticos_descriptivos", index_col=0)
    crec = _leer("02_crecimiento_por_periodo", index_col=0)
    est = _leer("03_pruebas_estacionariedad", index_col=0)
    sel = _leer("04_seleccion_modelos")
    coef = _leer("05_coeficientes_modelos")
    diag = _leer("06_diagnostico_residuos", index_col=0)
    fac = _leer("07_factores_estacionales", index_col=0)
    pmes = _leer("09_pronostico_mensual_mediana", index_col=0)
    anual = _leer("10_promedios_anuales_y_proyeccion_2027", index_col=0)
    productos = list(desc.index)
    fin = sel[sel["Modelo final"] == "Sí"].set_index("Producto")
    col_mape = [c for c in sel.columns if c.startswith("MAPE validación")][0]
    gana_rw = [p for p in productos if fin.loc[p, col_mape] < fin.loc[p, "MAPE paseo aleatorio (%)"]]
    n_arima = int((fin["Familia"] == "ARIMA").sum())
    var = anual["Var. % 2027/2026"].astype(float).sort_values(ascending=False)

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    doc.core_properties.author = AUTOR
    doc.core_properties.title = "Modelo ARIMA/SARIMA de precios de materias primas exportadas por el Perú"

    # Portada
    for _ in range(5):
        doc.add_paragraph()
    for txt, tam, neg in [(INSTITUCION, 14, True), ("Facultad de Ciencias Económicas", 12, False), ("", 12, False),
                          ("MODELO ECONOMÉTRICO ARIMA / SARIMA DE LOS PRECIOS DE LAS PRINCIPALES MATERIAS PRIMAS "
                           "QUE EXPORTA EL PERÚ", 18, True),
                          ("Evolución mensual 2000-2026 y proyección para 2027", 13, False), ("", 12, False),
                          (f"Autor: {AUTOR}", 12, True), (FECHA, 12, False)]:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(txt); r.font.size = Pt(tam); r.bold = neg
        if tam == 18:
            r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    doc.add_page_break()

    # 1. Introducción
    doc.add_heading("1. Introducción", 1)
    _par(doc, "El Perú es una economía pequeña y abierta cuya canasta exportadora está dominada por productos "
              "primarios, principalmente minerales (cobre, oro, zinc, plata, plomo, estaño, hierro) y, en menor medida, "
              "productos pesqueros (harina de pescado) y agrícolas tradicionales (café). Por ello, la evolución de sus "
              "precios internacionales determina en gran medida los términos de intercambio, la balanza comercial, "
              "la recaudación fiscal (impuesto a la renta minero, canon y regalías) y el ciclo económico del país.")
    _par(doc, "El objetivo de este trabajo es modelar econométricamente la dinámica de los precios mensuales de "
              f"{len(productos)} materias primas de exportación peruana entre enero de 2000 y octubre de 2026 mediante "
              "la metodología Box-Jenkins (modelos ARIMA y su extensión estacional SARIMA), interpretar los resultados "
              "y proyectar su evolución mensual hasta diciembre de 2027.")

    # 2. Datos
    doc.add_heading("2. Datos", 1)
    _par(doc, "Se utilizan precios nominales mensuales en dólares corrientes del Commodity Price Data (Pink Sheet) del "
              "Banco Mundial, actualizado al 3 de noviembre de 2026. La muestra comprende 322 observaciones mensuales "
              "(2000M01-2026M10). Los meses de noviembre y diciembre de 2026 y todo 2027 son proyecciones del modelo. "
              "Las referencias son: cobre, zinc, plomo y estaño (LME, US$/tm), oro y plata (Londres, US$/oz troy), "
              "mineral de hierro (CFR China, US$/tms), harina de pescado (64-65% proteína, Perú/Hamburgo, US$/tm) y "
              "café arábica (US$/kg). El molibdeno, también relevante en la canasta peruana, no se incluye por no estar "
              "disponible en la fuente. Nota: hasta 2010 el precio del hierro se fijaba mediante contratos anuales de "
              "referencia, por lo que la serie muestra tramos planos en esos años.")
    _figura(doc, "01_series_precios_2000_2026.png", "Figura 1. Precios mensuales 2000-2026 (sombreado: crisis financiera 2008-09 y COVID-19).")
    _figura(doc, "02_indice_base100.png", "Figura 2. Índice de precios (promedio 2000 = 100, escala logarítmica).")
    d = desc[["Unidad", "Promedio", "Mínimo", "Fecha mínimo", "Máximo", "Fecha máximo", "Precio último",
              "Var. acumulada (%)", "Crec. anual compuesto (%)", "Volatilidad anualizada (%)"]].reset_index()
    _tabla(doc, d, "Tabla 1. Estadísticos descriptivos, ene-2000 a oct-2026", dec=1)

    # 3. Evolución 2000-2026
    doc.add_heading("3. Evolución de los precios 2000-2026: interpretación", 1)
    top = desc["Var. acumulada (%)"].sort_values(ascending=False)
    _par(doc, f"Entre enero de 2000 y octubre de 2026 todos los precios analizados aumentaron de forma significativa en "
              f"términos nominales. Los mayores incrementos acumulados corresponden a {top.index[0].lower()} "
              f"(+{top.iloc[0]:,.0f}%), {top.index[1].lower()} (+{top.iloc[1]:,.0f}%), {top.index[2].lower()} "
              f"(+{top.iloc[2]:,.0f}%) y {top.index[3].lower()} (+{top.iloc[3]:,.0f}%). El cobre, principal producto de "
              f"exportación del Perú, pasó de US$ {desc.loc['Cobre', 'Precio ene-2000']:,.0f}/tm a "
              f"US$ {desc.loc['Cobre', 'Precio último']:,.0f}/tm (crecimiento anual compuesto de "
              f"{desc.loc['Cobre', 'Crec. anual compuesto (%)']:.1f}%), mientras que el oro registró un crecimiento anual "
              f"de {desc.loc['Oro', 'Crec. anual compuesto (%)']:.1f}%.")
    _par(doc, "La trayectoria no ha sido lineal; se distinguen claramente varios ciclos (Tabla 2):")
    for t in [
        "2003-2008, superciclo de materias primas: la industrialización y urbanización de China elevó la demanda de "
        f"metales; el cobre creció {crec.loc['2003-2008 (superciclo)', 'Cobre']:.1f}% anual, el plomo "
        f"{crec.loc['2003-2008 (superciclo)', 'Plomo']:.1f}% y el hierro {crec.loc['2003-2008 (superciclo)', 'Hierro']:.1f}%.",
        "2008-2009, crisis financiera internacional: caída abrupta de los metales básicos (el cobre perdió más de la mitad "
        "de su valor entre mediados de 2008 y comienzos de 2009), mientras el oro actuó como activo refugio.",
        "2009-2011, rebote post-crisis: estímulos fiscales y monetarios y la demanda china llevaron a nuevos máximos "
        f"(plata +{crec.loc['2009-2011 (rebote post-crisis)', 'Plata']:.1f}% anual, hierro "
        f"+{crec.loc['2009-2011 (rebote post-crisis)', 'Hierro']:.1f}%).",
        "2012-2015, fin del superciclo: desaceleración china, fortalecimiento del dólar y mayor oferta minera "
        f"(hierro {crec.loc['2012-2015 (fin del superciclo)', 'Hierro']:.1f}% anual, plata "
        f"{crec.loc['2012-2015 (fin del superciclo)', 'Plata']:.1f}%, cobre {crec.loc['2012-2015 (fin del superciclo)', 'Cobre']:.1f}%).",
        "2016-2019, recuperación moderada, y 2020 choque del COVID-19 seguido de un fuerte rebote en 2021-2022 por "
        "estímulos, restricciones de oferta y la transición energética.",
        "2023-2026, nuevo auge: los metales preciosos alcanzaron máximos históricos (oro "
        f"US$ {desc.loc['Oro', 'Máximo']:,.0f}/oz en {desc.loc['Oro', 'Fecha máximo']}, plata US$ {desc.loc['Plata', 'Máximo']:.1f}/oz "
        f"en {desc.loc['Plata', 'Fecha máximo']}), al igual que el cobre (US$ {desc.loc['Cobre', 'Máximo']:,.0f}/tm en "
        f"{desc.loc['Cobre', 'Fecha máximo']}), el estaño y la harina de pescado; en cambio, el plomo y el hierro "
        "mostraron precios estables o decrecientes.",
    ]:
        doc.add_paragraph(t, style="List Bullet")
    _tabla(doc, crec.reset_index(), "Tabla 2. Crecimiento anual compuesto del precio promedio por subperiodo (%)", dec=1)
    _par(doc, f"La volatilidad anualizada de los rendimientos mensuales es elevada: va desde "
              f"{desc['Volatilidad anualizada (%)'].min():.1f}% ({desc['Volatilidad anualizada (%)'].idxmin().lower()}) "
              f"hasta {desc['Volatilidad anualizada (%)'].max():.1f}% ({desc['Volatilidad anualizada (%)'].idxmax().lower()}), "
              "lo que confirma la alta exposición de los ingresos de exportación peruanos a choques externos.")

    # 4. Metodología
    doc.add_heading("4. Metodología econométrica", 1)
    _par(doc, "Se modela el logaritmo natural del precio, y_t = ln(P_t), de modo que las diferencias se interpretan como "
              "tasas de variación mensual y los intervalos de pronóstico son asimétricos y siempre positivos. "
              "El modelo SARIMA(p,d,q)x(P,D,Q)_12 es:")
    p = doc.add_paragraph("φ(L) Φ(L¹²) (1 − L)^d (1 − L¹²)^D y_t = c + θ(L) Θ(L¹²) ε_t,   ε_t ~ RB(0, σ²)")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.runs[0].italic = True
    _par(doc, "donde φ(L) y θ(L) son los polinomios autorregresivo y de medias móviles regulares, Φ(L¹²) y Θ(L¹²) los "
              "estacionales y c la deriva (crecimiento medio mensual cuando d = 1). El ARIMA es el caso particular sin "
              "componentes estacionales. Las etapas fueron:")
    for t in ["Identificación: pruebas de raíz unitaria ADF (H0: raíz unitaria) y KPSS (H0: estacionariedad) en niveles y "
              "en primeras diferencias; pruebas de raíz unitaria estacional OCSB y Canova-Hansen; fuerza de la "
              "estacionalidad con la descomposición STL; FAS y FAP de la serie diferenciada.",
              "Estimación: máxima verosimilitud (filtro de Kalman, statsmodels SARIMAX). Los órdenes (p, q, P, Q) se "
              "buscaron con el algoritmo de Hyndman-Khandakar (pmdarima) minimizando el AIC, por separado para la "
              "familia ARIMA y para la familia SARIMA (con d = 1).",
              "Selección: entre el mejor ARIMA y el mejor SARIMA se elige el de menor BIC (criterio más parsimonioso).",
              "Diagnóstico: Ljung-Box (autocorrelación residual), Jarque-Bera (normalidad) y ARCH-LM (heterocedasticidad condicional).",
              "Validación fuera de muestra: se re-estima con datos hasta oct-2025 y se pronostican los 12 meses "
              "nov-2025 a oct-2026, comparando el MAPE con el de un paseo aleatorio (pronóstico ingenuo).",
              "Pronóstico: nov-2026 a dic-2027 (14 meses) con intervalos de confianza de 80% y 95%. Se reporta la mediana "
              "(exp de la media del log) como pronóstico puntual y también la media log-normal."]:
        doc.add_paragraph(t, style="List Number")
    _figura(doc, "acf_pacf_cobre.png", "Figura 3. Identificación Box-Jenkins para el cobre (las figuras de los demás productos están en output/figuras).")

    # 5. Resultados
    doc.add_heading("5. Resultados de la estimación", 1)
    doc.add_heading("5.1 Estacionariedad y estacionalidad", 2)
    e = est[["ADF nivel (estad.)", "ADF nivel (p)", "ADF dif. (estad.)", "ADF dif. (p)", "KPSS nivel (estad.)",
             "KPSS dif. (estad.)", "D sugerido (OCSB)", "D sugerido (CH)"]].reset_index().rename(columns={"index": "Producto"})
    _tabla(doc, e, "Tabla 3. Pruebas de raíz unitaria sobre ln(precio) (ADF con constante y tendencia en niveles; KPSS valor crítico 5%: 0.146 nivel / 0.463 dif.)", dec=3)
    _par(doc, "En todos los casos la prueba ADF no rechaza la hipótesis de raíz unitaria en niveles (p-valores entre "
              f"{est['ADF nivel (p)'].min():.2f} y {est['ADF nivel (p)'].max():.2f}) y la rechaza contundentemente en "
              "primeras diferencias (p < 0.01); la KPSS rechaza la estacionariedad en niveles y no la rechaza en "
              "diferencias. Por tanto, los log-precios son integrados de orden uno, I(1): los choques sobre los precios "
              "tienen efectos permanentes y no existe un nivel de 'equilibrio' al que retornen mecánicamente. Esto "
              "justifica d = 1.")
    _par(doc, "Las pruebas OCSB y Canova-Hansen indican D = 0 para todos los productos (no hay raíz unitaria estacional) y "
              f"la fuerza estacional STL es prácticamente nula (máximo {fac['Fuerza estacional (0-1)'].max():.2f} en "
              f"{fac['Fuerza estacional (0-1)'].idxmax().lower()}, en una escala de 0 a 1). Los factores estacionales "
              "promedio (Figura 4) son pequeños comparados con la volatilidad de los precios. La explicación económica es "
              "que se trata de bienes almacenables, transados en mercados globales con mercados de futuros: si existiera "
              "un patrón estacional predecible, el arbitraje (comprar y almacenar en los meses baratos) lo eliminaría.")
    _figura(doc, "05_patron_estacional.png", "Figura 4. Factores estacionales promedio (STL), % respecto de la tendencia.")

    doc.add_heading("5.2 Selección de modelos: ARIMA vs SARIMA", 2)
    s = sel[["Producto", "Especificación", "AIC", "BIC", col_mape, "MAPE paseo aleatorio (%)", "Modelo final"]].fillna("")
    _tabla(doc, s, "Tabla 4. Comparación ARIMA vs SARIMA (criterios de información y validación fuera de muestra)", dec=2)
    aic_sar = [p for p in productos
               if sel[(sel["Producto"] == p) & (sel["Familia"] == "SARIMA")]["AIC"].iloc[0]
               < sel[(sel["Producto"] == p) & (sel["Familia"] == "ARIMA")]["AIC"].iloc[0]]
    val_sar = [p for p in productos
               if sel[(sel["Producto"] == p) & (sel["Familia"] == "SARIMA")][col_mape].iloc[0]
               < sel[(sel["Producto"] == p) & (sel["Familia"] == "ARIMA")][col_mape].iloc[0] - 1]
    _par(doc, f"En {n_arima} de {len(productos)} productos el BIC favorece la especificación ARIMA no estacional: los "
              "parámetros estacionales adicionales no mejoran el ajuste lo suficiente como para compensar la pérdida de "
              "parsimonia. "
              + (f"El AIC, menos exigente, prefiere el SARIMA solo en {' y '.join(x.lower() for x in aic_sar)}, con "
                 "diferencias pequeñas (menos de 6 puntos). " if aic_sar else "El AIC conduce a la misma conclusión. ")
              + "Esto es coherente con la ausencia de estacionalidad documentada en 5.1. Cuando la búsqueda automática "
              "no encontró componente estacional, se estimó un SARIMA(·)x(1,0,1)[12] para contrastarlo formalmente con "
              "el ARIMA. En la validación fuera de muestra, el error de ambos modelos es prácticamente el mismo"
              + (f", salvo en {' y '.join(x.lower() for x in val_sar)}, donde el SARIMA pronosticó mejor"
                 + (" (el café es un cultivo con un calendario de cosechas definido)" if "Café arábica" in val_sar else "")
                 + "; aun así, su ventaja en BIC no fue suficiente para seleccionarlo." if val_sar else "."))

    doc.add_heading("5.3 Coeficientes estimados", 2)
    _tabla(doc, coef, "Tabla 5. Coeficientes de los modelos finales (variable dependiente: Δln precio)", dec=4)
    ar1 = coef[coef["Parámetro"] == "ar.L1"].set_index("Producto")["Coeficiente"]
    der = coef[coef["Parámetro"].str.startswith("deriva")].set_index("Producto")
    txt_der = "; ".join(f"{p.lower()}: {v:.4f} mensual (≈{(np.exp(v * 12) - 1) * 100:.1f}% anual, p = {der.loc[p, 'p-valor']:.3f})"
                        for p, v in der["Coeficiente"].items())
    _par(doc, "Interpretación: los coeficientes AR(1) positivos y significativos "
              f"({', '.join(f'{p.lower()} {v:.2f}' for p, v in ar1.items() if abs(v) > 0.1)}) indican inercia o "
              "'momentum' de corto plazo: una fracción de la variación de precio de un mes se traslada al mes siguiente "
              f"(en el cobre, alrededor del {ar1.get('Cobre', np.nan) * 100:.0f}%). En el oro y el hierro la dinámica se "
              "captura con un término MA(1) positivo, que implica que el choque de un mes se refleja parcialmente en el "
              "siguiente y luego se disipa. Los parámetros de deriva estimados son: " + txt_der + ". Solo la deriva del "
              "oro es estadísticamente significativa al 5%, reflejando su tendencia alcista sostenida desde 2001. En el "
              "resto de productos no se detecta una deriva significativa: una vez diferenciados, sus precios se comportan "
              "aproximadamente como un paseo aleatorio con algo de autocorrelación de corto plazo, lo cual es consistente "
              "con la hipótesis de mercados eficientes en los mercados de commodities.")

    doc.add_heading("5.4 Diagnóstico de residuos", 2)
    _tabla(doc, diag.reset_index().rename(columns={"index": "Producto"}), "Tabla 6. Pruebas de diagnóstico sobre los residuos", dec=3)
    sin_ac = [p for p in productos if diag.loc[p, "p Q(12)"] > 0.05]
    con_arch = [p for p in productos if diag.loc[p, "p ARCH"] < 0.05]
    _par(doc, f"El test de Ljung-Box no rechaza la ausencia de autocorrelación residual a 12 rezagos en "
              f"{len(sin_ac)} de {len(productos)} productos, por lo que los modelos capturan adecuadamente la dinámica "
              "lineal" + ("" if len(sin_ac) == len(productos) else
                          f" (la excepción es {', '.join(x.lower() for x in productos if x not in sin_ac)}, con autocorrelación "
                          "débil remanente en rezagos intermedios)") + ". Jarque-Bera rechaza la normalidad en casi todos "
              "los casos por colas pesadas (meses de crisis como 2008 y 2020), y el test ARCH-LM detecta "
              f"heterocedasticidad condicional en {', '.join(x.lower() for x in con_arch)}: la volatilidad se agrupa en "
              "periodos. Esto no sesga el pronóstico puntual, pero implica que los intervalos de confianza (que suponen "
              "varianza constante) pueden subestimar el riesgo en periodos turbulentos. Una extensión natural es un "
              "modelo ARIMA-GARCH.")
    _figura(doc, "diagnostico_cobre.png", "Figura 5. Diagnóstico de residuos del modelo del cobre.")

    doc.add_heading("5.5 Validación fuera de muestra (nov-2025 a oct-2026)", 2)
    _par(doc, f"El modelo seleccionado obtuvo menor MAPE que el paseo aleatorio en {len(gana_rw)} de {len(productos)} "
              f"productos ({', '.join(x.lower() for x in gana_rw)}). Sin embargo, los errores son elevados en los "
              f"productos que tuvieron un rally extraordinario durante 2026 (cobre {fin.loc['Cobre', col_mape]:.1f}%, plata "
              f"{fin.loc['Plata', col_mape]:.1f}%, estaño {fin.loc['Estaño', col_mape]:.1f}%): un modelo univariado no puede "
              "anticipar choques de demanda, oferta o financieros que no están contenidos en la historia de la serie. "
              "Esto debe tenerse en cuenta al leer las proyecciones para 2027.")

    # 6. Proyección 2027
    doc.add_heading("6. Proyección para 2027", 1)
    _figura(doc, "03_panel_pronosticos_2027.png", "Figura 6. Proyección mensual a diciembre de 2027 (mediana e intervalos de 80% y 95%).")
    a = anual[["Unidad", "2025 (obs.)", "2026 (ene-oct obs. + nov-dic proy.)", "Último dato (oct-2026)",
               "2027 proyectado (mediana)", "Var. % 2027/2026", "Dic-2027 LI 95%", "Dic-2027 mediana", "Dic-2027 LS 95%"]].reset_index()
    _tabla(doc, a, "Tabla 7. Precios promedio anuales observados y proyectados para 2027", dec=2)
    _figura(doc, "04_variacion_2027_vs_2026.png", "Figura 7. Variación % del precio promedio 2027 (proyectado) respecto de 2026.", ancho=13)
    sube = [f"{p.lower()} ({v:+.1f}%)" for p, v in var.items() if v > 1]
    baja = [f"{p.lower()} ({v:+.1f}%)" for p, v in var.items() if v < -1]
    estable = [f"{p.lower()} ({v:+.1f}%)" for p, v in var.items() if -1 <= v <= 1]
    _par(doc, "Comparando el precio promedio proyectado de 2027 con el promedio de 2026, el modelo anticipa aumentos en "
              f"{', '.join(sube)}; estabilidad en {', '.join(estable) if estable else 'ningún producto'}; y descensos en "
              f"{', '.join(baja)}.")
    _par(doc, "Es fundamental interpretar correctamente estas cifras. Como los log-precios son I(1) sin deriva "
              "significativa (salvo el oro), el pronóstico ARIMA converge rápidamente al último nivel observado "
              "(octubre de 2026) y, en los modelos con deriva, crece a una tasa constante. Por ello, las variaciones "
              "2027/2026 reflejan principalmente un efecto arrastre estadístico: los productos que cerraron 2026 por encima "
              "de su promedio anual (cobre, zinc y harina de pescado, que alcanzaron máximos en el segundo semestre) "
              "muestran alzas, y los que cerraron por debajo de su promedio (plata e hierro, tras sus picos de inicios de "
              "2026) muestran caídas, aun cuando el modelo no predice un cambio de tendencia. En el caso del oro, el "
              "promedio 2027 resulta ligeramente inferior al de 2026 (año con un pico de US$ 5,020/oz en febrero), pero la "
              f"deriva positiva lleva el precio de US$ {anual.loc['Oro', 'Último dato (oct-2026)']:,.0f} en oct-2026 a "
              f"US$ {anual.loc['Oro', 'Dic-2027 mediana']:,.0f} en dic-2027.")
    _par(doc, f"Los intervalos de confianza son amplios: para el cobre, el intervalo de 95% a diciembre de 2027 va de "
              f"US$ {anual.loc['Cobre', 'Dic-2027 LI 95%']:,.0f} a US$ {anual.loc['Cobre', 'Dic-2027 LS 95%']:,.0f}/tm, "
              "y para el oro de "
              f"US$ {anual.loc['Oro', 'Dic-2027 LI 95%']:,.0f} a US$ {anual.loc['Oro', 'Dic-2027 LS 95%']:,.0f}/oz. Esta "
              "incertidumbre es inherente a procesos con raíz unitaria (la varianza del error de pronóstico crece con el "
              "horizonte) y debe comunicarse junto con la proyección puntual.")
    pm = pmes.copy(); pm.index.name = "Mes"
    _tabla(doc, pm.reset_index(), "Tabla 8. Pronóstico mensual (mediana) nov-2026 a dic-2027", dec=2, ancho=7.5)
    for prod in productos:
        _figura(doc, f"pronostico_{_slug(prod)}.png", f"Pronóstico de {prod.lower()} con validación fuera de muestra e intervalos de confianza.", ancho=15)

    # 7. Conclusiones
    doc.add_heading("7. Conclusiones e implicancias para el Perú", 1)
    for t in [
        "Los precios de las principales materias primas exportadas por el Perú se multiplicaron entre 2000 y 2026, con "
        "ciclos marcados (superciclo 2003-2011, corrección 2012-2015, auge 2021-2022 y nuevo auge 2023-2026). En 2026 el "
        "cobre, el oro, la plata, el estaño y la harina de pescado alcanzaron máximos históricos nominales.",
        "Todas las series (en logaritmos) son I(1) y no presentan estacionalidad significativa; por ello los modelos "
        "ARIMA no estacionales dominan a los SARIMA según los criterios de información, con desempeño predictivo similar.",
        "La dinámica estimada es cercana a un paseo aleatorio con inercia de corto plazo (AR o MA de orden bajo) y, en el "
        "caso del oro, con una deriva positiva significativa (~10% anual).",
        f"Para 2027 el modelo proyecta que el cobre se mantendría en torno a US$ {anual.loc['Cobre', '2027 proyectado (mediana)']:,.0f}/tm "
        f"en promedio ({anual.loc['Cobre', 'Var. % 2027/2026']:+.1f}% respecto de 2026) y el oro alrededor de "
        f"US$ {anual.loc['Oro', '2027 proyectado (mediana)']:,.0f}/oz, niveles históricamente altos que favorecerían los "
        "términos de intercambio, las exportaciones y la recaudación minera (impuesto a la renta, canon y regalías).",
        "El riesgo es considerable: la volatilidad es alta, existe agrupamiento de volatilidad (efectos ARCH) y los "
        "intervalos de 95% admiten tanto caídas como alzas de 40% o más. La política fiscal debería tratar los ingresos "
        "extraordinarios como transitorios (regla fiscal, fondo de estabilización).",
        "Limitaciones y extensiones: los modelos univariados no incorporan fundamentos (crecimiento de China, dólar, "
        "tasas de interés, inventarios, transición energética). Se recomienda complementar con modelos ARIMAX/VAR, "
        "ARIMA-GARCH para la volatilidad y precios en términos reales.",
    ]:
        doc.add_paragraph(t, style="List Bullet")

    doc.add_heading("Referencias", 1)
    for t in ["Banco Mundial (2026). Commodity Price Data (The Pink Sheet). Actualización del 3 de noviembre de 2026.",
              "Box, G., Jenkins, G., Reinsel, G. y Ljung, G. (2015). Time Series Analysis: Forecasting and Control. 5.ª ed. Wiley.",
              "Hyndman, R. y Athanasopoulos, G. (2021). Forecasting: Principles and Practice. 3.ª ed. OTexts.",
              "Hyndman, R. y Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of Statistical Software, 27(3).",
              "Wang, X., Smith, K. y Hyndman, R. (2006). Characteristic-based clustering for time series data. Data Mining and Knowledge Discovery, 13(3).",
              "Seabold, S. y Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python."]:
        doc.add_paragraph(t, style="List Bullet")

    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.footer.paragraphs[0].text = f"{AUTOR} – UNMSM – {FECHA}"
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    doc.save(SALIDA)
    print(f"Informe generado: {SALIDA}")


if __name__ == "__main__":
    generar_informe()
