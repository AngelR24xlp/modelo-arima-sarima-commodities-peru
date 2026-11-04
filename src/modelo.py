"""
Estimación econométrica ARIMA / SARIMA de precios de materias primas.
Metodología Box-Jenkins: identificación (raíz unitaria, estacionalidad), estimación,
diagnóstico, validación fuera de muestra y pronóstico.
Autor: Angel Marchán Collazos (UNMSM) - 4 de noviembre de 2026
"""
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

FIN_PRONOSTICO = "2027-12-01"
MESES_VALIDACION = 12


def pruebas_estacionariedad(y: pd.Series) -> dict:
    """ADF y KPSS sobre el log-precio en niveles y en primeras diferencias."""
    dy = y.diff().dropna()
    adf_n = adfuller(y, regression="ct", autolag="AIC")
    adf_d = adfuller(dy, regression="c", autolag="AIC")
    kpss_n = kpss(y, regression="ct", nlags="auto")
    kpss_d = kpss(dy, regression="c", nlags="auto")
    return {
        "ADF nivel (estad.)": adf_n[0], "ADF nivel (p)": adf_n[1],
        "ADF dif. (estad.)": adf_d[0], "ADF dif. (p)": adf_d[1],
        "KPSS nivel (estad.)": kpss_n[0], "KPSS nivel (p)": kpss_n[1],
        "KPSS dif. (estad.)": kpss_d[0], "KPSS dif. (p)": kpss_d[1],
        "d sugerido (ADF)": pm.arima.ndiffs(y, test="adf"),
        "d sugerido (KPSS)": pm.arima.ndiffs(y, test="kpss"),
        "D sugerido (OCSB)": pm.arima.nsdiffs(y, m=12, test="ocsb"),
        "D sugerido (CH)": pm.arima.nsdiffs(y, m=12, test="ch"),
    }


def fuerza_estacional(y: pd.Series) -> tuple[float, pd.Series]:
    """Fuerza de la estacionalidad (Wang, Smith y Hyndman, 2006) y factores estacionales STL (%)."""
    res = STL(y, period=12, robust=True).fit()
    fs = max(0.0, 1 - np.var(res.resid) / np.var(res.resid + res.seasonal))
    factores = res.seasonal.groupby(res.seasonal.index.month).mean() * 100
    return fs, factores


def seleccionar(y: pd.Series, estacional: bool):
    """Búsqueda automática (Hyndman-Khandakar) por AIC con d=1 fijo."""
    return pm.auto_arima(
        y, d=1, seasonal=estacional, m=12 if estacional else 1, D=0 if estacional else None,
        max_p=4, max_q=4, max_P=2, max_Q=2, information_criterion="aic",
        stepwise=True, suppress_warnings=True, error_action="ignore",
    )


def ajustar(y: pd.Series, orden, orden_est, tendencia: str):
    # Inicialización estacionaria: ARIMA y SARIMA usan la misma muestra efectiva en la verosimilitud,
    # de modo que sus AIC/BIC son comparables.
    mod = SARIMAX(y, order=orden, seasonal_order=orden_est, trend=tendencia,
                  enforce_stationarity=True, enforce_invertibility=True)
    return mod.fit(disp=False, maxiter=200)


def tendencia_de(auto) -> str:
    # En SARIMAX con d=1, la constante actúa sobre la serie diferenciada: es la deriva mensual
    # del log-precio (crecimiento medio). trend="t" generaría una deriva creciente en el tiempo.
    return "c" if auto.with_intercept else "n"


def etiqueta(orden, orden_est, tendencia) -> str:
    txt = f"ARIMA{tuple(orden)}"
    if orden_est and orden_est[3] > 1 and any(orden_est[:3]):
        txt = f"SARIMA{tuple(orden)}x{tuple(orden_est[:3])}[12]"
    return txt + (" con deriva" if tendencia == "c" else "")


def diagnostico(res) -> dict:
    e = pd.Series(res.resid).iloc[13:]
    lb12 = acorr_ljungbox(e, lags=[12], return_df=True).iloc[0]
    lb24 = acorr_ljungbox(e, lags=[24], return_df=True).iloc[0]
    jb = stats.jarque_bera(e)
    arch = het_arch(e, nlags=12)
    return {
        "Ljung-Box Q(12)": lb12["lb_stat"], "p Q(12)": lb12["lb_pvalue"],
        "Ljung-Box Q(24)": lb24["lb_stat"], "p Q(24)": lb24["lb_pvalue"],
        "Jarque-Bera": jb.statistic, "p JB": jb.pvalue,
        "ARCH-LM(12)": arch[0], "p ARCH": arch[1],
    }


def metricas(real: pd.Series, pred: pd.Series) -> dict:
    err = real - pred
    return {"RMSE": float(np.sqrt((err ** 2).mean())),
            "MAE": float(err.abs().mean()),
            "MAPE (%)": float((err.abs() / real).mean() * 100)}


def validar(y: pd.Series, orden, orden_est, tendencia):
    """Re-estima con datos hasta h meses antes del final y pronostica h pasos (multi-paso)."""
    h = MESES_VALIDACION
    ent, prueba = y.iloc[:-h], y.iloc[-h:]
    res = ajustar(ent, orden, orden_est, tendencia)
    pred = np.exp(res.get_forecast(h).predicted_mean)
    pred.index = prueba.index
    ingenuo = pd.Series(np.exp(ent.iloc[-1]), index=prueba.index)
    return pred, metricas(np.exp(prueba), pred), metricas(np.exp(prueba), ingenuo)


def pronosticar(res, y: pd.Series) -> pd.DataFrame:
    pasos = len(pd.date_range(y.index[-1], FIN_PRONOSTICO, freq="MS")) - 1
    f = res.get_forecast(pasos)
    media = f.predicted_mean
    ci95, ci80 = f.conf_int(alpha=0.05), f.conf_int(alpha=0.20)
    var = f.var_pred_mean
    return pd.DataFrame({
        "Pronóstico (mediana)": np.exp(media),
        "Pronóstico (media)": np.exp(media + var / 2),
        "LI 80%": np.exp(ci80.iloc[:, 0]), "LS 80%": np.exp(ci80.iloc[:, 1]),
        "LI 95%": np.exp(ci95.iloc[:, 0]), "LS 95%": np.exp(ci95.iloc[:, 1]),
    }, index=media.index)


def analizar_producto(precio: pd.Series) -> dict:
    y = np.log(precio).asfreq("MS")
    y.name = precio.name
    est = pruebas_estacionariedad(y)
    fs, factores = fuerza_estacional(y)

    candidatos = {}
    for nombre, estacional in (("ARIMA", False), ("SARIMA", True)):
        auto = seleccionar(y, estacional)
        orden, orden_est, tend = auto.order, auto.seasonal_order, tendencia_de(auto)
        if estacional and not any(orden_est[:3]):
            # Si la búsqueda no encuentra componente estacional, se fuerza SARIMA(1,0,1)[12]
            # para poder contrastarlo formalmente con el ARIMA no estacional.
            orden_est = (1, 0, 1, 12)
        res = ajustar(y, orden, orden_est, tend)
        pred_val, m_mod, m_ing = validar(y, orden, orden_est, tend)
        candidatos[nombre] = dict(orden=orden, orden_est=orden_est, tendencia=tend, res=res,
                                  etiqueta=etiqueta(orden, orden_est, tend),
                                  aic=res.aic, bic=res.bic, val_pred=pred_val,
                                  val=m_mod, val_ingenuo=m_ing)

    # Modelo final: menor BIC (criterio parsimonioso); se reporta también la validación
    final = min(candidatos, key=lambda k: candidatos[k]["bic"])
    c = candidatos[final]
    return dict(y=y, precio=precio, estacionariedad=est, fuerza_estacional=fs, factores=factores,
                candidatos=candidatos, final=final, res=c["res"],
                diagnostico=diagnostico(c["res"]), pronostico=pronosticar(c["res"], y))
