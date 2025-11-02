from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st


@st.cache_data(show_spinner=False)
def evaluate_models(series: pd.Series,
                    horizon: int = 30,
                    lookback: int = 14,
                    p_values: range = range(0, 4),
                    d_values: range = range(0, 2),
                    q_values: range = range(0, 4)) -> pd.DataFrame:
    """
    留出法（最后 horizon 天作为验证集）比较 ARIMA / KNN / GLM / SVR，
    输出 MAE・RMSE・MAPE，并按 RMSE 升序排序。
    """
    series = series.asfreq('D').fillna(0)
    if len(series) <= horizon + 10:
        raise ValueError("序列太短，无法留出 %d 天进行评估。" % horizon)

    train, test = series.iloc[:-horizon], series.iloc[-horizon:]

    def _to_series_like(pred, a_index):
        if isinstance(pred, pd.Series):
            return pred.reindex(a_index)
        return pd.Series(pred, index=a_index)

    def _metrics(a: pd.Series, p) -> dict:
        p = _to_series_like(p, a.index).astype(float)
        a = a.astype(float)
        mae = mean_absolute_error(a, p)
        try:
            rmse = mean_squared_error(a, p, squared=False)
        except TypeError:
            rmse = mean_squared_error(a, p) ** 0.5
        mape = np.nanmean(np.abs((a - p) / np.where(a == 0, np.nan, a))) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    results = {}

    best_aic, best_order = float('inf'), (1, 0, 1)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    aic = ARIMA(train, order=(p, d, q)).fit().aic
                    if aic < best_aic:
                        best_aic, best_order = aic, (p, d, q)
                except Exception:
                    continue
    arima_train = train.asfreq('D').fillna(0)
    arima_pred = ARIMA(arima_train, order=best_order).fit().forecast(steps=horizon)
    results['ARIMA'] = _metrics(test, arima_pred)

    # Import local utilities to avoid circular dependencies
    from services.forecast import knn_forecast_counterfactual, fit_and_extrapolate

    try:
        knn_pred, _ = knn_forecast_counterfactual(series,
                                                  train.index[-1] + pd.Timedelta(days=1),
                                                  lookback=lookback,
                                                  horizon=horizon)
        if knn_pred is not None:
            results['KNN'] = _metrics(test, knn_pred)
    except Exception:
        pass

    try:
        glm_pred, svr_pred, _ = fit_and_extrapolate(series,
                                                    train.index[-1] + pd.Timedelta(days=1),
                                                    days=horizon)
        if glm_pred is not None:
            results['GLM'] = _metrics(test, glm_pred)
        if svr_pred is not None:
            results['SVR'] = _metrics(test, svr_pred)
    except Exception:
        pass

    return (pd.DataFrame(results)
            .T.sort_values('RMSE')
            .round(3))

