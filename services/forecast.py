from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import ARIMA_P, ARIMA_D, ARIMA_Q, MAX_PRE_DAYS


@st.cache_data(show_spinner=False)
def evaluate_arima_model(series, arima_order):
    try:
        model = ARIMA(series, order=arima_order)
        model_fit = model.fit()
        return model_fit.aic
    except Exception:
        return float("inf")


@st.cache_data(show_spinner=False)
def arima_forecast_with_grid_search(accident_series: pd.Series,
                                    start_date: pd.Timestamp,
                                    horizon: int = 30,
                                    p_values: list = tuple(ARIMA_P),
                                    d_values: list = tuple(ARIMA_D),
                                    q_values: list = tuple(ARIMA_Q)) -> pd.DataFrame:
    series = accident_series.asfreq('D').fillna(0)
    start_date = pd.to_datetime(start_date)

    warnings.filterwarnings("ignore", category=ValueWarning)
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    aic = evaluate_arima_model(series, order)
                    if aic < best_score:
                        best_score, best_cfg = aic, order
                except Exception:
                    continue

    model = ARIMA(series, order=best_cfg)
    fit = model.fit()
    forecast_index = pd.date_range(start=start_date, periods=horizon, freq='D')
    res = fit.get_forecast(steps=horizon)
    df = res.summary_frame()
    df.index = forecast_index
    df.index.name = 'date'
    df.rename(columns={'mean': 'forecast'}, inplace=True)
    return df


def knn_forecast_counterfactual(accident_series: pd.Series,
                                intervention_date: pd.Timestamp,
                                lookback: int = 14,
                                horizon: int = 30):
    series = accident_series.asfreq('D').fillna(0)
    intervention_date = pd.to_datetime(intervention_date).normalize()

    df = pd.DataFrame({'y': series})
    for i in range(1, lookback + 1):
        df[f'lag_{i}'] = df['y'].shift(i)

    train = df.loc[:intervention_date - pd.Timedelta(days=1)].dropna()
    if len(train) < 5:
        return None, None
    X_train = train.filter(like='lag_').values
    y_train = train['y'].values
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    history = df.loc[:intervention_date - pd.Timedelta(days=1), 'y'].tolist()
    preds = []
    for _ in range(horizon):
        if len(history) < lookback:
            return None, None
        x = np.array(history[-lookback:][::-1]).reshape(1, -1)
        pred = knn.predict(x)[0]
        preds.append(pred)
        history.append(pred)

    pred_index = pd.date_range(intervention_date, periods=horizon, freq='D')
    return pd.Series(preds, index=pred_index, name='knn_pred'), None


def fit_and_extrapolate(series: pd.Series,
                        intervention_date: pd.Timestamp,
                        days: int = 30,
                        max_pre_days: int = MAX_PRE_DAYS):
    series = series.asfreq('D').fillna(0)
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    intervention_date = pd.to_datetime(intervention_date).tz_localize(None).normalize()

    pre = series.loc[:intervention_date - pd.Timedelta(days=1)]
    if len(pre) > max_pre_days:
        pre = pre.iloc[-max_pre_days:]
    if len(pre) < 3:
        return None, None, None

    x_pre = np.arange(len(pre))
    x_future = np.arange(len(pre), len(pre) + days)

    try:
        X_pre_glm = sm.add_constant(np.column_stack([x_pre, x_pre**2]))
        glm = sm.GLM(pre.values, X_pre_glm, family=sm.families.Poisson())
        glm_res = glm.fit()
        X_future_glm = sm.add_constant(np.column_stack([x_future, x_future**2]))
        glm_pred = glm_res.predict(X_future_glm)
    except Exception:
        glm_pred = None

    try:
        svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, gamma=0.1))
        svr.fit(x_pre.reshape(-1, 1), pre.values)
        svr_pred = svr.predict(x_future.reshape(-1, 1))
    except Exception:
        svr_pred = None

    post_index = pd.date_range(intervention_date, periods=days, freq='D')

    glm_pred = pd.Series(glm_pred, index=post_index, name='glm_pred') if glm_pred is not None else None
    svr_pred = pd.Series(svr_pred, index=post_index, name='svr_pred') if svr_pred is not None else None

    post = series.reindex(post_index)
    residuals = None
    if svr_pred is not None:
        residuals = pd.Series(post.values - svr_pred[:len(post)], index=post_index, name='residual')

    return glm_pred, svr_pred, residuals

