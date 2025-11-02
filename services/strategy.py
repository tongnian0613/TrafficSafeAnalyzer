from __future__ import annotations

import pandas as pd
import streamlit as st

from services.forecast import fit_and_extrapolate, arima_forecast_with_grid_search
from config.settings import MIN_PRE_DAYS, MAX_PRE_DAYS


def evaluate_strategy_effectiveness(actual_series: pd.Series,
                                    counterfactual_series: pd.Series,
                                    severity_series: pd.Series,
                                    strategy_date: pd.Timestamp,
                                    window: int = 30):
    strategy_date = pd.to_datetime(strategy_date)
    window_end = strategy_date + pd.Timedelta(days=window - 1)
    pre_sev = severity_series.loc[strategy_date - pd.Timedelta(days=window):strategy_date - pd.Timedelta(days=1)].sum()
    post_sev = severity_series.loc[strategy_date:window_end].sum()
    actual_post = actual_series.loc[strategy_date:window_end]
    counter_post = counterfactual_series.loc[strategy_date:window_end].reindex(actual_post.index)
    window_len = len(actual_post)
    if window_len == 0:
        return False, False, (0.0, 0.0), '三级'
    effective_days = (actual_post < counter_post).sum()
    count_effective = effective_days >= (window_len / 2)
    severity_effective = post_sev < pre_sev
    cf_sum = counter_post.sum()
    F1 = (cf_sum - actual_post.sum()) / cf_sum if cf_sum > 0 else 0.0
    F2 = (pre_sev - post_sev) / pre_sev if pre_sev > 0 else 0.0
    if F1 > 0.5 and F2 > 0.5:
        safety_state = '一级'
    elif F1 > 0.3:
        safety_state = '二级'
    else:
        safety_state = '三级'
    return count_effective, severity_effective, (F1, F2), safety_state


@st.cache_data(show_spinner=False)
def generate_output_and_recommendations(combined_data: pd.DataFrame,
                                        strategy_types: list,
                                        region: str = '全市',
                                        horizon: int = 30):
    results = {}
    combined_data = combined_data.copy().asfreq('D')
    combined_data[['accident_count','severity']] = combined_data[['accident_count','severity']].fillna(0)
    combined_data['strategy_type'] = combined_data['strategy_type'].apply(lambda x: x if isinstance(x, list) else [])

    acc_full = combined_data['accident_count']
    sev_full = combined_data['severity']

    max_fit_days = max(horizon + 60, MAX_PRE_DAYS)

    for strategy in strategy_types:
        has_strategy = combined_data['strategy_type'].apply(lambda x: strategy in x)
        if not has_strategy.any():
            continue
        candidate_dates = has_strategy[has_strategy].index
        intervention_date = None
        fit_start_dt = None
        for dt in candidate_dates:
            fit_start_dt = max(acc_full.index.min(), dt - pd.Timedelta(days=max_fit_days))
            pre_hist = acc_full.loc[fit_start_dt:dt - pd.Timedelta(days=1)]
            if len(pre_hist) >= MIN_PRE_DAYS:
                intervention_date = dt
                break
        if intervention_date is None:
            intervention_date = candidate_dates[0]
            fit_start_dt = max(acc_full.index.min(), intervention_date - pd.Timedelta(days=max_fit_days))

        acc = acc_full.loc[fit_start_dt:]
        sev = sev_full.loc[fit_start_dt:]
        horizon_eff = max(7, min(horizon, len(acc.loc[intervention_date:]) ))

        glm_pred, svr_pred, residuals = fit_and_extrapolate(acc, intervention_date, days=horizon_eff)

        counter = None
        if svr_pred is not None:
            counter = svr_pred
        elif glm_pred is not None:
            counter = glm_pred
        else:
            try:
                arima_df = arima_forecast_with_grid_search(acc.loc[:intervention_date],
                                                           start_date=intervention_date + pd.Timedelta(days=1),
                                                           horizon=horizon_eff)
                counter = pd.Series(arima_df['forecast'].values, index=arima_df.index, name='cf_arima')
                residuals = (acc.reindex(counter.index) - counter)
            except Exception:
                counter = None
        if counter is None:
            continue

        count_eff, sev_eff, (F1, F2), state = evaluate_strategy_effectiveness(
            actual_series=acc,
            counterfactual_series=counter,
            severity_series=sev,
            strategy_date=intervention_date,
            window=horizon_eff
        )
        results[strategy] = {
            'effect_strength': float(residuals.dropna().mean()) if residuals is not None else 0.0,
            'adaptability': float(F1 + F2),
            'count_effective': bool(count_eff),
            'severity_effective': bool(sev_eff),
            'safety_state': state,
            'F1': float(F1),
            'F2': float(F2),
            'intervention_date': str(intervention_date.date())
        }

    # Secondary attempt with 14-day window if no results
    if not results:
        for strategy in strategy_types:
            has_strategy = combined_data['strategy_type'].apply(lambda x: strategy in x)
            if not has_strategy.any():
                continue
            intervention_date = has_strategy[has_strategy].index[0]
            glm_pred, svr_pred, residuals = fit_and_extrapolate(acc_full, intervention_date, days=14)
            counter = None
            if svr_pred is not None:
                counter = svr_pred
            elif glm_pred is not None:
                counter = glm_pred
            else:
                try:
                    arima_df = arima_forecast_with_grid_search(acc_full.loc[:intervention_date],
                                                               start_date=intervention_date + pd.Timedelta(days=1),
                                                               horizon=14)
                    counter = pd.Series(arima_df['forecast'].values, index=arima_df.index, name='cf_arima')
                    residuals = (acc_full.reindex(counter.index) - counter)
                except Exception:
                    counter = None
            if counter is None:
                continue
            count_eff, sev_eff, (F1, F2), state = evaluate_strategy_effectiveness(
                actual_series=acc_full,
                counterfactual_series=counter,
                severity_series=sev_full,
                strategy_date=intervention_date,
                window=14
            )
            results[strategy] = {
                'effect_strength': float(residuals.dropna().mean()) if residuals is not None else 0.0,
                'adaptability': float(F1 + F2),
                'count_effective': bool(count_eff),
                'severity_effective': bool(sev_eff),
                'safety_state': state,
                'F1': float(F1),
                'F2': float(F2),
                'intervention_date': str(intervention_date.date())
            }

    best_strategy = max(results, key=lambda x: results[x]['adaptability']) if results else None
    recommendation = f"建议在{region}区域长期实施策略类型 {best_strategy}" if best_strategy else "无足够数据推荐策略"
    return results, recommendation

