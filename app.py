
import os
from datetime import datetime, timedelta
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --- Optional deps (graceful fallback)
try:
    from scipy.stats import ttest_ind, mannwhitneyu
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# Add import for OpenAI API
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


# =======================
# 1. Data Integration
# =======================
@st.cache_data(show_spinner=False)
def load_and_clean_data(accident_file, strategy_file):
    accident_df = pd.read_excel(accident_file, sheet_name=None)
    accident_data = pd.concat(accident_df.values(), ignore_index=True)

    accident_data['事故时间'] = pd.to_datetime(accident_data['事故时间'])
    accident_data = accident_data.dropna(subset=['事故时间', '所在街道', '事故类型'])

    strategy_df = pd.read_excel(strategy_file)
    strategy_df['发布时间'] = pd.to_datetime(strategy_df['发布时间'])
    strategy_df = strategy_df.dropna(subset=['发布时间', '交通策略类型'])

    severity_map = {'财损': 1, '伤人': 2, '亡人': 4}
    accident_data['severity'] = accident_data['事故类型'].map(severity_map).fillna(1)

    accident_data = accident_data[['事故时间', '所在街道', '事故类型', 'severity']] \
        .rename(columns={'事故时间': 'date_time', '所在街道': 'region', '事故类型': 'category'})
    strategy_df = strategy_df[['发布时间', '交通策略类型']] \
        .rename(columns={'发布时间': 'date_time', '交通策略类型': 'strategy_type'})

    return accident_data, strategy_df


@st.cache_data(show_spinner=False)
def aggregate_daily_data(accident_data: pd.DataFrame, strategy_data: pd.DataFrame) -> pd.DataFrame:
    # City-level aggregation
    accident_data = accident_data.copy()
    strategy_data = strategy_data.copy()

    accident_data['date'] = accident_data['date_time'].dt.date
    daily_accidents = accident_data.groupby('date').agg(
        accident_count=('date_time', 'count'),
        severity=('severity', 'sum')
    )
    daily_accidents.index = pd.to_datetime(daily_accidents.index)

    strategy_data['date'] = strategy_data['date_time'].dt.date
    daily_strategies = strategy_data.groupby('date')['strategy_type'].apply(list)
    daily_strategies.index = pd.to_datetime(daily_strategies.index)

    combined = daily_accidents.join(daily_strategies, how='left')
    combined['strategy_type'] = combined['strategy_type'].apply(lambda x: x if isinstance(x, list) else [])
    combined = combined.asfreq('D')
    combined[['accident_count', 'severity']] = combined[['accident_count', 'severity']].fillna(0)
    combined['strategy_type'] = combined['strategy_type'].apply(lambda x: x if isinstance(x, list) else [])
    return combined


@st.cache_data(show_spinner=False)
def aggregate_daily_data_by_region(accident_data: pd.DataFrame, strategy_data: pd.DataFrame) -> pd.DataFrame:
    """区域维度聚合。策略按天广播到所有区域（若策略本身无区域字段）。"""
    df = accident_data.copy()
    df['date'] = df['date_time'].dt.date
    g = df.groupby(['region', 'date']).agg(
        accident_count=('date_time', 'count'),
        severity=('severity', 'sum')
    )
    g.index = g.index.set_levels([g.index.levels[0], pd.to_datetime(g.index.levels[1])])
    g = g.sort_index()

    # 策略（每日列表）
    s = strategy_data.copy()
    s['date'] = s['date_time'].dt.date
    daily_strategies = s.groupby('date')['strategy_type'].apply(list)
    daily_strategies.index = pd.to_datetime(daily_strategies.index)

    # 广播
    regions = g.index.get_level_values(0).unique()
    dates = pd.date_range(g.index.get_level_values(1).min(), g.index.get_level_values(1).max(), freq='D')
    full_index = pd.MultiIndex.from_product([regions, dates], names=['region', 'date'])
    g = g.reindex(full_index).fillna(0)

    strat_map = daily_strategies.to_dict()
    g = g.assign(strategy_type=[strat_map.get(d, []) for d in g.index.get_level_values('date')])
    return g


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
import warnings

def evaluate_arima_model(series, arima_order):
    """Fit ARIMA model and return AIC for evaluation."""
    try:
        model = ARIMA(series, order=arima_order)
        model_fit = model.fit()
        return model_fit.aic
    except Exception:
        return float("inf")

def arima_forecast_with_grid_search(accident_series: pd.Series, start_date: pd.Timestamp,
                                    horizon: int = 30, p_values: list = range(0, 6),
                                    d_values: list = range(0, 2), q_values: list = range(0, 6)) -> pd.DataFrame:
    # Pre-process series
    series = accident_series.asfreq('D').fillna(0)
    start_date = pd.to_datetime(start_date)
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=ValueWarning)

    # Define the hyperparameters to search through
    best_score, best_cfg = float("inf"), None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    aic = evaluate_arima_model(series, order)
                    if aic < best_score:
                        best_score, best_cfg = aic, order
                except Exception as e:
                    continue
    
    # Fit the model with the best found order
    print(best_cfg)
    model = ARIMA(series, order=best_cfg)
    fit = model.fit()
    
    # Forecasting
    forecast_index = pd.date_range(start=start_date, periods=horizon, freq='D')
    res = fit.get_forecast(steps=horizon)
    df = res.summary_frame()
    df.index = forecast_index
    df.index.name = 'date'
    df.rename(columns={'mean': 'forecast'}, inplace=True)
    
    return df

# Example usage:
# dataframe = your_data_frame_here
# forecast_df = arima_forecast_with_grid_search(dataframe['accident_count'], start_date=pd.Timestamp('YYYY-MM-DD'), horizon=30)


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


def detect_anomalies(series: pd.Series, contamination: float = 0.1):
    series = series.asfreq('D').fillna(0)
    iso = IsolationForest(contamination=contamination, random_state=42)
    yhat = iso.fit_predict(series.values.reshape(-1, 1))
    anomaly_mask = (yhat == -1)
    anomaly_indices = series.index[anomaly_mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name='Accident Count'))
    fig.add_trace(go.Scatter(x=anomaly_indices, y=series.loc[anomaly_indices], mode='markers',
                             marker=dict(color='red', size=10), name='Anomalies'))
    fig.update_layout(title="Anomaly Detection in Accident Count",
                      xaxis_title="Date", yaxis_title="Count")
    return anomaly_indices, fig




def intervention_model(series: pd.Series,
                       intervention_date: pd.Timestamp,
                       intervention_type: str = 'persistent',
                       effect_type: str = 'sudden',
                       omega: float = 0.5,
                       decay: float = 10.0,
                       lag: int = 0):
    series = series.asfreq('D').fillna(0)
    intervention_date = pd.to_datetime(intervention_date)
    Z_t = pd.Series(0.0, index=series.index)
    if intervention_type == 'persistent':
        Z_t.loc[intervention_date:] = 1.0
    else:
        post_len = len(Z_t.loc[intervention_date:])
        Z_t.loc[intervention_date:] = np.exp(-np.arange(post_len) / decay)
    if effect_type == 'gradual':
        Z_t = Z_t * np.linspace(0, 1, len(Z_t))
    Z_t = Z_t.shift(lag).fillna(0)
    Y_t = series + omega * Z_t
    return Y_t, Z_t


def fit_and_extrapolate(series: pd.Series,
                        intervention_date: pd.Timestamp,
                        days: int = 30):

    series = series.asfreq('D').fillna(0)
    # 统一为无时区、按天的时间戳
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    intervention_date = pd.to_datetime(intervention_date).tz_localize(None).normalize()

    pre = series.loc[:intervention_date - pd.Timedelta(days=1)]
    if len(pre) < 5:
        return None, None, None

    x_pre = np.arange(len(pre))
    x_future = np.arange(len(pre), len(pre) + days)

    # 1️⃣ GLM：加入二次项
    X_pre_glm = sm.add_constant(np.column_stack([x_pre, x_pre**2]))
    glm = sm.GLM(pre.values, X_pre_glm, family=sm.families.Poisson())
    glm_res = glm.fit()
    X_future_glm = sm.add_constant(np.column_stack([x_future, x_future**2]))
    glm_pred = glm_res.predict(X_future_glm)

    # SVR
    # 2️⃣ SVR：加标准化 & 调参 / 改线性核
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf', C=10, gamma=0.1)   # 或 kernel='linear'
    )
    svr.fit(x_pre.reshape(-1, 1), pre.values)
    svr_pred = svr.predict(x_future.reshape(-1, 1))

    # 目标预测索引（未来可能超出历史范围 —— 用 reindex，不要 .loc[...]）
    post_index = pd.date_range(intervention_date, periods=days, freq='D')

    glm_pred = pd.Series(glm_pred, index=post_index, name='glm_pred')
    svr_pred = pd.Series(svr_pred, index=post_index, name='svr_pred')

    # ✅ 关键修复：对不存在的日期补 NaN，而不是 .loc[post_index]
    post = series.reindex(post_index)

    residuals = pd.Series(post.values - svr_pred[:len(post)],
                          index=post_index, name='residual')

    return glm_pred, svr_pred, residuals


def evaluate_strategy_effectiveness(actual_series: pd.Series,
                                    counterfactual_series: pd.Series,
                                    severity_series: pd.Series,
                                    strategy_date: pd.Timestamp,
                                    window: int = 30):
    strategy_date = pd.to_datetime(strategy_date)
    pre_sev = severity_series.loc[strategy_date - pd.Timedelta(days=window):strategy_date - pd.Timedelta(days=1)].sum()
    post_sev = severity_series.loc[strategy_date:strategy_date + pd.Timedelta(days=window - 1)].sum()
    actual_post = actual_series.loc[strategy_date:strategy_date + pd.Timedelta(days=window - 1)]
    counter_post = counterfactual_series.loc[strategy_date:strategy_date + pd.Timedelta(days=window - 1)]
    counter_post = counter_post.reindex(actual_post.index)
    effective_days = (actual_post < counter_post).sum()
    count_effective = effective_days >= (window / 2)
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


def generate_output_and_recommendations(combined_data: pd.DataFrame,
                                        strategy_types: list,
                                        region: str = '全市',
                                        horizon: int = 30):
    results = {}
    accident_series = combined_data['accident_count']
    severity_series = combined_data['severity']
    for strategy in strategy_types:
        has_strategy = combined_data['strategy_type'].apply(lambda x: strategy in x)
        if not has_strategy.any():
            continue
        intervention_date = has_strategy[has_strategy].index[0]
        glm_pred, svr_pred, residuals = fit_and_extrapolate(accident_series, intervention_date, days=horizon)
        if svr_pred is None:
            continue
        count_eff, sev_eff, (F1, F2), state = evaluate_strategy_effectiveness(
            actual_series=accident_series,
            counterfactual_series=svr_pred,
            severity_series=severity_series,
            strategy_date=intervention_date,
            window=horizon
        )
        results[strategy] = {
            'effect_strength': float(residuals.mean()),
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
    pd.DataFrame(results).T.to_csv('strategy_evaluation_results.csv', encoding='utf-8-sig')
    with open('recommendation.txt', 'w', encoding='utf-8') as f:
        f.write(recommendation)
    return results, recommendation


# =======================
# 3. UI Helpers
# =======================
def hash_like(obj: str) -> str:
    return hashlib.md5(obj.encode('utf-8')).hexdigest()[:8]


def compute_kpis(df_city: pd.DataFrame, arima_df: pd.DataFrame | None,
                 today: pd.Timestamp, window:int=30):
    # 今日/昨日
    today_date = pd.to_datetime(today.date())
    yesterday = today_date - pd.Timedelta(days=1)
    this_week_start = today_date - pd.Timedelta(days=today_date.weekday())  # 周一
    last_week_start = this_week_start - pd.Timedelta(days=7)
    this_week_end = today_date

    today_cnt = int(df_city['accident_count'].get(today_date, 0))
    yest_cnt = int(df_city['accident_count'].get(yesterday, 0))
    wow = (today_cnt - yest_cnt) / yest_cnt if yest_cnt > 0 else 0.0

    this_week = df_city.loc[this_week_start:this_week_end]['accident_count'].sum()
    last_week = df_city.loc[last_week_start:last_week_start + pd.Timedelta(days=(this_week_end - this_week_start).days)]['accident_count'].sum()
    yoy = (this_week - last_week) / last_week if last_week > 0 else 0.0

    # 预测偏差（近7天）
    forecast_bias = None
    if arima_df is not None:
        recent = df_city.index.max() - pd.Timedelta(days=6)
        actual = df_city.loc[recent:df_city.index.max(), 'accident_count']
        fcst = arima_df['forecast'].reindex(actual.index).fillna(method='ffill')
        denom = fcst.replace(0, np.nan)
        bias = (np.abs(actual - fcst) / denom).dropna()
        forecast_bias = float(bias.mean()) if len(bias) else None

    # 策略覆盖（近30天）
    last_window = df_city.index.max() - pd.Timedelta(days=window-1)
    strat_days = df_city.loc[last_window:, 'strategy_type'].apply(lambda x: len(x) > 0).sum()
    coverage = strat_days / window

    # 上线策略数（去重）
    active_strats = set(s for lst in df_city.loc[last_window:, 'strategy_type'] for s in lst)
    active_count = len(active_strats)

    # 近30天安全等级（用 generate_output_and_recommendations 里 best 的等级）
    # 这里只取最近出现过的策略做评估
    strategies = sorted(active_strats)
    safety_state = '—'
    if strategies:
        res, _ = generate_output_and_recommendations(df_city.loc[last_window:], strategies, region='全市', horizon=min(30, len(df_city.loc[last_window:])))
        if res:
            # 取适配度最高的策略的安全等级
            best = max(res, key=lambda k: res[k]['adaptability'])
            safety_state = res[best]['safety_state']

    return {
        'today_cnt': today_cnt,
        'wow': wow,
        'this_week': int(this_week),
        'yoy': yoy,
        'forecast_bias': forecast_bias,
        'active_count': active_count,
        'coverage': coverage,
        'safety_state': safety_state
    }


def significance_test(pre: pd.Series, post: pd.Series):
    pre = pre.dropna(); post = post.dropna()
    if len(pre) < 3 or len(post) < 3:
        return None, None
    if HAS_SCIPY:
        try:
            stat, p = ttest_ind(pre, post, equal_var=False)
        except Exception:
            stat, p = mannwhitneyu(pre, post, alternative='two-sided')
        return float(stat), float(p)
    return None, None


def save_fig_as_html(fig, filename):
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    return filename

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# 依赖：已在脚本前面定义的  knn_forecast_counterfactual()  和  fit_and_extrapolate()
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# 依赖：knn_forecast_counterfactual、fit_and_extrapolate 已存在

def evaluate_models(series: pd.Series,
                    horizon: int = 30,
                    lookback: int = 14,
                    p_values: range = range(0, 6),
                    d_values: range = range(0, 2),
                    q_values: range = range(0, 6)) -> pd.DataFrame:
    """
    留出法（最后 horizon 天作为验证集）比较 ARIMA / KNN / GLM / SVR，
    输出 MAE・RMSE・MAPE，并按 RMSE 升序排序。
    """
    # 统一日频 & 缺失补零
    series = series.asfreq('D').fillna(0)
    if len(series) <= horizon + 10:
        raise ValueError("序列太短，无法留出 %d 天进行评估。" % horizon)

    train, test = series.iloc[:-horizon], series.iloc[-horizon:]

    def _to_series_like(pred, a_index):
        # 将任意预测对齐成与 actual 同索引的 Series
        if isinstance(pred, pd.Series):
            return pred.reindex(a_index)
        return pd.Series(pred, index=a_index)

    def _metrics(a: pd.Series, p) -> dict:
        p = _to_series_like(p, a.index).astype(float)
        a = a.astype(float)

        mae = mean_absolute_error(a, p)

        # 兼容旧版 sklearn：没有 squared 参数时手动开方
        try:
            rmse = mean_squared_error(a, p, squared=False)
        except TypeError:
            rmse = mean_squared_error(a, p) ** 0.5

        # 忽略分母为 0 的样本
        mape = np.nanmean(np.abs((a - p) / np.where(a == 0, np.nan, a))) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    results = {}

    # ---------- ARIMA ----------
    best_aic, best_order = float('inf'), None
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

    # ---------- KNN ----------
    try:
        knn_pred, _ = knn_forecast_counterfactual(series,
                                                  train.index[-1] + pd.Timedelta(days=1),
                                                  lookback=lookback,
                                                  horizon=horizon)
        if knn_pred is not None:
            results['KNN'] = _metrics(test, knn_pred)
    except Exception:
        pass

    # ---------- GLM & SVR ----------
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


# =======================
# 4. App
# =======================
def run_streamlit_app():
    st.set_page_config(page_title="Traffic Safety Analysis", layout="wide")
    st.title("🚦 Traffic Safety Intervention Analysis System")

    # Sidebar — Upload & Global Filters & Auto Refresh
    st.sidebar.header("数据与筛选")
    
    # Create a form for data inputs to batch updates
    with st.sidebar.form(key="data_input_form"):
        accident_file = st.file_uploader("上传事故数据 (Excel)", type=['xlsx'])
        strategy_file = st.file_uploader("上传交通策略数据 (Excel)", type=['xlsx'])

        # Global filters
        st.markdown("---")
        st.subheader("全局筛选器")
        # Placeholder for region selection (will be populated after data is loaded)
        region_sel = st.selectbox("区域", options=["全市"], index=0, key="region_select")
        # Default date range (will be updated after data is loaded)
        min_date = pd.to_datetime('2022-01-01').date()
        max_date = pd.to_datetime('2022-12-31').date()
        date_range = st.date_input("时间范围", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        strat_filter = st.multiselect("策略类型（过滤）", options=[], help="为空表示不过滤策略；选择后仅保留当天包含所选策略的日期")
        
        # Apply button for data loading and filtering
        apply_button = st.form_submit_button("应用数据与筛选")

    # Auto-refresh controls (outside the form, as it’s independent)
    st.sidebar.markdown("---")
    st.sidebar.subheader("实时刷新")
    auto = st.sidebar.checkbox("自动刷新", value=False, help="启用后将按间隔自动刷新页面")
    interval = st.sidebar.number_input("刷新间隔（秒）", min_value=5, max_value=600, value=30, step=5)
    if auto and HAS_AUTOREFRESH:
        st_autorefresh(interval=int(interval*1000), key="autorefresh")
    elif auto and not HAS_AUTOREFRESH:
        st.sidebar.info("未安装 `streamlit-autorefresh`，请使用上方“重新运行”按钮或关闭再开启此开关。")

    # Add OpenAI API key input in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("GPT API 配置")
    openai_api_key = st.sidebar.text_input("GPT API Key", value='sk-dQhKOOG48iVEfgJfAb14458dA4474fB09aBbE8153d4aB3Fc', type="password", help="用于GPT分析结果的API密钥")
    open_ai_base_url = st.sidebar.text_input("GPT Base Url", value='https://az.gptplus5.com/v1', type='default')

    # Initialize session state to store processed data
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = {
            'combined_city': None,
            'combined_by_region': None,
            'accident_data': None,
            'strategy_data': None,
            'all_regions': ["全市"],
            'all_strategy_types': [],
            'min_date': min_date,
            'max_date': max_date,
            'region_sel': "全市",
            'date_range': (min_date, max_date),
            'strat_filter': []
        }

    # Process data only when Apply button is clicked
    if apply_button and accident_file and strategy_file:
        with st.spinner("数据载入中…"):
            # Load and clean data
            accident_data, strategy_data = load_and_clean_data(accident_file, strategy_file)
            combined_city = aggregate_daily_data(accident_data, strategy_data)
            combined_by_region = aggregate_daily_data_by_region(accident_data, strategy_data)

            # Update available options for filters
            all_regions = ["全市"] + sorted(accident_data['region'].unique().tolist())
            all_strategy_types = sorted({s for lst in combined_city['strategy_type'] for s in lst})
            min_date = combined_city.index.min().date()
            max_date = combined_city.index.max().date()

            # Store processed data in session state
            st.session_state['processed_data'].update({
                'combined_city': combined_city,
                'combined_by_region': combined_by_region,
                'accident_data': accident_data,
                'strategy_data': strategy_data,
                'all_regions': all_regions,
                'all_strategy_types': all_strategy_types,
                'min_date': min_date,
                'max_date': max_date,
                'region_sel': region_sel,
                'date_range': date_range,
                'strat_filter': strat_filter
            })

    # Retrieve data from session state
    combined_city = st.session_state['processed_data']['combined_city']
    combined_by_region = st.session_state['processed_data']['combined_by_region']
    accident_data = st.session_state['processed_data']['accident_data']
    strategy_data = st.session_state['processed_data']['strategy_data']
    all_regions = st.session_state['processed_data']['all_regions']
    all_strategy_types = st.session_state['processed_data']['all_strategy_types']
    min_date = st.session_state['processed_data']['min_date']
    max_date = st.session_state['processed_data']['max_date']
    region_sel = st.session_state['processed_data']['region_sel']
    date_range = st.session_state['processed_data']['date_range']
    strat_filter = st.session_state['processed_data']['strat_filter']

    # Update selectbox and multiselect options dynamically (outside the form for display)
    st.sidebar.markdown("---")
    st.sidebar.subheader("当前筛选状态")
    st.sidebar.write(f"区域: {region_sel}")
    st.sidebar.write(f"时间范围: {date_range[0]} 至 {date_range[1]}")
    st.sidebar.write(f"策略类型: {', '.join(strat_filter) or '无'}")

    # Proceed only if data is available
    if combined_city is not None and combined_by_region is not None:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
        if region_sel == "全市":
            base = combined_city.loc[start_dt:end_dt].copy()
        else:
            block = combined_by_region.xs(region_sel, level='region').copy()
            base = block.loc[start_dt:end_dt]
        if strat_filter:
            mask = base['strategy_type'].apply(lambda x: any(s in x for s in strat_filter))
            base = base[mask]

        # Last refresh info
        if 'last_refresh' not in st.session_state:
            st.session_state['last_refresh'] = datetime.now()
        last_refresh = st.session_state['last_refresh']

        # Compute ARIMA for KPI bias
        arima_df = None
        try:
            arima_df = arima_forecast_with_grid_search(
                base['accident_count'], base.index.max() + pd.Timedelta(days=1), horizon=7
            )
        except Exception:
            pass

        # KPI Overview
        kpi = compute_kpis(base, arima_df, today=pd.Timestamp('2022-12-01'))
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("今日事故数", f"{kpi['today_cnt']}", f"{kpi['wow']*100:.1f}% 环比")
        c2.metric("本周事故数", f"{kpi['this_week']}", f"{kpi['yoy']*100:.1f}% 同比")
        c3.metric("近7天预测偏差", ("{:.1f}%".format(kpi['forecast_bias']*100) if kpi['forecast_bias'] is not None else "—"))
        c4.metric("近30天策略数", f"{kpi['active_count']}")
        c5.metric("近30天策略覆盖率", f"{kpi['coverage']*100:.1f}%")
        c6.metric("近30天安全等级", kpi['safety_state'])

        # Top-right meta
        meta_col1, meta_col2 = st.columns([4, 1])
        with meta_col2:
            st.caption(f"🕒 最近刷新：{last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

        # Tabs (add new tab for GPT analysis)
        tab_dash, tab_pred, tab_eval, tab_anom, tab_strat, tab_comp, tab_sim, tab_gpt = st.tabs(
            ["🏠 总览", "📈 预测模型", "📊 模型评估", "⚠️ 异常检测", "📝 策略评估", "⚖️ 策略对比", "🧪 情景模拟", "🔍 GPT 分析"]
        )

        # --- Tab 1: 总览页
        with tab_dash:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=base.index, y=base['accident_count'], name='事故数', mode='lines'))
            fig_line.update_layout(title="事故数（过滤后）", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig_line, use_container_width=True)
            fname = save_fig_as_html(fig_line, "overview_series.html")
            st.download_button("下载图表 HTML", data=open(fname, 'rb').read(),
                               file_name="overview_series.html", mime="text/html")

            st.dataframe(base, use_container_width=True)
            csv_bytes = base.to_csv(index=True).encode('utf-8-sig')
            st.download_button("下载当前视图 CSV", data=csv_bytes, file_name="filtered_view.csv", mime="text/csv")

            meta = {
                "region": region_sel,
                "date_range": [str(start_dt.date()), str(end_dt.date())],
                "strategy_filter": strat_filter,
                "rows": int(len(base)),
                "min_date": str(base.index.min().date()) if len(base) else None,
                "max_date": str(base.index.max().date()) if len(base) else None
            }
            with open("run_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            st.download_button("下载运行参数 JSON", data=open("run_metadata.json", "rb").read(),
                               file_name="run_metadata.json", mime="application/json")

        # --- Tab 2: 预测模型
        with tab_pred:
            st.subheader("多模型预测比较")
            # 使用表单封装交互组件
            with st.form(key="predict_form"):
                default_date = base.index.max() - pd.Timedelta(days=60) if len(base) else pd.Timestamp('2022-01-01')
                selected_date = st.date_input("选择干预日期 / 预测起点", value=default_date)
                horizon = st.number_input("预测天数", min_value=7, max_value=90, value=30, step=1)
                submit_predict = st.form_submit_button("应用预测参数")

            if submit_predict and len(base.loc[:pd.to_datetime(selected_date)]) >= 10:
                first_date = pd.to_datetime(selected_date)
                try:
                    train_series = base['accident_count'].loc[:first_date]
                    arima30 = arima_forecast_with_grid_search(
                        train_series,
                        start_date=first_date + pd.Timedelta(days=1),
                        horizon=horizon
                    )
                except Exception as e:
                    st.warning(f"ARIMA 运行失败：{e}")
                    arima30 = None

                knn_pred, _ = knn_forecast_counterfactual(base['accident_count'],
                                                        first_date,
                                                        horizon=horizon)
                glm_pred, svr_pred, residuals = fit_and_extrapolate(base['accident_count'],
                                                                    first_date,
                                                                    days=horizon)

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=base.index, y=base['accident_count'],
                                            name="实际", mode="lines"))
                if arima30 is not None:
                    fig_pred.add_trace(go.Scatter(x=arima30.index, y=arima30['forecast'],
                                                name="ARIMA", mode="lines"))
                if knn_pred is not None:
                    fig_pred.add_trace(go.Scatter(x=knn_pred.index, y=knn_pred,
                                                name="KNN", mode="lines"))
                if glm_pred is not None:
                    fig_pred.add_trace(go.Scatter(x=glm_pred.index, y=glm_pred,
                                                name="GLM", mode="lines"))
                if svr_pred is not None:
                    fig_pred.add_trace(go.Scatter(x=svr_pred.index, y=svr_pred,
                                                name="SVR", mode="lines"))

                fig_pred.update_layout(
                    title=f"多模型预测比较（起点：{first_date.date()}，预测 {horizon} 天）",
                    xaxis_title="日期", yaxis_title="事故数"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                col_dl1, col_dl2 = st.columns(2)
                if arima30 is not None:
                    col_dl1.download_button("下载 ARIMA 预测 CSV",
                                        data=arima30.to_csv().encode("utf-8-sig"),
                                        file_name="arima_forecast.csv",
                                        mime="text/csv")
            elif submit_predict:
                st.info("⚠️ 干预前数据较少，可能影响拟合质量。")
            else:
                st.info("请设置预测参数并点击“应用预测参数”按钮。")

        # --- Tab 3: 模型评估
        with tab_eval:
            st.subheader("模型预测效果对比")
            with st.form(key="model_eval_form"):
                horizon_sel = st.slider("评估窗口（天）", 7, 60, 30, step=1)
                submit_eval = st.form_submit_button("应用评估参数")

            if submit_eval:
                try:
                    df_metrics = evaluate_models(base['accident_count'], horizon=horizon_sel)
                    st.dataframe(df_metrics, use_container_width=True)
                    best_model = df_metrics['RMSE'].idxmin()
                    st.success(f"过去 {horizon_sel} 天中，RMSE 最低的模型是：**{best_model}**")
                    st.download_button(
                        "下载评估结果 CSV",
                        data=df_metrics.to_csv().encode('utf-8-sig'),
                        file_name="model_evaluation.csv",
                        mime="text/csv"
                    )
                except ValueError as err:
                    st.warning(str(err))
            else:
                st.info("请设置评估窗口并点击“应用评估参数”按钮。")

        # --- Tab 4: 异常检测
        with tab_anom:
            anomalies, anomaly_fig = detect_anomalies(base['accident_count'])
            st.plotly_chart(anomaly_fig, use_container_width=True)
            st.write(f"检测到异常点：{len(anomalies)} 个")
            st.download_button("下载异常日期 CSV",
                            data=anomalies.to_series().to_csv(index=False).encode('utf-8-sig'),
                            file_name="anomalies.csv", mime="text/csv")

        # --- Tab 5: 策略评估
        with tab_strat:
            st.info(f"📌 检测到的策略类型：{', '.join(all_strategy_types) or '（数据中没有策略）'}")
            if all_strategy_types:
                results, recommendation = generate_output_and_recommendations(base, all_strategy_types,
                                                                              region=region_sel if region_sel!='全市' else '全市')
                st.subheader("各策略指标")
                df_res = pd.DataFrame(results).T
                st.dataframe(df_res, use_container_width=True)
                st.success(f"⭐ 推荐：{recommendation}")
                st.download_button("下载策略评估 CSV",
                                   data=df_res.to_csv().encode('utf-8-sig'),
                                   file_name="strategy_evaluation_results.csv", mime="text/csv")
                with open('recommendation.txt','r',encoding='utf-8') as f:
                    st.download_button("下载推荐文本", data=f.read().encode('utf-8'), file_name="recommendation.txt")
            else:
                st.warning("数据中没有检测到策略。")

        # --- Tab 6: 策略对比
        with tab_comp:
            def strategy_metrics(strategy):
                mask = base['strategy_type'].apply(lambda x: strategy in x)
                if not mask.any():
                    return None
                dt = mask[mask].index[0]
                glm_pred, svr_pred, residuals = fit_and_extrapolate(base['accident_count'], dt, days=30)
                if svr_pred is None:
                    return None
                actual_post = base['accident_count'].loc[dt:dt+pd.Timedelta(days=29)]
                pre = base['accident_count'].loc[dt-pd.Timedelta(days=30):dt-pd.Timedelta(days=1)]
                stat, p = significance_test(pre, actual_post)
                count_eff, sev_eff, (F1, F2), state = evaluate_strategy_effectiveness(
                    actual_series=base['accident_count'],
                    counterfactual_series=svr_pred,
                    severity_series=base['severity'],
                    strategy_date=dt, window=30
                )
                return {
                    "干预日": str(dt.date()),
                    "前30天事故": int(pre.sum()),
                    "后30天事故": int(actual_post.sum()),
                    "每日均值(前/后)": (float(pre.mean()), float(actual_post.mean())),
                    "t统计/p值": (stat, p),
                    "F1/F2": (float(F1), float(F2)),
                    "有效天数过半?": bool(count_eff),
                    "严重度下降?": bool(sev_eff),
                    "安全等级": state
                }
            if all_strategy_types:
                st.subheader("策略对比")
                with st.form(key="strategy_compare_form"):
                    colA, colB = st.columns(2)
                    with colA:
                        sA = st.selectbox("策略 A", options=all_strategy_types, key="stratA")
                    with colB:
                        sB = st.selectbox("策略 B", options=[s for s in all_strategy_types if s != st.session_state.get("stratA")], key="stratB")
                    submit_compare = st.form_submit_button("应用策略对比")

                if submit_compare:
                    mA = strategy_metrics(sA)
                    mB = strategy_metrics(sB)
                    if mA and mB:
                        show = pd.DataFrame({
                            "指标": ["干预日", "前30天事故", "后30天事故", "每日均值(前)", "每日均值(后)", "t统计", "p值", "F1", "F2", "有效天数过半?", "严重度下降?", "安全等级"],
                            f"{sA}": [mA["干预日"], mA["前30天事故"], mA["后30天事故"],
                                    mA["每日均值(前/后)"][0], mA["每日均值(前/后)"][1],
                                    mA["t统计/p值"][0], mA["t统计/p值"][1],
                                    mA["F1/F2"][0], mA["F1/F2"][1],
                                    mA["有效天数过半?"], mA["严重度下降?"], mA["安全等级"]],
                            f"{sB}": [mB["干预日"], mB["前30天事故"], mB["后30天事故"],
                                    mB["每日均值(前/后)"][0], mB["每日均值(前/后)"][1],
                                    mB["t统计/p值"][0], mB["t统计/p值"][1],
                                    mB["F1/F2"][0], mB["F1/F2"][1],
                                    mB["有效天数过半?"], mB["严重度下降?"], mB["安全等级"]],
                        })
                        st.dataframe(show, use_container_width=True)
                        st.download_button("下载对比表 CSV",
                                        data=show.to_csv(index=False).encode('utf-8-sig'),
                                        file_name="strategy_compare.csv", mime="text/csv")
                    else:
                        st.info("所选策略可能缺少足够的干预前数据或未在当前过滤范围内出现。")
                else:
                    st.info("请选择策略并点击“应用策略对比”按钮。")
            else:
                st.warning("没有策略可供对比。")

        # --- Tab 7: 情景模拟
        with tab_sim:
            st.subheader("情景模拟")
            st.write("选择一个日期与策略，模拟“在该日期上线该策略”的影响：")
            with st.form(key="simulation_form"):
                sim_date = st.date_input("模拟策略上线日期", value=(base.index.max() - pd.Timedelta(days=14)))
                sim_strategy = st.selectbox("模拟策略类型", options=all_strategy_types or ["示例策略"])
                sim_days = st.slider("模拟天数", 7, 60, 30)
                submit_simulation = st.form_submit_button("应用模拟参数")

            if submit_simulation:
                glm_pred, svr_pred, residuals = fit_and_extrapolate(base['accident_count'], pd.to_datetime(sim_date), days=sim_days)
                if svr_pred is None:
                    st.warning("干预前数据不足，无法进行模拟。")
                else:
                    count_eff, sev_eff, (F1, F2), state = evaluate_strategy_effectiveness(
                        actual_series=base['accident_count'],
                        counterfactual_series=svr_pred,
                        severity_series=base['severity'],
                        strategy_date=pd.to_datetime(sim_date),
                        window=sim_days
                    )
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=base.index, y=base['accident_count'], name='实际', mode='lines'))
                    fig_sim.add_trace(go.Scatter(x=svr_pred.index, y=svr_pred, name='Counterfactual(SVR)', mode='lines'))
                    fig_sim.update_layout(title=f"情景模拟：{sim_strategy} 自 {sim_date} 起", xaxis_title="日期", yaxis_title="事故数")
                    st.plotly_chart(fig_sim, use_container_width=True)

                    st.success(f"模拟结果：F1={F1:.2f}, F2={F2:.2f}, 等级={state}；"
                            f"{'事故数在多数天小于counterfactual' if count_eff else '效果不明显'}；"
                            f"{'严重度下降' if sev_eff else '严重度无下降'}。")
                    st.download_button("下载模拟图 HTML",
                                    data=open(save_fig_as_html(fig_sim, "simulation.html"), "rb").read(),
                                    file_name="simulation.html", mime="text/html")
            else:
                st.info("请设置模拟参数并点击“应用模拟参数”按钮。")

        # --- New Tab 8: GPT 分析
        with tab_gpt:
            from openai import OpenAI
            st.subheader("GPT 数据分析与改进建议")
            # open_ai_key = f"sk-dQhKOOG48iVEfgJfAb14458dA4474fB09aBbE8153d4aB3Fc"
            if not HAS_OPENAI:
                st.warning("未安装 `openai` 库。请安装后重试。")
            elif not openai_api_key:
                st.info("请在左侧边栏输入 OpenAI API Key 以启用 GPT 分析。")
            else:
                if all_strategy_types:
                    # Generate results if not already
                    results, recommendation = generate_output_and_recommendations(base, all_strategy_types,
                                                                                  region=region_sel if region_sel != '全市' else '全市')
                    df_res = pd.DataFrame(results).T
                    kpi_json = json.dumps(kpi, ensure_ascii=False, indent=2)
                    results_json = df_res.to_json(orient="records", force_ascii=False)
                    recommendation_text = recommendation

                    # Prepare data to send
                    data_to_analyze = {
                        "kpis": kpi_json,
                        "strategy_results": results_json,
                        "recommendation": recommendation_text
                    }
                    data_str = json.dumps(data_to_analyze, ensure_ascii=False)

                    prompt = str(f"""
                    请分析以下交通安全分析结果，包括KPI指标、策略评估结果和推荐。
                    提供数据结果的详细分析，以及改进思路和建议。
                    数据：{str(data_str)}
                    """)
                    #st.text_area(prompt)
                    if st.button("上传数据至 GPT 并获取分析"):
                        try:
                            client = OpenAI(
                                    base_url=open_ai_base_url,
                                    # sk-xxx替换为自己的key
                                    api_key=openai_api_key
                            )
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that analyzes traffic safety data."},
                                    {"role": "user", "content": prompt}
                                ],
                                stream=False
                            )
                            gpt_response = response.choices[0].message.content 
                            st.markdown("### GPT 分析结果与改进思路")
                            st.markdown(gpt_response, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"调用 OpenAI API 失败：{str(e)}")
                else:
                    st.warning("没有策略数据可供分析。")

                # Update refresh time
                st.session_state['last_refresh'] = datetime.now()

    else:
        st.info("请先在左侧上传事故数据与策略数据，并点击“应用数据与筛选”按钮。")

if __name__ == "__main__":
    run_streamlit_app()