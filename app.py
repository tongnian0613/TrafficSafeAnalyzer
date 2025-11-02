from __future__ import annotations

import os
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
from typing import Optional

from sklearn.ensemble import IsolationForest

import streamlit as st
import plotly.graph_objects as go

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


from services.io import (
    load_and_clean_data,
    aggregate_daily_data,
    aggregate_daily_data_by_region,
    load_accident_records,
)
from services.forecast import (
    arima_forecast_with_grid_search,
    knn_forecast_counterfactual,
    fit_and_extrapolate,
)
from services.strategy import (
    evaluate_strategy_effectiveness,
    generate_output_and_recommendations,
)
from services.metrics import evaluate_models

try:
    from ui_sections import (
        render_overview,
        render_forecast,
        render_model_eval,
        render_strategy_eval,
        render_hotspot,
    )
except Exception:  # pragma: no cover - fallback to inline logic
    render_overview = None
    render_forecast = None
    render_model_eval = None
    render_strategy_eval = None
    render_hotspot = None

def detect_anomalies(series: pd.Series, contamination: float = 0.1):
    series = series.asfreq('D').fillna(0)
    iso = IsolationForest(n_estimators=50, contamination=contamination, random_state=42, n_jobs=-1)
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


# =======================
# 3. UI Helpers
# =======================

def compute_kpis(df_city: pd.DataFrame, arima_df: Optional[pd.DataFrame],
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


# =======================
# 4. App
# =======================


# =======================
# 4. App
# =======================
def run_streamlit_app():
    # Must be the first Streamlit command
    st.set_page_config(page_title="Traffic Safety Analysis", layout="wide")
    st.title("🚦 Traffic Safety Intervention Analysis System")

    # Sidebar — Upload & Global Filters & Auto Refresh
    st.sidebar.header("数据与筛选")

    default_min_date = pd.to_datetime('2022-01-01').date()
    default_max_date = pd.to_datetime('2022-12-31').date()

    def clamp_date_range(requested, minimum, maximum):
        """Ensure the requested tuple stays within [minimum, maximum]."""
        if not isinstance(requested, (list, tuple)):
            requested = (requested, requested)
        start, end = requested
        if start > end:
            start, end = end, start
        if end < minimum or start > maximum:
            return minimum, maximum
        start = max(minimum, start)
        end = min(maximum, end)
        return start, end

    # Initialize session state to store processed data (before rendering controls)
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = {
            'combined_city': None,
            'combined_by_region': None,
            'accident_data': None,
            'accident_records': None,
            'strategy_data': None,
            'all_regions': ["全市"],
            'all_strategy_types': [],
            'min_date': default_min_date,
            'max_date': default_max_date,
            'region_sel': "全市",
            'date_range': (default_min_date, default_max_date),
            'strat_filter': [],
            'accident_source_name': None,
        }

    sidebar_state = st.session_state['processed_data']

    available_regions = sidebar_state['all_regions'] if sidebar_state['all_regions'] else ["全市"]
    current_region = sidebar_state['region_sel'] if sidebar_state['region_sel'] in available_regions else available_regions[0]
    available_strategies = sidebar_state['all_strategy_types']
    current_strategies = [s for s in sidebar_state['strat_filter'] if s in available_strategies]

    min_date = sidebar_state['min_date']
    max_date = sidebar_state['max_date']
    raw_start, raw_end = sidebar_state['date_range']
    start_default = max(min_date, min(raw_start, max_date))
    end_default = max(start_default, min(raw_end, max_date))
    
    # Create a form for data inputs to batch updates
    with st.sidebar.form(key="data_input_form"):
        accident_file = st.file_uploader("上传事故数据 (Excel)", type=['xlsx'])
        strategy_file = st.file_uploader("上传交通策略数据 (Excel)", type=['xlsx'])

        # Global filters
        st.markdown("---")
        st.subheader("全局筛选器")
        region_sel = st.selectbox(
            "区域",
            options=available_regions,
            index=available_regions.index(current_region),
            key="region_select",
        )
        date_range = st.date_input(
            "时间范围",
            value=(start_default, end_default),
            min_value=min_date,
            max_value=max_date,
        )
        strat_filter = st.multiselect(
            "策略类型（过滤）",
            options=available_strategies,
            default=current_strategies,
            help="为空表示不过滤策略；选择后仅保留当天包含所选策略的日期",
        )
        
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
    st.sidebar.subheader("AI API 配置")
    openai_api_key = st.sidebar.text_input("AI API Key", value='sk-sXY934yPqjh7YKKC08380b198fEb47308cDa09BeE23d9c8a', type="password", help="用于 AI 分析结果的 API 密钥")
    open_ai_base_url = st.sidebar.text_input("AI Base Url", value='https://aihubmix.com/v1', type='default')

    # Process data only when Apply button is clicked
    if apply_button and accident_file and strategy_file:
        with st.spinner("数据载入中…"):
            # Load and clean data
            accident_records = load_accident_records(accident_file, require_location=True)
            accident_data, strategy_data = load_and_clean_data(accident_file, strategy_file)
            combined_city = aggregate_daily_data(accident_data, strategy_data)
            combined_by_region = aggregate_daily_data_by_region(accident_data, strategy_data)

            # Update available options for filters
            all_regions = ["全市"] + sorted(accident_data['region'].unique().tolist())
            all_strategy_types = sorted({s for lst in combined_city['strategy_type'] for s in lst})
            min_date = combined_city.index.min().date()
            max_date = combined_city.index.max().date()

            # Store processed data in session state
            sanitized_start, sanitized_end = clamp_date_range(date_range, min_date, max_date)
            st.session_state['processed_data'].update({
                'combined_city': combined_city,
                'combined_by_region': combined_by_region,
                'accident_data': accident_data,
                'accident_records': accident_records,
                'strategy_data': strategy_data,
                'all_regions': all_regions,
                'all_strategy_types': all_strategy_types,
                'min_date': min_date,
                'max_date': max_date,
                'region_sel': region_sel,
                'date_range': (sanitized_start, sanitized_end),
                'strat_filter': strat_filter,
                'accident_source_name': getattr(accident_file, "name", "事故数据.xlsx"),
            })

    sanitized_start, sanitized_end = clamp_date_range(date_range, min_date, max_date)

    # Persist the latest sidebar selections for display and downstream filtering
    st.session_state['processed_data']['region_sel'] = region_sel
    st.session_state['processed_data']['date_range'] = (sanitized_start, sanitized_end)
    st.session_state['processed_data']['strat_filter'] = strat_filter

    # Retrieve data from session state
    combined_city = st.session_state['processed_data']['combined_city']
    combined_by_region = st.session_state['processed_data']['combined_by_region']
    accident_data = st.session_state['processed_data']['accident_data']
    accident_records = st.session_state['processed_data']['accident_records']
    strategy_data = st.session_state['processed_data']['strategy_data']
    all_regions = st.session_state['processed_data']['all_regions']
    all_strategy_types = st.session_state['processed_data']['all_strategy_types']
    min_date = st.session_state['processed_data']['min_date']
    max_date = st.session_state['processed_data']['max_date']
    region_sel = st.session_state['processed_data']['region_sel']
    date_range = st.session_state['processed_data']['date_range']
    strat_filter = st.session_state['processed_data']['strat_filter']
    accident_source_name = st.session_state['processed_data']['accident_source_name']

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

        tab_labels = [
            "🏠 总览",
            "📍 事故热点",
            "🔍 AI 分析",
            "📈 预测模型",
            "📊 模型评估",
            "⚠️ 异常检测",
            "📝 策略评估",
            "⚖️ 策略对比",
            "🧪 情景模拟",
        ]
        default_tab = st.session_state.get("active_tab", tab_labels[0])
        if default_tab not in tab_labels:
            default_tab = tab_labels[0]
        selected_tab = st.radio(
            "功能分区",
            tab_labels,
            index=tab_labels.index(default_tab),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state["active_tab"] = selected_tab


        if selected_tab == "🏠 总览":
            if render_overview is not None:
                render_overview(base, region_sel, start_dt, end_dt, strat_filter)
            else:
                st.warning("概览模块未能加载，请检查 `ui_sections/overview.py`。")

        elif selected_tab == "📍 事故热点":
            if render_hotspot is not None:
                render_hotspot(accident_records, accident_source_name)
            else:
                st.warning("事故热点模块未能加载，请检查 `ui_sections/hotspot.py`。")

        elif selected_tab == "🔍 AI 分析":
            from openai import OpenAI
            st.subheader("AI 数据分析与改进建议")
            if not HAS_OPENAI:
                st.warning("未安装 `openai` 库。请安装后重试。")
            elif not openai_api_key:
                st.info("请在左侧边栏输入 OpenAI API Key 以启用 AI 分析。")
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

                    prompt = (
                        "你是一名资深交通安全数据分析顾问。请基于以下结构化数据输出一份专业报告，需包含：\n"
                        "1. 核心指标洞察：按要点总结事故趋势、显著波动及可能原因。\n"
                        "2. 策略绩效评估：对比主要策略的优势、短板与适用场景。\n"
                        "3. 优化建议：为短期（0-3个月）、中期（3-12个月）与长期（12个月以上）分别给出2-3条可操作措施。\n"
                        "请保持正式语气，引用关键数值支撑结论，并用清晰的小节或列表呈现。\n"
                        f"数据摘要：{data_str}\n"
                    )
                    if st.button("上传数据至 AI 并获取分析"):
                        if not openai_api_key.strip():
                            st.info("请提供有效的 AI API Key。")
                        elif not open_ai_base_url.strip():
                            st.info("请提供可访问的 AI Base Url。")
                        else:
                            try:
                                client = OpenAI(
                                        base_url=open_ai_base_url,
                                        # sk-xxx替换为自己的key
                                        api_key=openai_api_key
                                )
                                st.markdown("### AI 分析结果与改进思路")
                                placeholder = st.empty()
                                accumulated_response: list[str] = []
                                with st.spinner("AI 正在生成专业报告，请稍候…"):
                                    stream = client.chat.completions.create(
                                        model="gpt-5-mini",
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": "You are a professional traffic safety analyst who writes concise, well-structured Chinese reports."
                                            },
                                            {"role": "user", "content": prompt},
                                        ],
                                        stream=True,
                                    )
                                    for chunk in stream:
                                        delta = chunk.choices[0].delta if chunk.choices else None
                                        piece = getattr(delta, "content", None) if delta else None
                                        if piece:
                                            accumulated_response.append(piece)
                                            placeholder.markdown("".join(accumulated_response), unsafe_allow_html=True)
                                final_text = "".join(accumulated_response)
                                if not final_text:
                                    placeholder.info("AI 未返回可用内容，请稍后重试或检查凭据配置。")
                            except Exception as e:
                                st.error(f"调用 OpenAI API 失败：{str(e)}")
                else:
                    st.warning("没有策略数据可供分析。")

                # Update refresh time
                st.session_state['last_refresh'] = datetime.now()

        elif selected_tab == "📈 预测模型":
            if render_forecast is not None:
                render_forecast(base)
            else:
                st.subheader("多模型预测比较")
                # 使用表单封装交互组件
                with st.form(key="predict_form"):
                    # 缩短默认回溯窗口，提升首次渲染速度
                    default_date = base.index.max() - pd.Timedelta(days=30) if len(base) else pd.Timestamp('2022-01-01')
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
        elif selected_tab == "📊 模型评估":
            if render_model_eval is not None:
                render_model_eval(base)
            else:
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
        elif selected_tab == "⚠️ 异常检测":
            anomalies, anomaly_fig = detect_anomalies(base['accident_count'])
            st.plotly_chart(anomaly_fig, use_container_width=True)
            st.write(f"检测到异常点：{len(anomalies)} 个")
            st.download_button("下载异常日期 CSV",
                            data=anomalies.to_series().to_csv(index=False).encode('utf-8-sig'),
                            file_name="anomalies.csv", mime="text/csv")

        # --- Tab 5: 策略评估
        elif selected_tab == "📝 策略评估":
            if render_strategy_eval is not None:
                render_strategy_eval(base, all_strategy_types, region_sel)
            else:
                st.warning("策略评估模块不可用，请检查 `ui_sections/strategy_eval.py`。")

        # --- Tab 6: 策略对比
        elif selected_tab == "⚖️ 策略对比":
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
        elif selected_tab == "🧪 情景模拟":
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

    else:
        st.info("请先在左侧上传事故数据与策略数据，并点击“应用数据与筛选”按钮。")

if __name__ == "__main__":
    run_streamlit_app()
