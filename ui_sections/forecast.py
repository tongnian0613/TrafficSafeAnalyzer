from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.forecast import (
    arima_forecast_with_grid_search,
    knn_forecast_counterfactual,
    fit_and_extrapolate,
)


def render_forecast(base: pd.DataFrame):
    st.subheader("多模型预测比较")

    if base is None or base.empty:
        st.info("暂无可用于预测的事故数据，请先在侧边栏上传数据并应用筛选。")
        st.session_state.setdefault(
            "forecast_state",
            {"results": None, "last_message": "暂无可用于预测的事故数据。"},
        )
        return

    forecast_state = st.session_state.setdefault(
        "forecast_state",
        {
            "selected_date": None,
            "horizon": 30,
            "results": None,
            "last_message": None,
            "data_signature": None,
        },
    )

    earliest_date = base.index.min().date()
    latest_date = base.index.max().date()
    fallback_date = max(
        (base.index.max() - pd.Timedelta(days=30)).date(),
        earliest_date,
    )
    current_signature = (
        earliest_date.isoformat(),
        latest_date.isoformat(),
        int(len(base)),
        float(base["accident_count"].sum()),
    )

    # Reset cached results if the underlying dataset has changed
    if forecast_state.get("data_signature") != current_signature:
        forecast_state.update(
            {
                "data_signature": current_signature,
                "results": None,
                "last_message": None,
                "selected_date": fallback_date,
            }
        )

    default_date = forecast_state.get("selected_date") or fallback_date
    if default_date < earliest_date:
        default_date = earliest_date
    if default_date > latest_date:
        default_date = latest_date

    with st.form(key="predict_form"):
        selected_date = st.date_input(
            "选择干预日期 / 预测起点",
            value=default_date,
            min_value=earliest_date,
            max_value=latest_date,
        )
        horizon = st.number_input(
            "预测天数",
            min_value=7,
            max_value=90,
            value=int(forecast_state.get("horizon", 30)),
            step=1,
        )
        submit_predict = st.form_submit_button("应用预测参数")

    forecast_state["selected_date"] = selected_date
    forecast_state["horizon"] = int(horizon)

    if submit_predict:
        history = base.loc[:pd.to_datetime(selected_date)]
        if len(history) < 10:
            forecast_state.update(
                {
                    "results": None,
                    "last_message": "干预前数据不足（至少需要 10 个观测点）。",
                }
            )
        else:
            with st.spinner("正在生成预测结果…"):
                warnings: list[str] = []
                try:
                    train_series = history["accident_count"]
                    arima_df = arima_forecast_with_grid_search(
                        train_series,
                        start_date=pd.to_datetime(selected_date) + pd.Timedelta(days=1),
                        horizon=int(horizon),
                    )
                except Exception as exc:
                    arima_df = None
                    warnings.append(f"ARIMA 运行失败：{exc}")

                knn_pred, _ = knn_forecast_counterfactual(
                    base["accident_count"],
                    pd.to_datetime(selected_date),
                    horizon=int(horizon),
                )
                if knn_pred is None:
                    warnings.append("KNN 预测未生成结果（历史数据不足或维度不满足要求）。")

                glm_pred, svr_pred, _ = fit_and_extrapolate(
                    base["accident_count"],
                    pd.to_datetime(selected_date),
                    days=int(horizon),
                )
                if glm_pred is None and svr_pred is None:
                    warnings.append("GLM/SVR 预测未生成结果，建议缩短预测窗口或检查源数据。")

                forecast_state.update(
                    {
                        "results": {
                            "selected_date": selected_date,
                            "horizon": int(horizon),
                            "arima_df": arima_df,
                            "knn_pred": knn_pred,
                            "glm_pred": glm_pred,
                            "svr_pred": svr_pred,
                            "warnings": warnings,
                        },
                        "last_message": None,
                    }
                )

    results = forecast_state.get("results")
    if not results:
        if forecast_state.get("last_message"):
            st.warning(forecast_state["last_message"])
        else:
            st.info("请设置预测参数并点击“应用预测参数”按钮。")
        return

    first_date = pd.to_datetime(results["selected_date"])
    horizon_days = int(results["horizon"])
    arima_df = results["arima_df"]
    knn_pred = results["knn_pred"]
    glm_pred = results["glm_pred"]
    svr_pred = results["svr_pred"]

    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(x=base.index, y=base["accident_count"], name="实际", mode="lines")
    )
    if arima_df is not None:
        fig_pred.add_trace(
            go.Scatter(x=arima_df.index, y=arima_df["forecast"], name="ARIMA", mode="lines")
        )
    if knn_pred is not None:
        fig_pred.add_trace(go.Scatter(x=knn_pred.index, y=knn_pred, name="KNN", mode="lines"))
    if glm_pred is not None:
        fig_pred.add_trace(go.Scatter(x=glm_pred.index, y=glm_pred, name="GLM", mode="lines"))
    if svr_pred is not None:
        fig_pred.add_trace(go.Scatter(x=svr_pred.index, y=svr_pred, name="SVR", mode="lines"))

    fig_pred.update_layout(
        title=f"多模型预测比较（起点：{first_date.date()}，预测 {horizon_days} 天）",
        xaxis_title="日期",
        yaxis_title="事故数",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    if arima_df is not None:
        st.download_button(
            "下载 ARIMA 预测 CSV",
            data=arima_df.to_csv().encode("utf-8-sig"),
            file_name="arima_forecast.csv",
            mime="text/csv",
        )

    for warning_text in results.get("warnings", []):
        st.warning(warning_text)
