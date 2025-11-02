from __future__ import annotations

import pandas as pd
import streamlit as st

from services.metrics import evaluate_models


def render_model_eval(base: pd.DataFrame):
    st.subheader("模型预测效果对比")
    with st.form(key="model_eval_form"):
        horizon_sel = st.slider("评估窗口（天）", 7, 60, 30, step=1)
        submit_eval = st.form_submit_button("应用评估参数")

    if not submit_eval:
        st.info("请设置评估窗口并点击“应用评估参数”按钮。")
        return

    try:
        df_metrics = evaluate_models(base['accident_count'], horizon=int(horizon_sel))
        st.dataframe(df_metrics, use_container_width=True)
        best_model = df_metrics['RMSE'].idxmin()
        st.success(f"过去 {int(horizon_sel)} 天中，RMSE 最低的模型是：**{best_model}**")
        st.download_button(
            "下载评估结果 CSV",
            data=df_metrics.to_csv().encode('utf-8-sig'),
            file_name="model_evaluation.csv",
            mime="text/csv",
        )
    except ValueError as err:
        st.warning(str(err))

