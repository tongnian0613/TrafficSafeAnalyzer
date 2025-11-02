from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from services.strategy import generate_output_and_recommendations


def render_strategy_eval(base: pd.DataFrame, all_strategy_types: list[str], region_sel: str):
    st.info(f"📌 检测到的策略类型：{', '.join(all_strategy_types) or '（数据中没有策略）'}")
    if not all_strategy_types:
        st.warning("数据中没有检测到策略。")
        return

    with st.form(key="strategy_eval_form"):
        horizon_eval = st.slider("评估窗口（天）", 7, 60, 14, step=1)
        submit_strat_eval = st.form_submit_button("应用评估参数")

    if not submit_strat_eval:
        st.info("请设置评估窗口并点击“应用评估参数”按钮。")
        return

    results, recommendation = generate_output_and_recommendations(
        base,
        all_strategy_types,
        region=region_sel if region_sel != '全市' else '全市',
        horizon=horizon_eval,
    )

    if not results:
        st.warning("⚠️ 未能完成策略评估。请尝试缩短评估窗口或扩大日期范围。")
        return

    st.subheader("各策略指标")
    df_res = pd.DataFrame(results).T
    st.dataframe(df_res, use_container_width=True)
    st.success(f"⭐ 推荐：{recommendation}")

    st.download_button(
        "下载策略评估 CSV",
        data=df_res.to_csv().encode('utf-8-sig'),
        file_name="strategy_evaluation_results.csv",
        mime="text/csv",
    )

    if os.path.exists('recommendation.txt'):
        with open('recommendation.txt','r',encoding='utf-8') as f:
            st.download_button("下载推荐文本", data=f.read().encode('utf-8'), file_name="recommendation.txt")

