from __future__ import annotations

import json
from datetime import datetime

import plotly.express as px
import streamlit as st

from services.hotspot import (
    analyze_hotspot_frequency,
    calculate_hotspot_risk_score,
    generate_hotspot_strategies,
    prepare_hotspot_dataset,
    serialise_datetime_columns,
)


@st.cache_data(show_spinner=False)
def _prepare_hotspot_data(df):
    return prepare_hotspot_dataset(df)


def render_hotspot(accident_records, accident_source_name: str | None) -> None:
    st.header("📍 事故多发路口分析")
    st.markdown("独立分析事故数据，识别高风险路口并生成针对性策略。")

    if accident_records is None:
        st.info("请在左侧上传事故数据并点击“应用数据与筛选”后再执行热点分析。")
        st.markdown(
            """
            ### 📝 支持的数据格式要求：
            - **文件格式**：Excel (.xlsx)
            - **必要字段**：
              - `事故时间`
              - `事故类型`
              - `事故具体地点`
              - `所在街道`
              - `道路类型`
              - `路口路段类型`
            """
        )
        return

    with st.spinner("正在准备事故热点数据…"):
        hotspot_data = _prepare_hotspot_data(accident_records)

    st.success(f"✅ 成功加载数据：{len(hotspot_data)} 条事故记录")

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric(
            "数据时间范围",
            f"{hotspot_data['事故时间'].min().strftime('%Y-%m-%d')} 至 {hotspot_data['事故时间'].max().strftime('%Y-%m-%d')}",
        )
    with metric_cols[1]:
        st.metric(
            "事故类型分布",
            f"财损: {len(hotspot_data[hotspot_data['事故类型'] == '财损'])}起",
        )
    with metric_cols[2]:
        st.metric("涉及区域", f"{hotspot_data['所在街道'].nunique()}个街道")

    st.subheader("🔧 分析参数设置")
    settings_cols = st.columns(3)
    with settings_cols[0]:
        time_window = st.selectbox(
            "统计时间窗口",
            options=["7D", "15D", "30D"],
            index=0,
            key="hotspot_window",
        )
    with settings_cols[1]:
        min_accidents = st.number_input(
            "最小事故数", min_value=1, max_value=50, value=3, key="hotspot_min_accidents"
        )
    with settings_cols[2]:
        top_n = st.slider("显示热点数量", min_value=3, max_value=20, value=8, key="hotspot_top_n")

    if not st.button("🚀 开始热点分析", type="primary"):
        return

    with st.spinner("正在分析事故热点分布…"):
        hotspots = analyze_hotspot_frequency(hotspot_data, time_window=time_window)
        hotspots = hotspots[hotspots["accident_count"] >= min_accidents]

        if hotspots.empty:
            st.warning("⚠️ 未找到符合条件的事故热点数据，请调整筛选参数。")
            return

        hotspots_with_risk = calculate_hotspot_risk_score(hotspots.head(top_n * 3))
        top_hotspots = hotspots_with_risk.head(top_n)

    st.subheader("📊 事故多发路口排名（前{0}个）".format(top_n))
    display_columns = {
        "accident_count": "累计事故数",
        "recent_count": "近期事故数",
        "trend_ratio": "趋势比例",
        "main_accident_type": "主要类型",
        "main_intersection_type": "路口类型",
        "risk_score": "风险评分",
        "risk_level": "风险等级",
    }
    display_df = top_hotspots[list(display_columns.keys())].rename(columns=display_columns)
    styled_df = display_df.style.format({"趋势比例": "{:.2f}", "风险评分": "{:.1f}"}).background_gradient(
        subset=["风险评分"], cmap="Reds"
    )
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("🎯 针对性策略建议")
    strategies = generate_hotspot_strategies(top_hotspots, time_period="本周")
    for index, strategy_info in enumerate(strategies, start=1):
        message = f"**{index}. {strategy_info['strategy']}**"
        risk_level = strategy_info["risk_level"]
        if risk_level == "高风险":
            st.error(f"🚨 {message}")
        elif risk_level == "中风险":
            st.warning(f"⚠️ {message}")
        else:
            st.info(f"✅ {message}")

    st.subheader("📈 数据分析可视化")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig_hotspots = px.bar(
            top_hotspots.head(10),
            x=top_hotspots.head(10).index,
            y=["accident_count", "recent_count"],
            title="事故频次TOP10分布",
            labels={"value": "事故数量", "variable": "类型", "index": "路口名称"},
            barmode="group",
        )
        fig_hotspots.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_hotspots, use_container_width=True)

    with chart_cols[1]:
        risk_distribution = top_hotspots["risk_level"].value_counts()
        fig_risk = px.pie(
            values=risk_distribution.values,
            names=risk_distribution.index,
            title="风险等级分布",
            color_discrete_map={"高风险": "red", "中风险": "orange", "低风险": "green"},
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("💾 数据导出")
    download_cols = st.columns(2)
    with download_cols[0]:
        hotspot_csv = top_hotspots.to_csv().encode("utf-8-sig")
        st.download_button(
            "📥 下载热点数据CSV",
            data=hotspot_csv,
            file_name=f"accident_hotspots_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with download_cols[1]:
        serializable = serialise_datetime_columns(
            top_hotspots.reset_index(),
            columns=[col for col in top_hotspots.columns if "time" in col or "date" in col],
        )
        report_payload = {
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_window": time_window,
            "data_source": accident_source_name or "事故数据",
            "total_records": int(len(hotspot_data)),
            "analysis_parameters": {"min_accidents": int(min_accidents), "top_n": int(top_n)},
            "top_hotspots": serializable.to_dict("records"),
            "recommended_strategies": strategies,
            "summary": {
                "high_risk_count": int((top_hotspots["risk_level"] == "高风险").sum()),
                "medium_risk_count": int((top_hotspots["risk_level"] == "中风险").sum()),
                "total_analyzed_locations": int(len(hotspots)),
                "most_dangerous_location": top_hotspots.index[0]
                if len(top_hotspots)
                else "无",
            },
        }
        st.download_button(
            "📄 下载完整分析报告",
            data=json.dumps(report_payload, ensure_ascii=False, indent=2),
            file_name=f"hotspot_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )

    with st.expander("📋 查看原始数据预览"):
        preview_cols = ["事故时间", "所在街道", "事故类型", "事故具体地点", "道路类型"]
        preview_df = hotspot_data[preview_cols].copy()
        st.dataframe(preview_df.head(10), use_container_width=True)

