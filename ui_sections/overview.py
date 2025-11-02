from __future__ import annotations

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def render_overview(base: pd.DataFrame, region_sel: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                    strat_filter: list[str]):
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=base.index, y=base['accident_count'], name='事故数', mode='lines'))
    fig_line.update_layout(title="事故数（过滤后）", xaxis_title="Date", yaxis_title="Count")
    st.plotly_chart(fig_line, use_container_width=True)

    html = fig_line.to_html(full_html=True, include_plotlyjs='cdn')
    st.download_button("下载图表 HTML", data=html.encode('utf-8'),
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
        "max_date": str(base.index.max().date()) if len(base) else None,
    }
    st.download_button("下载运行参数 JSON", data=json.dumps(meta, ensure_ascii=False, indent=2).encode('utf-8'),
                       file_name="run_metadata.json", mime="application/json")

