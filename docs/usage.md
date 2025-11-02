# Usage Guide

TrafficSafeAnalyzer delivers accident analytics and decision support through a Streamlit interface. This guide walks through the daily workflow, expected inputs, and where to find generated artefacts.

## Start the app

1. Activate your virtual or conda environment（或在容器中运行，见下）.
2. From the project root, run:

   ```bash
   streamlit run app.py
   ```

3. Open `http://localhost:8501`. Keep the terminal running while you work in the browser.

> 使用 Docker？运行 `docker build -t trafficsafeanalyzer .` 与 `docker run --rm -p 8501:8501 trafficsafeanalyzer` 后，同样访问 `http://localhost:8501`。

## Load input data

Use the sidebar form labelled “数据与筛选”.

- **Accident data (`.xlsx`)** — columns should include at minimum:
  - `事故时间` (timestamp)
  - `所在街道` (region or district)
  - `事故类型`
  - `事故数`/`accident_count` (if absent, the loader aggregates counts)
- **Strategy data (`.xlsx`)** — include:
  - `发布时间`
  - `交通策略类型`
  - optional descriptors such as `策略名称`, `策略内容`
- Select the global filters (region, date window, strategy filter) and click `应用数据与筛选`.
- Uploaded files are cached. Upload a new file or press “Rerun” to refresh after making edits.
- Sample datasets for rapid smoke testing live in `sample/事故/*.xlsx` (accidents) and `sample/交通策略/*.xlsx` (strategies); copy them before making modifications.

> Tip: `services/io.py` performs validation; rows missing key columns are dropped with a warning in the Streamlit log.

## Navigate the workspace

- **🏠 总览 (Overview)** — KPI cards, time-series plot, filtered table, and download buttons for HTML (`overview_series.html`), CSV (`filtered_view.csv`), and run metadata (`run_metadata.json`).
- **📈 预测模型 (Forecast)** — choose an intervention date and horizon, compare ARIMA / KNN / GLM / SVR forecasts, and export `arima_forecast.csv`（提交后结果会在同一数据集下保留，便于调整其他控件）。
- **📊 模型评估 (Model evaluation)** — run rolling-window backtests, inspect RMSE/MAE/MAPE, and download `model_evaluation.csv`.
- **⚠️ 异常检测 (Anomaly detection)** — isolation forest marks outliers on the accident series; tweak contamination via the main page controls.
- **📝 策略评估 (Strategy evaluation)** — Aggregates metrics per strategy type, recommends the best option, writes `strategy_evaluation_results.csv`, and updates `recommendation.txt`.
- **⚖️ 策略对比 (Strategy comparison)** — side-by-side metrics for selected strategies, useful for “what worked best last month” reviews.
- **🧪 情景模拟 (Scenario simulation)** — apply intervention models (persistent/decay, lagged effects) to test potential roll-outs.
- **🔍 AI 分析** — 默认示例 API Key/Base URL 已预填，可直接体验；如需切换自有凭据，可在侧边栏更新后生成洞察（运行时读取，不会写入磁盘）。
- **📍 事故热点 (Hotspot)** — reuse the already uploaded accident data to identify high-risk intersections and produce targeted mitigation ideas; no separate hotspot upload is required.

Each tab remembers the active filters from the sidebar so results stay consistent.

## Downloaded artefacts

Generated files are saved to the project root unless you override paths in the code:

- `overview_series.html`
- `filtered_view.csv`
- `run_metadata.json`
- `arima_forecast.csv`
- `model_evaluation.csv`
- `strategy_evaluation_results.csv`
- `recommendation.txt`

After a session, review and archive these outputs under `docs/` or a dated folder as needed.

## Operational tips

- **Auto refresh**: enable from the sidebar (requires `streamlit-autorefresh`). Set the interval in seconds for live dashboards.
- **Logging**: set `LOG_LEVEL=DEBUG` before launch to see detailed diagnostics in the terminal and Streamlit log.
- **Reset filters**: choose “全市” and the full date span, then re-run the sidebar form.
- **Common warnings**:
  - *“数据中没有检测到策略”*: verify the strategy Excel file and column names.
  - *ARIMA failures*: shorten the horizon or ensure at least 10 historical data points before the intervention date.
  - *Hotspot data issues*: ensure the accident workbook includes `事故时间`, `所在街道`, `事故类型`, and `事故具体地点` so intersections can be resolved.

Need deeper integration or batch automation? Extract the core functions from `services/` and orchestrate them in a notebook or scheduled job.
