# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py

# Run tests (if tests/ directory exists)
pytest -q
```

## Architecture Overview

This is a Streamlit-based traffic safety analysis system with a three-layer architecture:

### Layer Structure

```
app.py (Main Entry & UI Orchestration)
    ↓
ui_sections/ (UI Components - render_* functions)
    ↓
services/ (Business Logic)
    ↓
config/settings.py (Configuration)
```

### Data Flow

1. **Input**: Excel files uploaded via Streamlit sidebar (事故数据 + 策略数据)
2. **Processing**: `services/io.py` handles loading, column aliasing, and cleaning
3. **Aggregation**: Data aggregated to daily time series with `aggregate_daily_data()`
4. **Analysis**: Various services process the aggregated data
5. **Output**: Interactive Plotly charts, CSV exports, AI-generated reports

### Key Services

| Module | Purpose |
|--------|---------|
| `services/io.py` | Data loading, column normalization (COLUMN_ALIASES), region inference |
| `services/forecast.py` | ARIMA grid search, KNN counterfactual, GLM/SVR extrapolation |
| `services/strategy.py` | Strategy effectiveness evaluation (F1/F2 metrics, safety states) |
| `services/hotspot.py` | Location extraction, risk scoring, strategy generation |
| `services/metrics.py` | Model evaluation metrics (RMSE, MAE) |

### UI Sections

Each tab in the app corresponds to a `render_*` function in `ui_sections/`:
- `render_overview`: KPI dashboard and time series visualization
- `render_forecast`: Multi-model prediction comparison
- `render_model_eval`: Model accuracy metrics
- `render_strategy_eval`: Single strategy evaluation
- `render_hotspot`: Accident hotspot analysis with risk levels

### Session State Pattern

The app uses `st.session_state['processed_data']` to persist:
- Loaded DataFrames (`combined_city`, `combined_by_region`, `accident_records`)
- Filter state (`region_sel`, `date_range`, `strat_filter`)
- Derived metadata (`all_regions`, `all_strategy_types`, `min_date`, `max_date`)

### AI Integration

Uses DeepSeek API (OpenAI-compatible) for generating analysis reports. Configuration in sidebar:
- Base URL: `https://api.deepseek.com`
- Model: `deepseek-chat`
- Streaming response rendered incrementally

## Coding Conventions

- Python 3.8+ with type hints (`from __future__ import annotations`)
- Functions/variables: `snake_case`; Classes: `PascalCase`; Constants: `UPPER_SNAKE_CASE`
- Use `@st.cache_data` for expensive computations
- Column aliases defined in `COLUMN_ALIASES` dict for flexible Excel input
- Prefer pandas vectorization over loops

## Data Format Requirements

**Accident Data Excel** must contain (or aliases of):
- `事故时间` (accident time)
- `所在街道` (street/region)
- `事故类型` (accident type: 财损/伤人/亡人)

**Strategy Data Excel** must contain:
- `发布时间` (publish date)
- `交通策略类型` (strategy type)

## Configuration (config/settings.py)

Key parameters:
- `ARIMA_P/D/Q`: Grid search ranges for ARIMA
- `MIN_PRE_DAYS` / `MAX_PRE_DAYS`: Historical data requirements
- `ANOMALY_CONTAMINATION`: Isolation Forest contamination rate
