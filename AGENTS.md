# Repository Guidelines

## Project Structure & Module Organization
- Source: `app.py` (Streamlit UI, data processing, forecasting, anomaly detection, evaluation).
- Docs & outputs: `docs/`, `overview_series.html`, `strategy_evaluation_results.csv`.
- Samples: `sample/` for example data only; avoid sensitive content.
- Meta: `requirements.txt`, `readme.md`, `LICENSE`, `CHANGELOG.md`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate` (or follow conda steps in `readme.md`).
- Install deps: `pip install -r requirements.txt`.
- Run app: `streamlit run app.py` then open `http://localhost:8501`.
- Export artifacts: charts save as HTML (Plotly); forecasts may be written to CSV as noted in `readme.md`.

## Coding Style & Naming Conventions
- Python ≥3.8; 4-space indentation; UTF-8.
- Names: functions/variables `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Files: keep scope focused; use descriptive output names (e.g., `arima_forecast.csv`).
- Data handling: prefer pandas/NumPy vectorization; validate inputs; avoid global state except constants.

## Testing Guidelines
- Framework: pytest (recommended). Place tests under `tests/`.
- Naming: `test_<module>.py` and `test_<behavior>()`.
- Run: `pytest -q`. Focus on `load_and_clean_data`, aggregation, model selection, and metrics.
- Keep tests fast and deterministic; avoid large I/O. Use small DataFrame fixtures.

## Commit & Pull Request Guidelines
- Messages: concise, present tense. Prefixes seen: `modify:`, `Add`, `Update`.
- Include scope and reason: e.g., `modify: update requirements for statsmodels`.
- PRs: clear description, linked issues, repro steps/screenshots for UI, and notes on any schema or output changes.

## Security & Configuration Tips
- Do not commit real accident data or secrets. Use `sample/` for examples.
- Optional envs: `LOG_LEVEL=DEBUG`. Keep any API keys in environment variables, not in code.
- Validate Excel column names before processing; handle missing columns/rows defensively.

