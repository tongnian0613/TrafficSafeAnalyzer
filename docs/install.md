# Installation Guide

This document explains how to set up TrafficSafeAnalyzer for local development and exploration. The application runs on Streamlit and officially supports Python 3.8.

## Prerequisites
- Python 3.8 (3.9+ is not yet validated; use 3.8 to avoid dependency issues)
- Git
- `pip` (bundled with Python)
- Optional: Conda (for environment management) or Docker (for container-based runs)

## 1. Obtain the source code

```bash
git clone https://github.com/tongnian0613/TrafficSafeAnalyzer.git
cd TrafficSafeAnalyzer
```

If you already have the repository, pull the latest changes instead:

```bash
git pull origin main
```

## 2. Create a dedicated environment

### Option A: Built-in virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Option B: Conda environment

```bash
conda create -n trafficsa python=3.8 -y
conda activate trafficsa
```

## 3. Install project dependencies

Install the full dependency set listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you prefer a minimal installation before pulling in extras, install the core stack first:

```bash
pip install streamlit pandas numpy matplotlib plotly scikit-learn statsmodels scipy
```

Then add optional packages as needed (Excel readers, auto-refresh, OpenAI integration):

```bash
pip install streamlit-autorefresh openpyxl xlrd cryptography openai
```

## 4. Verify the setup

1. Ensure the environment is still active (`which python` should point to `.venv` or the conda env).
2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open `http://localhost:8501` in your browser. The home page should load without import errors.

## 5. Run with Docker (optional)

If you prefer an isolated container build, use the included `Dockerfile`:

```bash
docker build -t trafficsafeanalyzer .
docker run --rm -p 8501:8501 trafficsafeanalyzer
```

To work with local data, mount the host folder containing Excel files:

```bash
docker run --rm -p 8501:8501 \
  -v "$(pwd)/sample:/app/sample" \
  trafficsafeanalyzer
```

The container exposes Streamlit on port 8501 by default. Override configuration via environment variables when needed, for example `-e STREAMLIT_SERVER_PORT=8502`.

## Troubleshooting tips

- **Missing package**: Re-run `pip install -r requirements.txt`.
- **Python version mismatch**: Confirm `python --version` reports 3.8.x inside your environment.
- **OpenSSL or cryptography errors** (macOS/Linux): Update the system OpenSSL libraries and reinstall `cryptography`.
- **Taking too long to install**: if a dependency download stalls due to a firewall, retry using a mirror (`-i https://pypi.tuna.tsinghua.edu.cn/simple`) consistent with your environment policy.

After a successful launch, continue with the usage guide in `docs/usage.md` to load data and explore forecasts.
