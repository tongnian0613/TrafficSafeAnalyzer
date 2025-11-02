from __future__ import annotations

# Default configuration and feature flags

# Forecasting
ARIMA_P = range(0, 4)
ARIMA_D = range(0, 2)
ARIMA_Q = range(0, 4)

DEFAULT_HORIZON_PREDICT = 30
DEFAULT_HORIZON_EVAL = 14
MIN_PRE_DAYS = 5
MAX_PRE_DAYS = 120

# Anomaly detection
ANOMALY_N_ESTIMATORS = 50
ANOMALY_CONTAMINATION = 0.10

# Performance flags
FAST_MODE = True

