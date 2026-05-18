"""
Forecasting methods for Farmley Sales Data.
Each function takes a pd.Series with DatetimeIndex (monthly) and returns a list of 9 forecasted values.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _safe_series(s):
    vals = s.values.astype(float)
    if np.all(vals == 0):
        return vals, False
    return vals, True


# ─── 1. SEASONAL NAIVE ──────────────────────────────────────────────────────────
def seasonal_naive_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    n = len(vals)
    forecast = []
    for i in range(9):
        src_idx = i % n
        forecast.append(max(0, vals[src_idx]))
    return forecast


# ─── 2. MOVING AVERAGE ──────────────────────────────────────────────────────────
def moving_average_forecast(series, window=3):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    extended = list(vals)
    for _ in range(9):
        avg = np.mean(extended[-window:])
        extended.append(max(0, avg))
    return extended[len(vals):]


# ─── 3. WEIGHTED MOVING AVERAGE ─────────────────────────────────────────────────
def weighted_moving_average_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    w_len = len(weights)
    extended = list(vals)
    for _ in range(9):
        window = extended[-w_len:]
        if len(window) < w_len:
            w = weights[-len(window):]
            w = w / w.sum()
        else:
            w = weights
        wma = np.dot(window, w)
        extended.append(max(0, wma))
    return extended[len(vals):]


# ─── 4. EXPONENTIAL SMOOTHING (Simple) ──────────────────────────────────────────
def exponential_smoothing_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    n = len(vals)

    def sse(alpha):
        a = alpha[0]
        f = [vals[0]]
        for t in range(1, n):
            f.append(a * vals[t - 1] + (1 - a) * f[-1])
        return np.sum((vals[1:] - np.array(f[1:])) ** 2)

    res = minimize(sse, [0.3], bounds=[(0.01, 0.99)], method="L-BFGS-B")
    alpha = res.x[0]

    f = [vals[0]]
    for t in range(1, n):
        f.append(alpha * vals[t - 1] + (1 - alpha) * f[-1])

    last_f = alpha * vals[-1] + (1 - alpha) * f[-1]
    forecast = [max(0, last_f)] * 9
    return forecast


# ─── 5. HOLT-WINTERS ────────────────────────────────────────────────────────────
def holt_winters_forecast(series, seasonal="mul"):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    n = len(vals)

    if n < 4:
        return moving_average_forecast(series, window=3)

    period = min(12, n)

    if seasonal == "mul" and np.any(vals[:period] <= 0):
        seasonal = "add"

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(
            vals, trend="add", seasonal=seasonal,
            seasonal_periods=period,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=True)
        fc = fit.forecast(9)
        return [max(0, v) for v in fc]
    except Exception:
        return _linear_trend_seasonal_core(vals, period)


# ─── 6. LINEAR TREND + SEASONALITY ──────────────────────────────────────────────
def linear_trend_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    return _linear_trend_seasonal_core(vals, min(12, len(vals)))


def _linear_trend_seasonal_core(vals, period):
    n = len(vals)
    t = np.arange(n)

    coeffs = np.polyfit(t, vals, 1)
    trend_line = np.polyval(coeffs, t)

    if np.any(trend_line == 0):
        ratios = vals - trend_line
        seasonal_idx = np.zeros(period)
        for i in range(period):
            month_vals = [ratios[j] for j in range(i, n, period)]
            seasonal_idx[i] = np.mean(month_vals) if month_vals else 0
        forecast = []
        for i in range(9):
            future_t = n + i
            trend_val = np.polyval(coeffs, future_t)
            s_idx = (n + i) % period
            forecast.append(max(0, trend_val + seasonal_idx[s_idx]))
    else:
        ratios = vals / trend_line
        seasonal_idx = np.ones(period)
        for i in range(period):
            month_vals = [ratios[j] for j in range(i, n, period)]
            seasonal_idx[i] = np.mean(month_vals) if month_vals else 1.0
        forecast = []
        for i in range(9):
            future_t = n + i
            trend_val = np.polyval(coeffs, future_t)
            s_idx = (n + i) % period
            forecast.append(max(0, trend_val * seasonal_idx[s_idx]))

    return forecast


# ─── 7. SEASONAL DECOMPOSITION ──────────────────────────────────────────────────
def seasonal_decomposition_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    n = len(vals)
    period = min(12, n)

    if n < 4:
        return moving_average_forecast(series, window=3)

    try:
        from statsmodels.tsa.seasonal import STL
        s = pd.Series(vals, index=pd.date_range("2025-04-01", periods=n, freq="MS"))
        stl = STL(s, period=period, robust=True)
        result = stl.fit()

        trend = result.trend.values
        seasonal = result.seasonal.values

        t_idx = np.arange(n)
        coeffs = np.polyfit(t_idx, trend, 1)

        forecast = []
        for i in range(9):
            future_t = n + i
            trend_val = np.polyval(coeffs, future_t)
            s_val = seasonal[(n + i) % period]
            forecast.append(max(0, trend_val + s_val))
        return forecast
    except Exception:
        return _linear_trend_seasonal_core(vals, period)


# ─── 8. ENSEMBLE ────────────────────────────────────────────────────────────────
def ensemble_forecast(series):
    methods = [
        seasonal_naive_forecast,
        lambda s: holt_winters_forecast(s, seasonal="mul"),
        linear_trend_forecast,
    ]
    results = []
    for fn in methods:
        try:
            results.append(fn(series))
        except Exception:
            pass

    if not results:
        return [0.0] * 9

    arr = np.array(results)
    return [max(0, v) for v in np.median(arr, axis=0)]


# ─── METHOD DESCRIPTIONS ────────────────────────────────────────────────────────
METHOD_DESCRIPTIONS = {
    "Seasonal Naive": """
**Concept:** The simplest seasonal method — forecast for each month equals the actual value from the same month in the previous year.

**Formula:** `F(Apr-2026) = Actual(Apr-2025)`, `F(May-2026) = Actual(May-2025)`, etc.

**When it works well:** When products have strong, stable seasonal patterns and little year-over-year growth. This is the baseline benchmark — any good method should beat this.

**Limitation:** Ignores trends (growth/decline). If sales are growing 20% YoY, this will under-forecast.
""",

    "Moving Average (3M)": """
**Concept:** Each forecast is the average of the 3 most recent values. As forecasts are generated, they feed into subsequent calculations.

**Formula:** `F(t) = [A(t-1) + A(t-2) + A(t-3)] / 3`

**When it works well:** Smooths out short-term noise. Good for relatively stable items without strong seasonality.

**Limitation:** Loses all seasonal pattern information — averages flatten peaks and valleys. Multi-step forecasts converge to a flat line.
""",

    "Moving Average (6M)": """
**Concept:** Same as 3M moving average but uses 6 months, giving a smoother (but more lagged) forecast.

**Formula:** `F(t) = [A(t-1) + ... + A(t-6)] / 6`

**When it works well:** More stable than 3M for items with high month-to-month volatility.

**Limitation:** Even more seasonal pattern loss than 3M. Very slow to react to trend changes.
""",

    "Weighted Moving Average": """
**Concept:** Like moving average, but recent months get higher weights (30%, 25%, 20%, 15%, 10% for last 5 months).

**Formula:** `F(t) = 0.30*A(t-1) + 0.25*A(t-2) + 0.20*A(t-3) + 0.15*A(t-4) + 0.10*A(t-5)`

**When it works well:** Better than simple MA when recent data is more relevant. Reacts faster to trend changes.

**Limitation:** Still loses seasonal patterns. The weights are fixed — ideally they'd be optimized per item.
""",

    "Exponential Smoothing": """
**Concept:** A weighted average where the weight (alpha) decays exponentially for older observations. Alpha is optimized by minimizing sum of squared errors on historical data.

**Formula:** `F(t) = α * A(t-1) + (1-α) * F(t-1)`, where α is optimized (0 < α < 1).

**When it works well:** Good for items with no trend or seasonality — produces a "best level estimate."

**Limitation:** Single exponential smoothing produces a flat forecast (same value for all 9 months). Does not capture trend or seasonality.
""",

    "Holt-Winters (Additive)": """
**Concept:** Extends exponential smoothing with trend and additive seasonal components. The model has three equations updating level, trend, and seasonal factors.

**Formula:**
- Level: `L(t) = α(Y(t) - S(t-p)) + (1-α)(L(t-1) + T(t-1))`
- Trend: `T(t) = β(L(t) - L(t-1)) + (1-β)T(t-1)`
- Seasonal: `S(t) = γ(Y(t) - L(t)) + (1-γ)S(t-p)`
- Forecast: `F(t+h) = L(t) + h*T(t) + S(t+h-p)`

**When it works well:** When seasonal swings are roughly constant in absolute terms (e.g., always ±500 units in festive months).

**Limitation:** Can produce negative forecasts for items with low base values and large seasonal swings.
""",

    "Holt-Winters (Multiplicative)": """
**Concept:** Like additive Holt-Winters, but seasonal component is multiplicative — seasonal effect is a ratio rather than absolute amount.

**Formula:**
- Level: `L(t) = α(Y(t) / S(t-p)) + (1-α)(L(t-1) + T(t-1))`
- Seasonal: `S(t) = γ(Y(t) / L(t)) + (1-γ)S(t-p)`
- Forecast: `F(t+h) = (L(t) + h*T(t)) * S(t+h-p)`

**When it works well:** Best for your data — when seasonal peaks scale proportionally with the level (e.g., September is always ~2x average, not always +5000 units). This is common in FMCG/food products like Farmley's.

**Limitation:** Cannot handle zero values (division by zero). Falls back to additive when zeros exist.
""",

    "Linear Trend + Seasonality": """
**Concept:** Decomposes the data into a linear trend (using OLS regression) and multiplicative seasonal indices (average ratio of actual to trend for each month position).

**Formula:**
1. Fit `Trend(t) = a + b*t` via least squares
2. Compute seasonal index: `S(month) = AVG(Actual(t) / Trend(t))` for each calendar month
3. Forecast: `F(t) = Trend(t) * S(month)`

**When it works well:** Captures both growth trends and repeating seasonal patterns. Robust and interpretable. Good when you have at least 6-12 months of data.

**Limitation:** Assumes trend is strictly linear — won't capture accelerating growth or plateau effects.
""",

    "Seasonal Decomposition (STL)": """
**Concept:** Uses STL (Seasonal and Trend decomposition using LOESS) to separate the time series into trend, seasonal, and residual components. Trend is extrapolated forward using linear regression, and the seasonal pattern repeats.

**Formula:**
1. STL decomposes: `Y(t) = Trend(t) + Seasonal(t) + Residual(t)`
2. Extrapolate trend linearly
3. Forecast: `F(t) = Extrapolated_Trend(t) + Seasonal(t mod period)`

**When it works well:** Most robust decomposition method — handles non-linear trends better than classical decomposition. Good for noisy data.

**Limitation:** With only 12 months of data, the decomposition may not fully separate trend from seasonal. Better with 2+ years of data.
""",

    "Ensemble (Best-of-3)": """
**Concept:** Takes the median of three complementary methods: Seasonal Naive, Holt-Winters (Multiplicative), and Linear Trend + Seasonality. The median is more robust than any single method.

**Formula:** `F(t) = MEDIAN(Seasonal_Naive(t), Holt_Winters_Mul(t), Linear_Trend(t))`

**Why these three?**
- Seasonal Naive = pure seasonality baseline
- Holt-Winters Mul = adaptive seasonal model
- Linear Trend = interpretable trend + seasonal model

The median ignores outlier forecasts from any single method that might go off.

**When it works well:** Best general-purpose choice when you're unsure which method suits an item. Reduces the risk of any single method producing wild numbers.

**Limitation:** Will never be the best for any single item — but will rarely be the worst either. The safe choice.
""",
}
