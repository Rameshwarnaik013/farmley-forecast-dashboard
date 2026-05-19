"""
Forecasting methods for Farmley Sales Data.
Pure numpy/pandas implementation — no statsmodels or scipy needed.
Each function takes a pd.Series with DatetimeIndex (monthly) and returns a list of 9 forecasted values.
"""
import numpy as np
import pandas as pd


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
    return [max(0.0, float(vals[i % n])) for i in range(9)]


# ─── 2. MOVING AVERAGE ──────────────────────────────────────────────────────────
def moving_average_forecast(series, window=3):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    extended = list(vals)
    for _ in range(9):
        extended.append(max(0.0, float(np.mean(extended[-window:]))))
    return extended[len(vals):]


# ─── 3. WEIGHTED MOVING AVERAGE ─────────────────────────────────────────────────
def weighted_moving_average_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    w_len = len(weights)
    extended = list(vals)
    for _ in range(9):
        window = extended[-w_len:]
        if len(window) < w_len:
            w = weights[-len(window):]
            w = w / w.sum()
        else:
            w = weights
        extended.append(max(0.0, float(np.dot(window, w))))
    return extended[len(vals):]


# ─── 4. EXPONENTIAL SMOOTHING (Simple) ──────────────────────────────────────────
def exponential_smoothing_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    n = len(vals)

    best_alpha, best_sse = 0.3, float("inf")
    for a_int in range(5, 96, 5):
        a = a_int / 100.0
        f = [vals[0]]
        for t in range(1, n):
            f.append(a * vals[t - 1] + (1 - a) * f[-1])
        sse = float(np.sum((vals[1:] - np.array(f[1:])) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_alpha = a

    f = [vals[0]]
    for t in range(1, n):
        f.append(best_alpha * vals[t - 1] + (1 - best_alpha) * f[-1])
    last_f = best_alpha * vals[-1] + (1 - best_alpha) * f[-1]
    return [max(0.0, float(last_f))] * 9


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

    best_fc, best_sse = None, float("inf")

    for alpha_i in range(10, 91, 20):
        for beta_i in range(1, 52, 25):
            for gamma_i in range(10, 91, 20):
                a = alpha_i / 100.0
                b = beta_i / 100.0
                g = gamma_i / 100.0
                try:
                    fc = _hw_core(vals, n, period, a, b, g, seasonal)
                    fitted = _hw_fitted(vals, n, period, a, b, g, seasonal)
                    sse = float(np.sum((vals[period:] - np.array(fitted[period:])) ** 2))
                    if sse < best_sse:
                        best_sse = sse
                        best_fc = fc
                except Exception:
                    continue

    if best_fc is not None:
        return best_fc
    return _linear_trend_seasonal_core(vals, period)


def _hw_core(vals, n, period, alpha, beta, gamma, seasonal):
    level = np.mean(vals[:period])
    trend = (np.mean(vals[period // 2:period]) - np.mean(vals[:period // 2])) / (period // 2) if period >= 4 else 0.0

    if seasonal == "mul":
        season = np.array([vals[i] / max(level, 1e-10) for i in range(period)])
    else:
        season = np.array([vals[i] - level for i in range(period)])

    for t in range(period, n):
        si = t % period
        if seasonal == "mul":
            new_level = alpha * (vals[t] / max(season[si], 1e-10)) + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            season[si] = gamma * (vals[t] / max(new_level, 1e-10)) + (1 - gamma) * season[si]
        else:
            new_level = alpha * (vals[t] - season[si]) + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            season[si] = gamma * (vals[t] - new_level) + (1 - gamma) * season[si]
        level = new_level
        trend = new_trend

    forecast = []
    for h in range(1, 10):
        si = (n + h - 1) % period
        if seasonal == "mul":
            forecast.append(max(0.0, float((level + h * trend) * season[si])))
        else:
            forecast.append(max(0.0, float(level + h * trend + season[si])))
    return forecast


def _hw_fitted(vals, n, period, alpha, beta, gamma, seasonal):
    level = np.mean(vals[:period])
    trend = (np.mean(vals[period // 2:period]) - np.mean(vals[:period // 2])) / (period // 2) if period >= 4 else 0.0

    if seasonal == "mul":
        season = np.array([vals[i] / max(level, 1e-10) for i in range(period)])
    else:
        season = np.array([vals[i] - level for i in range(period)])

    fitted = list(vals[:period])
    for t in range(period, n):
        si = t % period
        if seasonal == "mul":
            fitted.append(float((level + trend) * season[si]))
            new_level = alpha * (vals[t] / max(season[si], 1e-10)) + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            season[si] = gamma * (vals[t] / max(new_level, 1e-10)) + (1 - gamma) * season[si]
        else:
            fitted.append(float(level + trend + season[si]))
            new_level = alpha * (vals[t] - season[si]) + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            season[si] = gamma * (vals[t] - new_level) + (1 - gamma) * season[si]
        level = new_level
        trend = new_trend
    return fitted


# ─── 6. LINEAR TREND + SEASONALITY ──────────────────────────────────────────────
def linear_trend_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    return _linear_trend_seasonal_core(vals, min(12, len(vals)))


def _linear_trend_seasonal_core(vals, period):
    n = len(vals)
    t = np.arange(n, dtype=float)

    t_mean = np.mean(t)
    v_mean = np.mean(vals)
    slope = np.sum((t - t_mean) * (vals - v_mean)) / max(np.sum((t - t_mean) ** 2), 1e-10)
    intercept = v_mean - slope * t_mean
    trend_line = intercept + slope * t

    if np.any(np.abs(trend_line) < 1e-10):
        deseason = vals - trend_line
        seasonal_idx = np.zeros(period)
        for i in range(period):
            month_vals = [deseason[j] for j in range(i, n, period)]
            seasonal_idx[i] = np.mean(month_vals) if month_vals else 0.0
        forecast = []
        for i in range(9):
            future_t = n + i
            trend_val = intercept + slope * future_t
            forecast.append(max(0.0, float(trend_val + seasonal_idx[(n + i) % period])))
    else:
        ratios = vals / trend_line
        seasonal_idx = np.ones(period)
        for i in range(period):
            month_vals = [ratios[j] for j in range(i, n, period)]
            seasonal_idx[i] = np.mean(month_vals) if month_vals else 1.0
        forecast = []
        for i in range(9):
            future_t = n + i
            trend_val = intercept + slope * future_t
            forecast.append(max(0.0, float(trend_val * seasonal_idx[(n + i) % period])))
    return forecast


# ─── 7. SEASONAL DECOMPOSITION (manual STL-like) ────────────────────────────────
def seasonal_decomposition_forecast(series):
    vals, ok = _safe_series(series)
    if not ok:
        return [0.0] * 9
    n = len(vals)
    period = min(12, n)
    if n < 4:
        return moving_average_forecast(series, window=3)

    # Moving average to extract trend
    if period >= 3:
        half = period // 2
        trend = np.full(n, np.nan)
        for i in range(half, n - half):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            trend[i] = np.mean(vals[start:end])
        mask = ~np.isnan(trend)
        if np.sum(mask) < 2:
            return _linear_trend_seasonal_core(vals, period)
        t_idx = np.arange(n, dtype=float)
        t_valid = t_idx[mask]
        trend_valid = trend[mask]
        t_mean = np.mean(t_valid)
        tr_mean = np.mean(trend_valid)
        slope = np.sum((t_valid - t_mean) * (trend_valid - tr_mean)) / max(np.sum((t_valid - t_mean) ** 2), 1e-10)
        intercept = tr_mean - slope * t_mean
        full_trend = intercept + slope * t_idx
    else:
        return _linear_trend_seasonal_core(vals, period)

    detrended = vals - full_trend
    seasonal = np.zeros(period)
    for i in range(period):
        month_vals = [detrended[j] for j in range(i, n, period)]
        seasonal[i] = np.mean(month_vals) if month_vals else 0.0

    forecast = []
    for i in range(9):
        future_t = n + i
        trend_val = intercept + slope * future_t
        s_val = seasonal[(n + i) % period]
        forecast.append(max(0.0, float(trend_val + s_val)))
    return forecast


def holt_winters_components(series, seasonal="mul"):
    """Return (level, trend, seasonal_factors, period, seasonal_type) for Excel helper columns."""
    vals, ok = _safe_series(series)
    if not ok:
        return 0.0, 0.0, [0.0]*12, 12, seasonal
    n = len(vals)
    if n < 4:
        return 0.0, 0.0, [1.0]*12, 12, seasonal
    period = min(12, n)
    if seasonal == "mul" and np.any(vals[:period] <= 0):
        seasonal = "add"

    best_params = None
    best_sse = float("inf")
    for alpha_i in range(10, 91, 20):
        for beta_i in range(1, 52, 25):
            for gamma_i in range(10, 91, 20):
                a, b, g = alpha_i/100.0, beta_i/100.0, gamma_i/100.0
                try:
                    fitted = _hw_fitted(vals, n, period, a, b, g, seasonal)
                    sse = float(np.sum((vals[period:] - np.array(fitted[period:])) ** 2))
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (a, b, g)
                except Exception:
                    continue

    if best_params is None:
        return 0.0, 0.0, [1.0 if seasonal == "mul" else 0.0]*period, period, seasonal

    a, b, g = best_params
    level = np.mean(vals[:period])
    trend = (np.mean(vals[period//2:period]) - np.mean(vals[:period//2])) / (period//2) if period >= 4 else 0.0
    if seasonal == "mul":
        season = np.array([vals[i] / max(level, 1e-10) for i in range(period)])
    else:
        season = np.array([vals[i] - level for i in range(period)])

    for t in range(period, n):
        si = t % period
        if seasonal == "mul":
            new_level = a * (vals[t] / max(season[si], 1e-10)) + (1-a) * (level + trend)
            new_trend = b * (new_level - level) + (1-b) * trend
            season[si] = g * (vals[t] / max(new_level, 1e-10)) + (1-g) * season[si]
        else:
            new_level = a * (vals[t] - season[si]) + (1-a) * (level + trend)
            new_trend = b * (new_level - level) + (1-b) * trend
            season[si] = g * (vals[t] - new_level) + (1-g) * season[si]
        level = new_level
        trend = new_trend

    return float(level), float(trend), [float(s) for s in season], period, seasonal


def seasonal_decomposition_components(series):
    """Return (intercept, slope, seasonal_factors, period) for Excel helper columns."""
    vals, ok = _safe_series(series)
    if not ok:
        return 0.0, 0.0, [0.0]*12, 12
    n = len(vals)
    period = min(12, n)
    if n < 4:
        return 0.0, 0.0, [0.0]*period, period

    if period >= 3:
        half = period // 2
        trend_arr = np.full(n, np.nan)
        for i in range(half, n - half):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            trend_arr[i] = np.mean(vals[start:end])
        mask = ~np.isnan(trend_arr)
        if np.sum(mask) < 2:
            t = np.arange(n, dtype=float)
            t_mean, v_mean = np.mean(t), np.mean(vals)
            slope = np.sum((t - t_mean) * (vals - v_mean)) / max(np.sum((t - t_mean)**2), 1e-10)
            intercept = v_mean - slope * t_mean
        else:
            t_idx = np.arange(n, dtype=float)
            t_valid = t_idx[mask]
            trend_valid = trend_arr[mask]
            t_mean = np.mean(t_valid)
            tr_mean = np.mean(trend_valid)
            slope = np.sum((t_valid - t_mean) * (trend_valid - tr_mean)) / max(np.sum((t_valid - t_mean)**2), 1e-10)
            intercept = tr_mean - slope * t_mean
    else:
        t = np.arange(n, dtype=float)
        t_mean, v_mean = np.mean(t), np.mean(vals)
        slope = np.sum((t - t_mean) * (vals - v_mean)) / max(np.sum((t - t_mean)**2), 1e-10)
        intercept = v_mean - slope * t_mean

    full_trend = intercept + slope * np.arange(n, dtype=float)
    detrended = vals - full_trend
    seasonal = np.zeros(period)
    for i in range(period):
        month_vals = [detrended[j] for j in range(i, n, period)]
        seasonal[i] = np.mean(month_vals) if month_vals else 0.0

    return float(intercept), float(slope), [float(s) for s in seasonal], period


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
    return [max(0.0, float(v)) for v in np.median(arr, axis=0)]


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
**Concept:** A weighted average where the weight (alpha) decays exponentially for older observations. Alpha is optimized by grid search minimizing sum of squared errors on historical data.

**Formula:** `F(t) = α * A(t-1) + (1-α) * F(t-1)`, where α is optimized (0.05 to 0.95).

**When it works well:** Good for items with no trend or seasonality — produces a "best level estimate."

**Limitation:** Single exponential smoothing produces a flat forecast (same value for all 9 months). Does not capture trend or seasonality.
""",

    "Holt-Winters (Additive)": """
**Concept:** Extends exponential smoothing with trend and additive seasonal components. Three equations update level, trend, and seasonal factors. Parameters (α, β, γ) optimized via grid search.

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

**When it works well:** Best for FMCG data — when seasonal peaks scale proportionally with the level (e.g., September is always ~2x average). This is common in food products like Farmley's.

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
**Concept:** Separates the time series into trend, seasonal, and residual components using centered moving averages. Trend is extrapolated forward using linear regression, and the seasonal pattern repeats.

**Formula:**
1. Extract trend via centered moving average (window = seasonal period)
2. Detrended = Actual - Trend
3. Seasonal = average of detrended values for each month position
4. Forecast: `F(t) = Extrapolated_Trend(t) + Seasonal(t mod period)`

**When it works well:** Good for noisy data where you want to separate signal from noise.

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
