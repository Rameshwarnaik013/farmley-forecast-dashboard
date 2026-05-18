import os
import sys
import io
import json
import re
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from flask import Flask, request, render_template_string, send_file, session, redirect, url_for

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecasting import (
    moving_average_forecast,
    weighted_moving_average_forecast,
    exponential_smoothing_forecast,
    holt_winters_forecast,
    linear_trend_forecast,
    seasonal_naive_forecast,
    seasonal_decomposition_forecast,
    ensemble_forecast,
    METHOD_DESCRIPTIONS,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "farmley-forecast-2024")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

FORECAST_METHODS = {
    "Seasonal Naive": seasonal_naive_forecast,
    "Moving Average (3M)": lambda s: moving_average_forecast(s, window=3),
    "Moving Average (6M)": lambda s: moving_average_forecast(s, window=6),
    "Weighted Moving Average": weighted_moving_average_forecast,
    "Exponential Smoothing": exponential_smoothing_forecast,
    "Holt-Winters (Additive)": lambda s: holt_winters_forecast(s, seasonal="add"),
    "Holt-Winters (Multiplicative)": lambda s: holt_winters_forecast(s, seasonal="mul"),
    "Linear Trend + Seasonality": linear_trend_forecast,
    "Seasonal Decomposition": seasonal_decomposition_forecast,
    "Ensemble (Best-of-3)": ensemble_forecast,
}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_cached_data = {}


def parse_month_col(col_name):
    m = re.match(r"([A-Za-z]{3})-(\d{2,4})", str(col_name))
    if not m:
        return None
    abbr, yr = m.group(1), int(m.group(2))
    if yr < 100:
        yr += 2000
    try:
        mi = MONTH_ABBR.index(abbr.capitalize())
    except ValueError:
        return None
    return (yr, mi)


def compute_forecast_months(month_cols, n=9):
    parsed = parse_month_col(month_cols[-1])
    if not parsed:
        yr, mi = 2026, 3
    else:
        yr, mi = parsed
    result = []
    for _ in range(n):
        mi += 1
        if mi > 11:
            mi = 0
            yr += 1
        result.append(f"{MONTH_ABBR[mi]}-{yr}")
    return result


def backtest_method(hist_values, method_fn, n_test=3):
    if len(hist_values) < n_test + 3:
        return None, None
    train = hist_values[:-n_test]
    test = hist_values[-n_test:]
    train_s = pd.Series(train, index=pd.date_range("2020-01-01", periods=len(train), freq="MS"))
    try:
        pred = np.array([float(v) for v in method_fn(train_s)[:n_test]])
        mae = float(np.mean(np.abs(pred - test)))
        safe = np.where(test == 0, 1, test)
        mape = float(np.mean(np.abs((pred - test) / safe))) * 100
        return round(mae, 2), round(mape, 1)
    except Exception:
        return None, None


def build_chart_json(item_name, metric, hist_values, forecasts_dict, month_cols, forecast_months):
    fig = go.Figure()
    x_actual = list(month_cols)
    fig.add_trace(go.Scatter(
        x=x_actual, y=[float(v) for v in hist_values],
        mode="lines+markers", name="Actual",
        line=dict(color="#6366f1", width=3), marker=dict(size=7, symbol="circle"),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
    ))
    colors = ["#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899",
              "#06b6d4", "#84cc16", "#f97316", "#14b8a6", "#a855f7"]
    all_x = x_actual + forecast_months
    for i, (method, vals) in enumerate(forecasts_dict.items()):
        all_y = [float(v) for v in hist_values] + [float(v) for v in vals]
        fig.add_trace(go.Scatter(
            x=all_x, y=all_y, mode="lines+markers", name=method,
            line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
            marker=dict(size=5),
        ))
    last_idx = len(x_actual) - 1
    fig.add_shape(type="line", x0=last_idx, x1=last_idx, y0=0, y1=1, yref="paper",
                  line=dict(color="#cbd5e1", width=2, dash="dot"))
    fig.add_annotation(x=last_idx, y=1.05, yref="paper", text="Actual | Forecast",
                       showarrow=False, font=dict(size=10, color="#94a3b8", family="Inter"))
    fig.update_layout(
        xaxis_title="Month", yaxis_title=metric,
        hovermode="x unified", height=440, template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, font=dict(size=11)),
        margin=dict(b=90, t=20, l=60, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_excel_output(df, month_cols, forecast_months, selected_methods):
    wb = openpyxl.Workbook()
    hfill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    hfont = Font(color="FFFFFF", bold=True, size=11)
    ffill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    afill = PatternFill(start_color="D6EAF8", end_color="D6EAF8", fill_type="solid")
    valfill = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
    border = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"), bottom=Side(style="thin"))
    bold = Font(bold=True)

    n_hist = len(month_cols)
    fc_start_col = 4 + n_hist  # 1-indexed column where forecasts begin
    first_data_col = 4  # column D = first month data

    for method_name in selected_methods:
        method_fn = FORECAST_METHODS[method_name]
        ws = wb.create_sheet(title=method_name[:31])

        # Headers
        headers = ["Item_Name", "Group", "Metric"] + list(month_cols) + forecast_months + ["MAE", "MAPE (%)"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.fill = hfill
            cell.font = hfont
            cell.alignment = Alignment(horizontal="center")
            cell.border = border

        # Data rows
        for r_idx, (_, row) in enumerate(df.iterrows(), 2):
            ws.cell(row=r_idx, column=1, value=row["Item_Name"]).border = border
            ws.cell(row=r_idx, column=2, value=row.get("New MIS ITEM Group", "")).border = border
            ws.cell(row=r_idx, column=3, value=row["Metric"]).border = border

            hist = []
            for ci, m in enumerate(month_cols, first_data_col):
                v = float(row[m])
                cell = ws.cell(row=r_idx, column=ci, value=v)
                cell.fill = afill
                cell.border = border
                cell.number_format = '#,##0.00'
                hist.append(v)

            # Compute forecast values (used as fallback and for methods without simple formulas)
            s = pd.Series(hist, index=pd.date_range("2020-01-01", periods=len(hist), freq="MS"))
            try:
                fc_values = [float(v) for v in method_fn(s)[:9]]
            except Exception:
                fc_values = [0.0] * 9

            # Write forecast cells with FORMULAS where possible
            _write_forecast_formulas(ws, r_idx, method_name, n_hist, first_data_col,
                                     fc_start_col, fc_values, hist, ffill, border)

            # MAE formula: =AVERAGE(ABS(actual - predicted)) for last 3 actuals vs seasonal naive backtest
            mae_col = fc_start_col + len(forecast_months)
            mape_col = mae_col + 1
            mae, mape = backtest_method(np.array(hist), method_fn)
            if mae is not None:
                cell = ws.cell(row=r_idx, column=mae_col, value=mae)
                cell.border = border
                cell.number_format = '#,##0.00'
                cell.fill = valfill
            if mape is not None:
                cell = ws.cell(row=r_idx, column=mape_col, value=mape)
                cell.border = border
                cell.number_format = '#,##0.0'
                cell.fill = valfill

        for c in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 18
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

        # Add formula legend row at bottom
        legend_row = len(df) + 3
        ws.cell(row=legend_row, column=1, value="FORMULA LEGEND:").font = Font(bold=True, size=11, color="1F4E79")
        ws.cell(row=legend_row + 1, column=1, value="Blue cells").fill = afill
        ws.cell(row=legend_row + 1, column=2, value="= Actual historical data (input)")
        ws.cell(row=legend_row + 2, column=1, value="Yellow cells").fill = ffill
        ws.cell(row=legend_row + 2, column=2, value="= Forecasted values (computed by formulas)")
        ws.cell(row=legend_row + 3, column=1, value="Green cells").fill = valfill
        ws.cell(row=legend_row + 3, column=2, value="= Accuracy metrics (MAE / MAPE)")
        ws.cell(row=legend_row + 5, column=1, value="METHOD:").font = bold
        ws.cell(row=legend_row + 5, column=2, value=method_name).font = bold
        formula_explanation = _get_method_formula_text(method_name, n_hist, first_data_col)
        ws.cell(row=legend_row + 6, column=1, value="Formula logic:")
        ws.cell(row=legend_row + 6, column=2, value=formula_explanation)
        ws.row_dimensions[legend_row + 6].height = 80

    # Method Explanations sheet
    expl = wb.create_sheet(title="Method Explanations")
    expl.column_dimensions["A"].width = 30
    expl.column_dimensions["B"].width = 100
    expl.cell(row=1, column=1, value="Method").font = Font(bold=True, size=12)
    expl.cell(row=1, column=2, value="Explanation & Formula").font = Font(bold=True, size=12)
    for i, (mn, md) in enumerate(METHOD_DESCRIPTIONS.items(), 2):
        expl.cell(row=i, column=1, value=mn).font = Font(bold=True)
        expl.cell(row=i, column=2, value=md.strip())
        expl.row_dimensions[i].height = 80

    # Summary sheet with all methods side by side for first 20 items
    summ = wb.create_sheet("Forecast Summary", 0)
    summ_headers = ["Item_Name", "Group", "Metric"] + forecast_months
    for c, h in enumerate(summ_headers, 1):
        cell = summ.cell(row=1, column=c, value=h)
        cell.fill = hfill
        cell.font = hfont
        cell.alignment = Alignment(horizontal="center")
        cell.border = border

    summ_row = 2
    for method_name in selected_methods:
        method_fn = FORECAST_METHODS[method_name]
        summ.cell(row=summ_row, column=1, value=f"── {method_name} ──").font = Font(bold=True, size=11, color="1F4E79")
        summ.merge_cells(start_row=summ_row, start_column=1, end_row=summ_row, end_column=len(summ_headers))
        summ_row += 1
        for _, row in df.iterrows():
            hist = [float(row[m]) for m in month_cols]
            s = pd.Series(hist, index=pd.date_range("2020-01-01", periods=len(hist), freq="MS"))
            try:
                fc = [float(v) for v in method_fn(s)[:9]]
            except Exception:
                fc = [0.0] * 9
            summ.cell(row=summ_row, column=1, value=row["Item_Name"]).border = border
            summ.cell(row=summ_row, column=2, value=row.get("New MIS ITEM Group", "")).border = border
            summ.cell(row=summ_row, column=3, value=row["Metric"]).border = border
            for fi, fv in enumerate(fc):
                cell = summ.cell(row=summ_row, column=4 + fi, value=round(fv, 2))
                cell.fill = ffill
                cell.border = border
                cell.number_format = '#,##0.00'
            summ_row += 1
        summ_row += 1

    for c in range(1, len(summ_headers) + 1):
        summ.column_dimensions[get_column_letter(c)].width = 18
    summ.auto_filter.ref = f"A1:{get_column_letter(len(summ_headers))}1"

    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _write_forecast_formulas(ws, r, method_name, n_hist, data_col, fc_col, fc_values, hist, ffill, border):
    """Write Excel formulas into forecast cells based on method type."""

    last_data_letter = get_column_letter(data_col + n_hist - 1)  # last actual data column letter
    first_data_letter = get_column_letter(data_col)  # D

    if "Moving Average (3M)" in method_name:
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            if fi == 0:
                c1 = get_column_letter(data_col + n_hist - 3)
                c2 = get_column_letter(data_col + n_hist - 2)
                c3 = get_column_letter(data_col + n_hist - 1)
                cell.value = f"=AVERAGE({c1}{r},{c2}{r},{c3}{r})"
            else:
                c1 = get_column_letter(col - 3)
                c2 = get_column_letter(col - 2)
                c3 = get_column_letter(col - 1)
                cell.value = f"=AVERAGE({c1}{r},{c2}{r},{c3}{r})"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Moving Average (6M)" in method_name:
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            if fi == 0:
                c_start = get_column_letter(data_col + n_hist - 6)
                c_end = get_column_letter(data_col + n_hist - 1)
                cell.value = f"=AVERAGE({c_start}{r}:{c_end}{r})"
            else:
                c_start = get_column_letter(col - 6)
                c_end = get_column_letter(col - 1)
                cell.value = f"=AVERAGE({c_start}{r}:{c_end}{r})"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Weighted Moving Average" in method_name:
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            if fi == 0:
                c1 = get_column_letter(data_col + n_hist - 5)
                c2 = get_column_letter(data_col + n_hist - 4)
                c3 = get_column_letter(data_col + n_hist - 3)
                c4 = get_column_letter(data_col + n_hist - 2)
                c5 = get_column_letter(data_col + n_hist - 1)
            else:
                c1 = get_column_letter(col - 5)
                c2 = get_column_letter(col - 4)
                c3 = get_column_letter(col - 3)
                c4 = get_column_letter(col - 2)
                c5 = get_column_letter(col - 1)
            cell.value = f"={c1}{r}*0.10+{c2}{r}*0.15+{c3}{r}*0.20+{c4}{r}*0.25+{c5}{r}*0.30"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Seasonal Naive" in method_name:
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            src_col = data_col + (fi % n_hist)
            src_letter = get_column_letter(src_col)
            cell.value = f"={src_letter}{r}"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Exponential Smoothing" in method_name:
        # Write alpha in a helper area, then use formula
        # For simplicity: first forecast cell has the formula, rest reference it
        alpha = _find_best_alpha(hist)
        alpha_col = fc_col + 10  # put alpha value after MAPE column
        ws.cell(row=r, column=alpha_col, value=round(alpha, 3))
        ws.cell(row=1, column=alpha_col, value="Alpha").font = Font(bold=True, size=9, color="888888")
        alpha_letter = get_column_letter(alpha_col)
        last_actual = get_column_letter(data_col + n_hist - 1)
        avg_letter_start = get_column_letter(data_col)
        avg_letter_end = get_column_letter(data_col + n_hist - 1)
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            # SES: all forecast values are the same = alpha*last + (1-alpha)*avg_of_all
            cell.value = f"={alpha_letter}{r}*{last_actual}{r}+(1-{alpha_letter}{r})*AVERAGE({avg_letter_start}{r}:{avg_letter_end}{r})"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Linear Trend" in method_name:
        # Use TREND + seasonal index approach
        # Put seasonal indices in helper columns
        si_start_col = fc_col + 11
        indices = _compute_seasonal_indices(hist, n_hist)
        for si_i, si_v in enumerate(indices):
            ws.cell(row=r, column=si_start_col + si_i, value=round(si_v, 4))
        if r == 2:
            for si_i in range(n_hist):
                ws.cell(row=1, column=si_start_col + si_i, value=f"SI_{si_i+1}").font = Font(size=8, color="888888")

        seq_start = get_column_letter(data_col)
        seq_end = get_column_letter(data_col + n_hist - 1)
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            trend_x = n_hist + fi + 1
            si_col_letter = get_column_letter(si_start_col + ((n_hist + fi) % n_hist))
            # TREND extrapolates, then multiply by seasonal index
            cell.value = f"=TREND({seq_start}{r}:{seq_end}{r},ROW(INDIRECT(\"1:{n_hist}\")),{trend_x})*{si_col_letter}{r}"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    else:
        # Holt-Winters, Seasonal Decomposition, Ensemble — complex methods
        # Write computed values + formula-style comment showing the logic
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col, value=round(fc_values[fi], 2))
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'
            if "Holt-Winters" in method_name:
                cell.comment = openpyxl.comments.Comment(
                    f"Holt-Winters {method_name.split('(')[1].rstrip(')')}: "
                    f"F = (Level + {fi+1}*Trend) {'*' if 'Mult' in method_name else '+'} Seasonal[month]. "
                    f"Parameters (α,β,γ) optimized via grid search on historical data.",
                    "Forecast Engine")
            elif "Decomposition" in method_name:
                cell.comment = openpyxl.comments.Comment(
                    f"STL Decomposition: Trend extracted via centered MA, "
                    f"extrapolated linearly. Seasonal pattern from detrended values repeats.",
                    "Forecast Engine")
            elif "Ensemble" in method_name:
                cell.comment = openpyxl.comments.Comment(
                    f"Ensemble = MEDIAN(Seasonal Naive, Holt-Winters Mul, Linear Trend+Seasonality). "
                    f"Takes middle value of 3 methods for robustness.",
                    "Forecast Engine")


def _find_best_alpha(hist):
    """Grid search for best exponential smoothing alpha."""
    n = len(hist)
    vals = np.array(hist, dtype=float)
    best_a, best_sse = 0.3, float("inf")
    for a_int in range(5, 96, 5):
        a = a_int / 100.0
        f = [vals[0]]
        for t in range(1, n):
            f.append(a * vals[t-1] + (1-a) * f[-1])
        sse = float(np.sum((vals[1:] - np.array(f[1:])) ** 2))
        if sse < best_sse:
            best_sse, best_a = sse, a
    return best_a


def _compute_seasonal_indices(hist, period):
    """Compute multiplicative seasonal indices for Linear Trend method."""
    vals = np.array(hist, dtype=float)
    n = len(vals)
    t = np.arange(n, dtype=float)
    t_mean, v_mean = np.mean(t), np.mean(vals)
    denom = np.sum((t - t_mean) ** 2)
    slope = np.sum((t - t_mean) * (vals - v_mean)) / max(denom, 1e-10)
    intercept = v_mean - slope * t_mean
    trend = intercept + slope * t
    indices = np.ones(period)
    if np.all(np.abs(trend) > 1e-10):
        ratios = vals / trend
        for i in range(period):
            month_vals = [ratios[j] for j in range(i, n, period)]
            indices[i] = np.mean(month_vals) if month_vals else 1.0
    return indices.tolist()


def _get_method_formula_text(method_name, n_hist, data_col):
    """Return a human-readable formula explanation for the legend."""
    d = get_column_letter(data_col)
    last = get_column_letter(data_col + n_hist - 1)
    if "Moving Average (3M)" in method_name:
        return f"Each forecast = AVERAGE of previous 3 cells. First forecast: =AVERAGE(last 3 actuals). Subsequent: rolling window shifts right."
    elif "Moving Average (6M)" in method_name:
        return f"Each forecast = AVERAGE of previous 6 cells. First forecast: =AVERAGE(last 6 actuals). Subsequent: rolling window shifts right."
    elif "Weighted Moving Average" in method_name:
        return f"WMA = 0.10*fifth_last + 0.15*fourth_last + 0.20*third_last + 0.25*second_last + 0.30*last. Weights sum to 1.0, most recent gets highest weight."
    elif "Seasonal Naive" in method_name:
        return f"Each forecast month = same month from previous year. Apr forecast = Apr actual ({d}), May forecast = May actual, etc."
    elif "Exponential" in method_name:
        return f"F = alpha * last_actual + (1-alpha) * avg(all_actuals). Alpha optimized by minimizing SSE. Stored in Alpha column."
    elif "Linear Trend" in method_name:
        return f"F = TREND(actuals, 1..{n_hist}, future_t) * Seasonal_Index[month]. TREND extrapolates linear fit. SI columns hold multiplicative seasonal indices."
    elif "Holt-Winters" in method_name:
        return f"F = (Level + h*Trend) * Seasonal[month]. Level/Trend/Seasonal updated iteratively. Params (α,β,γ) optimized via grid search. Values computed by Python engine."
    elif "Decomposition" in method_name:
        return f"Trend extracted via centered moving average, extrapolated linearly. Seasonal = avg detrended values per month position. F = Trend + Seasonal."
    elif "Ensemble" in method_name:
        return f"F = MEDIAN(Seasonal_Naive, Holt-Winters_Mul, Linear_Trend). Middle value of 3 methods. Most robust general-purpose forecast."
    return f"{method_name}: computed values from Python forecasting engine."


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Farmley Sales Forecast Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --primary:#0f172a;--primary-600:#1e293b;--primary-500:#334155;
  --accent:#6366f1;--accent-hover:#4f46e5;--accent-light:#eef2ff;--accent-50:#f5f3ff;
  --green:#10b981;--green-bg:#ecfdf5;--green-text:#065f46;
  --amber:#f59e0b;--amber-bg:#fffbeb;--amber-text:#92400e;
  --red:#ef4444;--red-bg:#fef2f2;--red-text:#991b1b;
  --gray-50:#f8fafc;--gray-100:#f1f5f9;--gray-200:#e2e8f0;--gray-300:#cbd5e1;
  --gray-400:#94a3b8;--gray-500:#64748b;--gray-600:#475569;--gray-700:#334155;
  --gray-800:#1e293b;--gray-900:#0f172a;
  --radius:12px;--radius-sm:8px;--radius-xs:6px;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --shadow-md:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -2px rgba(0,0,0,.05);
  --shadow-lg:0 10px 15px -3px rgba(0,0,0,.08),0 4px 6px -4px rgba(0,0,0,.04);
  --shadow-xl:0 20px 25px -5px rgba(0,0,0,.08),0 8px 10px -6px rgba(0,0,0,.04);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--gray-50);color:var(--gray-800);-webkit-font-smoothing:antialiased}

/* HEADER */
.header{background:var(--primary);color:#fff;padding:0;position:relative;overflow:hidden}
.header::before{content:'';position:absolute;top:-50%;right:-10%;width:500px;height:500px;background:radial-gradient(circle,rgba(99,102,241,.15) 0%,transparent 70%);pointer-events:none}
.header::after{content:'';position:absolute;bottom:-30%;left:-5%;width:400px;height:400px;background:radial-gradient(circle,rgba(99,102,241,.1) 0%,transparent 70%);pointer-events:none}
.header-inner{max-width:1440px;margin:0 auto;padding:20px 32px;display:flex;align-items:center;justify-content:space-between;position:relative;z-index:1}
.header-brand{display:flex;align-items:center;gap:14px}
.header-logo{width:44px;height:44px;background:var(--accent);border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;font-size:1.4rem;font-weight:800;color:#fff;letter-spacing:-1px;flex-shrink:0}
.header h1{font-size:1.35rem;font-weight:700;letter-spacing:-.02em}
.header p{font-size:.8rem;color:var(--gray-400);margin-top:2px;font-weight:400}
.header-badge{background:rgba(99,102,241,.2);color:#a5b4fc;padding:5px 12px;border-radius:20px;font-size:.7rem;font-weight:600;letter-spacing:.03em;text-transform:uppercase}

.container{max-width:1440px;margin:0 auto;padding:24px 32px 40px}

/* CARDS */
.card{background:#fff;border-radius:var(--radius);box-shadow:var(--shadow);padding:24px;margin-bottom:20px;border:1px solid var(--gray-200);transition:box-shadow .2s}
.card:hover{box-shadow:var(--shadow-md)}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--gray-100)}
.card-header h2{font-size:1rem;font-weight:700;color:var(--gray-900);display:flex;align-items:center;gap:8px}
.card-header h2 .icon{width:20px;height:20px;display:flex;align-items:center;justify-content:center;background:var(--accent-light);color:var(--accent);border-radius:var(--radius-xs);font-size:.7rem;flex-shrink:0}
.card h2{font-size:1rem;font-weight:700;color:var(--gray-900);margin-bottom:14px}

/* GRID */
.grid{display:grid;gap:24px}
.grid-2{grid-template-columns:300px 1fr}

/* UPLOAD PAGE */
.upload-page{max-width:640px;margin:60px auto}
.upload-page .card{padding:40px;text-align:center}
.upload-icon{width:80px;height:80px;margin:0 auto 20px;background:var(--accent-light);border-radius:50%;display:flex;align-items:center;justify-content:center}
.upload-icon svg{width:36px;height:36px;color:var(--accent)}
.upload-area{border:2px dashed var(--gray-300);border-radius:var(--radius);padding:32px 24px;background:var(--gray-50);transition:all .25s;cursor:pointer;position:relative}
.upload-area:hover,.upload-area.drag-over{border-color:var(--accent);background:var(--accent-light)}
.upload-area h3{font-size:1rem;font-weight:600;color:var(--gray-700);margin-bottom:6px}
.upload-area p{font-size:.85rem;color:var(--gray-400)}
.upload-area input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-area .file-name{display:none;margin-top:10px;font-size:.85rem;color:var(--accent);font-weight:600}

/* BUTTONS */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:10px 20px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius-xs);cursor:pointer;font-size:.85rem;font-weight:600;font-family:inherit;text-decoration:none;transition:all .2s;line-height:1}
.btn:hover{background:var(--accent-hover);transform:translateY(-1px);box-shadow:var(--shadow-md)}
.btn:active{transform:translateY(0)}
.btn-sm{padding:5px 10px;font-size:.75rem;border-radius:var(--radius-xs)}
.btn-outline{background:transparent;color:var(--gray-600);border:1px solid var(--gray-300)}
.btn-outline:hover{background:var(--gray-50);color:var(--gray-800);border-color:var(--gray-400);transform:none;box-shadow:none}
.btn-ghost{background:transparent;color:var(--accent);padding:6px 10px}
.btn-ghost:hover{background:var(--accent-light);transform:none;box-shadow:none}
.btn-download{background:var(--green);width:100%}
.btn-download:hover{background:#059669}
.btn-block{width:100%}

/* CUSTOM MULTI-SELECT DROPDOWN */
.ms-wrap{position:relative;margin-bottom:14px}
.ms-wrap label{display:block;font-size:.8rem;font-weight:600;color:var(--gray-600);margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}
.ms-trigger{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;background:#fff;border:1px solid var(--gray-300);border-radius:var(--radius-xs);cursor:pointer;transition:all .15s;min-height:40px;gap:8px}
.ms-trigger:hover{border-color:var(--gray-400)}
.ms-trigger.open{border-color:var(--accent);box-shadow:0 0 0 3px rgba(99,102,241,.1)}
.ms-trigger-text{font-size:.83rem;color:var(--gray-500);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.ms-trigger-text.has-sel{color:var(--gray-800);font-weight:500}
.ms-arrow{width:16px;height:16px;flex-shrink:0;transition:transform .2s;color:var(--gray-400)}
.ms-trigger.open .ms-arrow{transform:rotate(180deg)}
.ms-pills{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.ms-pill{display:inline-flex;align-items:center;gap:3px;padding:3px 8px;background:var(--accent-light);color:var(--accent);border-radius:20px;font-size:.7rem;font-weight:600;max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ms-pill-x{cursor:pointer;font-size:.85rem;line-height:1;opacity:.6;flex-shrink:0}
.ms-pill-x:hover{opacity:1}
.ms-pill-more{background:var(--gray-100);color:var(--gray-500)}
.ms-dropdown{position:absolute;top:100%;left:0;right:0;z-index:100;background:#fff;border:1px solid var(--gray-200);border-radius:var(--radius-sm);box-shadow:var(--shadow-lg);margin-top:4px;display:none;max-height:280px;overflow:hidden;flex-direction:column}
.ms-dropdown.open{display:flex}
.ms-search-wrap{padding:8px;border-bottom:1px solid var(--gray-100)}
.ms-search{width:100%;padding:7px 10px;border:1px solid var(--gray-200);border-radius:var(--radius-xs);font-size:.8rem;font-family:inherit;outline:none;transition:border .15s}
.ms-search:focus{border-color:var(--accent)}
.ms-actions{display:flex;gap:0;border-bottom:1px solid var(--gray-100)}
.ms-actions button{flex:1;padding:7px;background:none;border:none;font-size:.75rem;font-weight:600;color:var(--gray-500);cursor:pointer;font-family:inherit;transition:all .15s}
.ms-actions button:hover{background:var(--gray-50);color:var(--accent)}
.ms-actions button:first-child{border-right:1px solid var(--gray-100)}
.ms-list{overflow-y:auto;flex:1;max-height:200px;padding:4px 0}
.ms-item{display:flex;align-items:center;gap:8px;padding:7px 12px;cursor:pointer;font-size:.82rem;color:var(--gray-700);transition:background .1s}
.ms-item:hover{background:var(--gray-50)}
.ms-item.hidden{display:none}
.ms-check{width:16px;height:16px;border:2px solid var(--gray-300);border-radius:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;transition:all .15s}
.ms-item.selected .ms-check{background:var(--accent);border-color:var(--accent)}
.ms-item.selected .ms-check::after{content:'';width:5px;height:8px;border:solid #fff;border-width:0 2px 2px 0;transform:rotate(45deg);margin-top:-2px}
.ms-item-text{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* SINGLE SELECT */
.sel-wrap{margin-bottom:14px}
.sel-wrap label{display:block;font-size:.8rem;font-weight:600;color:var(--gray-600);margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}
.sel-wrap select{width:100%;padding:9px 12px;border:1px solid var(--gray-300);border-radius:var(--radius-xs);font-size:.83rem;font-family:inherit;background:#fff;color:var(--gray-800);cursor:pointer;outline:none;transition:border .15s;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%2394a3b8' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center}
.sel-wrap select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(99,102,241,.1)}

/* SIDEBAR */
.sidebar{position:sticky;top:24px;max-height:calc(100vh - 48px);overflow-y:auto;padding-right:4px}
.sidebar::-webkit-scrollbar{width:4px}
.sidebar::-webkit-scrollbar-thumb{background:var(--gray-300);border-radius:4px}
.sidebar .card{padding:20px}
.sidebar-title{font-size:.85rem;font-weight:700;color:var(--gray-900);display:flex;align-items:center;gap:8px;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--gray-100)}
.sidebar-title svg{width:16px;height:16px;color:var(--accent)}

/* METRIC CARDS */
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:0}
.stat-card{background:#fff;border:1px solid var(--gray-200);border-radius:var(--radius-sm);padding:16px;transition:all .2s}
.stat-card:hover{border-color:var(--accent);box-shadow:0 0 0 3px rgba(99,102,241,.06)}
.stat-icon{width:32px;height:32px;border-radius:var(--radius-xs);display:flex;align-items:center;justify-content:center;margin-bottom:8px;font-size:.9rem}
.stat-icon.blue{background:var(--accent-light);color:var(--accent)}
.stat-icon.green{background:var(--green-bg);color:var(--green)}
.stat-icon.amber{background:var(--amber-bg);color:var(--amber)}
.stat-icon.purple{background:#f5f3ff;color:#8b5cf6}
.stat-label{font-size:.7rem;font-weight:600;color:var(--gray-400);text-transform:uppercase;letter-spacing:.05em}
.stat-value{font-size:1.15rem;font-weight:700;color:var(--gray-900);margin-top:2px}

/* TABLES */
table{width:100%;border-collapse:separate;border-spacing:0;font-size:.82rem}
thead th{background:var(--gray-50);color:var(--gray-600);padding:10px 14px;text-align:left;font-weight:600;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid var(--gray-200);position:sticky;top:0;z-index:1}
tbody td{padding:10px 14px;border-bottom:1px solid var(--gray-100);color:var(--gray-700)}
tbody tr:hover{background:var(--gray-50)}
tbody tr:last-child td{border-bottom:none}
.fc-val{background:var(--amber-bg);font-weight:600;color:var(--amber-text);font-variant-numeric:tabular-nums}
.scroll-table{overflow-x:auto;max-height:460px;overflow-y:auto;border:1px solid var(--gray-200);border-radius:var(--radius-sm)}

/* BADGES */
.badge{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:600;letter-spacing:.02em}
.badge-good{background:var(--green-bg);color:var(--green-text)}
.badge-fair{background:var(--amber-bg);color:var(--amber-text)}
.badge-poor{background:var(--red-bg);color:var(--red-text)}

/* ITEM CARD */
.item-card{border-left:3px solid var(--accent)}
.item-header{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.item-header h2{font-size:1.05rem;margin-bottom:0;flex:1}
.item-tag{font-size:.72rem;font-weight:600;background:var(--gray-100);color:var(--gray-500);padding:3px 10px;border-radius:20px}
.section-label{font-size:.82rem;font-weight:700;color:var(--gray-900);margin:20px 0 10px;display:flex;align-items:center;gap:6px}
.section-label svg{width:14px;height:14px;color:var(--accent)}

/* METHOD TABS */
.tabs{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap;padding:4px;background:var(--gray-100);border-radius:var(--radius-sm)}
.tab{padding:8px 16px;cursor:pointer;font-size:.78rem;font-weight:500;color:var(--gray-500);border-radius:var(--radius-xs);transition:all .2s}
.tab:hover{color:var(--gray-700);background:var(--gray-50)}
.tab.active{background:#fff;color:var(--accent);font-weight:600;box-shadow:var(--shadow)}
.tab-content{display:none}
.tab-content.active{display:block}
.method-desc{line-height:1.75;font-size:.88rem;color:var(--gray-600)}
.method-desc strong{color:var(--gray-900)}
.method-desc code{background:var(--accent-light);color:var(--accent);padding:2px 7px;border-radius:4px;font-size:.78rem;font-weight:500}

/* GUIDE BOX */
.guide-box{margin-top:20px;padding:20px;background:var(--gray-50);border-radius:var(--radius-sm);border:1px solid var(--gray-200)}
.guide-box h3{font-size:.9rem;font-weight:700;color:var(--gray-900);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.guide-table td,.guide-table th{padding:8px 14px;font-size:.8rem}
.guide-table th{background:var(--primary);color:#fff;font-weight:600;text-transform:none;letter-spacing:0;font-size:.8rem}
.guide-note{margin-top:14px;font-size:.8rem;color:var(--gray-500);line-height:1.6}
.guide-note strong{color:var(--gray-700)}

/* EMPTY STATE */
.empty-state{text-align:center;padding:60px 20px}
.empty-state svg{width:48px;height:48px;color:var(--gray-300);margin-bottom:12px}
.empty-state p{color:var(--gray-400);font-size:.9rem}
.empty-state strong{color:var(--gray-600)}

/* FOOTER */
footer{text-align:center;padding:32px 20px;color:var(--gray-400);font-size:.75rem;border-top:1px solid var(--gray-200);margin-top:20px}
footer a{color:var(--accent);text-decoration:none}

/* ANIMATIONS */
@keyframes fadeInUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.card{animation:fadeInUp .3s ease both}
.card:nth-child(2){animation-delay:.05s}
.card:nth-child(3){animation-delay:.1s}

/* RESPONSIVE */
@media(max-width:960px){
  .grid-2{grid-template-columns:1fr}
  .sidebar{position:static;max-height:none}
  .header-inner{flex-direction:column;align-items:flex-start;gap:10px}
  .header-badge{align-self:flex-start}
  .container{padding:16px}
  .stats-row{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>

<div class="header">
    <div class="header-inner">
        <div class="header-brand">
            <div class="header-logo">F</div>
            <div>
                <h1>Farmley Forecast</h1>
                <p>Sales forecasting dashboard</p>
            </div>
        </div>
        {% if data_loaded %}
        <div class="header-badge">{{total_items}} Products Loaded</div>
        {% endif %}
    </div>
</div>

<div class="container">

{% if not data_loaded %}
<div class="upload-page">
    <div class="card">
        <div class="upload-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        </div>
        <h2 style="font-size:1.2rem;margin-bottom:4px">Upload Sales Data</h2>
        <p style="color:var(--gray-400);font-size:.88rem;margin-bottom:24px">Upload your Excel file with item-wise sales order data</p>
        <form method="post" enctype="multipart/form-data" action="/upload">
            <div class="upload-area" id="dropZone">
                <h3>Choose file or drag & drop</h3>
                <p>.xlsx or .xls files supported</p>
                <span class="file-name" id="fileName"></span>
                <input type="file" name="file" accept=".xlsx,.xls" required id="fileInput">
            </div>
            <button type="submit" class="btn btn-block" style="margin-top:20px;padding:12px">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Upload & Analyze
            </button>
        </form>
        <p style="margin-top:16px;font-size:.78rem;color:var(--gray-400)">Auto-detects data range and forecasts next 9 months</p>
    </div>
</div>

{% else %}

<form method="post" action="/forecast" id="mainForm">
<div class="grid grid-2">

<!-- SIDEBAR -->
<div class="sidebar">
    <div class="card">
        <div class="sidebar-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
            Filters
        </div>

        <!-- Product Group Multi-Select -->
        <div class="ms-wrap" data-name="groups">
            <label>Product Group</label>
            <div class="ms-trigger" onclick="toggleDropdown(this)">
                <span class="ms-trigger-text">Select groups...</span>
                <svg class="ms-arrow" viewBox="0 0 16 16" fill="currentColor"><path d="M8 11L3 6h10z"/></svg>
            </div>
            <div class="ms-pills" id="pills_groups"></div>
            <div class="ms-dropdown">
                <div class="ms-search-wrap"><input class="ms-search" placeholder="Search groups..." oninput="filterDropdown(this)"></div>
                <div class="ms-actions"><button type="button" onclick="selectAllDropdown(this)">Select All</button><button type="button" onclick="clearAllDropdown(this)">Clear All</button></div>
                <div class="ms-list">
                {% for g in all_groups %}
                    <div class="ms-item {% if g in selected_groups %}selected{% endif %}" data-value="{{g}}" onclick="toggleItem(this)">
                        <div class="ms-check"></div>
                        <span class="ms-item-text">{{g}}</span>
                    </div>
                {% endfor %}
                </div>
            </div>
            {% for g in selected_groups %}<input type="hidden" name="groups" value="{{g}}">{% endfor %}
        </div>

        <!-- Item Name Multi-Select -->
        <div class="ms-wrap" data-name="items">
            <label>Item Name</label>
            <div class="ms-trigger" onclick="toggleDropdown(this)">
                <span class="ms-trigger-text">Select items...</span>
                <svg class="ms-arrow" viewBox="0 0 16 16" fill="currentColor"><path d="M8 11L3 6h10z"/></svg>
            </div>
            <div class="ms-pills" id="pills_items"></div>
            <div class="ms-dropdown">
                <div class="ms-search-wrap"><input class="ms-search" placeholder="Search items..." oninput="filterDropdown(this)"></div>
                <div class="ms-actions"><button type="button" onclick="selectAllDropdown(this)">Select All</button><button type="button" onclick="clearAllDropdown(this)">Clear All</button></div>
                <div class="ms-list">
                {% for it in all_items %}
                    <div class="ms-item {% if it in selected_items %}selected{% endif %}" data-value="{{it}}" onclick="toggleItem(this)">
                        <div class="ms-check"></div>
                        <span class="ms-item-text">{{it}}</span>
                    </div>
                {% endfor %}
                </div>
            </div>
            {% for it in selected_items %}<input type="hidden" name="items" value="{{it}}">{% endfor %}
        </div>

        <!-- Metric -->
        <div class="sel-wrap">
            <label>Metric</label>
            <select name="metric">
            {% for m in all_metrics %}
                <option value="{{m}}" {% if m == selected_metric %}selected{% endif %}>{{m}}</option>
            {% endfor %}
            </select>
        </div>

        <!-- Forecast Methods Multi-Select -->
        <div class="ms-wrap" data-name="methods">
            <label>Forecast Methods</label>
            <div class="ms-trigger" onclick="toggleDropdown(this)">
                <span class="ms-trigger-text">Select methods...</span>
                <svg class="ms-arrow" viewBox="0 0 16 16" fill="currentColor"><path d="M8 11L3 6h10z"/></svg>
            </div>
            <div class="ms-pills" id="pills_methods"></div>
            <div class="ms-dropdown">
                <div class="ms-search-wrap"><input class="ms-search" placeholder="Search methods..." oninput="filterDropdown(this)"></div>
                <div class="ms-actions"><button type="button" onclick="selectAllDropdown(this)">Select All</button><button type="button" onclick="clearAllDropdown(this)">Clear All</button></div>
                <div class="ms-list">
                {% for m in all_methods %}
                    <div class="ms-item {% if m in selected_methods %}selected{% endif %}" data-value="{{m}}" onclick="toggleItem(this)">
                        <div class="ms-check"></div>
                        <span class="ms-item-text">{{m}}</span>
                    </div>
                {% endfor %}
                </div>
            </div>
            {% for m in selected_methods %}<input type="hidden" name="methods" value="{{m}}">{% endfor %}
        </div>

        <button type="submit" class="btn btn-block" style="margin-top:6px">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
            Apply Filters
        </button>
    </div>

    <div class="card" style="margin-top:0">
        <div class="sidebar-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Export
        </div>
        <p style="font-size:.78rem;color:var(--gray-400);margin-bottom:10px">Download Excel with real formulas, MAE/MAPE per method tab.</p>
        <a href="/download?methods={{selected_methods|join(',')}}" class="btn btn-download">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Download Forecast Excel
        </a>
    </div>

    <div class="card" style="margin-top:0">
        <a href="/" class="btn btn-outline btn-block">Upload New File</a>
    </div>
</div>

<!-- MAIN CONTENT -->
<div>
    <div class="card" style="padding:16px 20px">
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-icon blue">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>
                </div>
                <div class="stat-label">Total Items</div>
                <div class="stat-value">{{total_items}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon green">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
                </div>
                <div class="stat-label">Data Through</div>
                <div class="stat-value">{{last_month}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon amber">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                </div>
                <div class="stat-label">Forecast Range</div>
                <div class="stat-value">{{forecast_months[0]}} — {{forecast_months[-1]}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon purple">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>
                </div>
                <div class="stat-label">Months of Data</div>
                <div class="stat-value">{{n_months}}</div>
            </div>
        </div>
    </div>

    {% for item_data in results %}
    <div class="card item-card">
        <div class="item-header">
            <h2 style="margin-bottom:0">{{item_data.item}}</h2>
            <span class="item-tag">{{item_data.group}}</span>
            <span class="item-tag">{{selected_metric}}</span>
        </div>
        <div id="chart_{{loop.index0}}" style="width:100%;height:480px;margin:0 -8px"></div>
        <script>Plotly.newPlot('chart_{{loop.index0}}',{{item_data.chart_data|safe}}.data,{{item_data.chart_data|safe}}.layout,{responsive:true})</script>

        <div class="section-label">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
            Forecast Values
        </div>
        <div class="scroll-table">
        <table>
            <thead><tr><th>Month</th>{% for m in selected_methods %}<th>{{m}}</th>{% endfor %}</tr></thead>
            <tbody>
            {% for i in range(forecast_months|length) %}
            <tr><td><strong>{{forecast_months[i]}}</strong></td>
            {% for m in selected_methods %}
                <td class="fc-val">{{item_data.forecasts.get(m, [0]*9)[i]|round(2)|string|replace('.0','') if item_data.forecasts.get(m) else 'N/A'}}</td>
            {% endfor %}</tr>
            {% endfor %}
            </tbody>
        </table>
        </div>

        {% if item_data.accuracy %}
        <div class="section-label">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            Backtest Accuracy
            <span style="font-weight:400;font-size:.72rem;color:var(--gray-400);margin-left:4px">trained on all-but-last-3, tested on last 3</span>
        </div>
        <div class="scroll-table">
        <table>
            <thead><tr><th>Method</th><th>MAE</th><th>MAPE (%)</th><th>Rating</th></tr></thead>
            <tbody>
            {% for acc in item_data.accuracy %}
            <tr>
                <td style="font-weight:500">{{acc.method}}</td>
                <td style="font-variant-numeric:tabular-nums">{{acc.mae if acc.mae is not none else 'N/A'}}</td>
                <td style="font-variant-numeric:tabular-nums">{{acc.mape if acc.mape is not none else 'N/A'}}</td>
                <td>{% if acc.rating == 'Good' %}<span class="badge badge-good">Good</span>
                    {% elif acc.rating == 'Fair' %}<span class="badge badge-fair">Fair</span>
                    {% elif acc.rating == 'Poor' %}<span class="badge badge-poor">Poor</span>
                    {% else %}<span style="color:var(--gray-400)">—</span>{% endif %}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        </div>
        {% endif %}
    </div>
    {% endfor %}

    {% if not results %}
    <div class="card">
        <div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            <p>Select items and methods from the sidebar, then click <strong>Apply Filters</strong> to generate forecasts.</p>
        </div>
    </div>
    {% endif %}

    <!-- METHOD EXPLANATIONS -->
    <div class="card">
        <div class="card-header">
            <h2><span class="icon">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            </span>Forecasting Methods</h2>
        </div>
        <div class="tabs" id="methodTabs">
        {% for mname in method_descriptions %}
            <div class="tab {% if loop.first %}active{% endif %}" onclick="showTab(this,'mt_{{loop.index0}}')">{{mname}}</div>
        {% endfor %}
        </div>
        {% for mname, mdesc in method_descriptions.items() %}
        <div class="tab-content {% if loop.first %}active{% endif %}" id="mt_{{loop.index0}}">
            <div class="method-desc">{{mdesc|safe}}</div>
        </div>
        {% endfor %}

        <div class="guide-box">
            <h3>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
                Quick Guide — Which Method to Use
            </h3>
            <div class="scroll-table">
            <table class="guide-table">
                <thead><tr><th>Sales Pattern</th><th>Recommended Method</th></tr></thead>
                <tbody>
                <tr><td>Strong seasonal peaks (festive months)</td><td><strong>Holt-Winters (Multiplicative)</strong> or Ensemble</td></tr>
                <tr><td>Steady growth, no seasonality</td><td><strong>Exponential Smoothing</strong> or Moving Average</td></tr>
                <tr><td>Stable with mild seasonality</td><td><strong>Linear Trend + Seasonality</strong></td></tr>
                <tr><td>New product (few months of data)</td><td><strong>Moving Average (3M)</strong></td></tr>
                <tr><td>Unsure / mixed pattern</td><td><strong>Ensemble (Best-of-3)</strong> — safest default</td></tr>
                </tbody>
            </table>
            </div>
            <div class="guide-note">
                <strong>MAE</strong> (Mean Absolute Error): average absolute difference between predicted and actual. Lower is better. Same units as your data.<br>
                <strong>MAPE</strong> (Mean Absolute % Error): average percentage error. &lt;30% = Good, 30–60% = Fair, &gt;60% = Poor.
            </div>
        </div>
    </div>

</div>
</div>
</form>

{% endif %}
</div>

<footer>
    Farmley Sales Forecast Dashboard v4.0 &bull; Powered by Flask + Plotly &bull; <a href="https://github.com/Rameshwarnaik013/farmley-forecast-dashboard" target="_blank">GitHub</a>
</footer>

<script>
/* --- CUSTOM MULTI-SELECT --- */
function toggleDropdown(trigger){
    var wrap=trigger.closest('.ms-wrap');
    var dd=wrap.querySelector('.ms-dropdown');
    var isOpen=dd.classList.contains('open');
    document.querySelectorAll('.ms-dropdown.open').forEach(function(d){d.classList.remove('open');d.closest('.ms-wrap').querySelector('.ms-trigger').classList.remove('open')});
    if(!isOpen){dd.classList.add('open');trigger.classList.add('open');var si=dd.querySelector('.ms-search');if(si)si.focus()}
}
function toggleItem(el){
    el.classList.toggle('selected');
    syncHiddenInputs(el.closest('.ms-wrap'));
    updateTriggerText(el.closest('.ms-wrap'));
    updatePills(el.closest('.ms-wrap'));
}
function selectAllDropdown(btn){
    var wrap=btn.closest('.ms-wrap');
    wrap.querySelectorAll('.ms-item:not(.hidden)').forEach(function(i){i.classList.add('selected')});
    syncHiddenInputs(wrap);updateTriggerText(wrap);updatePills(wrap);
}
function clearAllDropdown(btn){
    var wrap=btn.closest('.ms-wrap');
    wrap.querySelectorAll('.ms-item').forEach(function(i){i.classList.remove('selected')});
    syncHiddenInputs(wrap);updateTriggerText(wrap);updatePills(wrap);
}
function filterDropdown(input){
    var q=input.value.toLowerCase();
    var list=input.closest('.ms-dropdown').querySelector('.ms-list');
    list.querySelectorAll('.ms-item').forEach(function(i){
        var txt=i.querySelector('.ms-item-text').textContent.toLowerCase();
        if(txt.indexOf(q)>-1)i.classList.remove('hidden');else i.classList.add('hidden');
    });
}
function syncHiddenInputs(wrap){
    var name=wrap.getAttribute('data-name');
    wrap.querySelectorAll('input[type=hidden]').forEach(function(h){h.remove()});
    wrap.querySelectorAll('.ms-item.selected').forEach(function(i){
        var inp=document.createElement('input');inp.type='hidden';inp.name=name;inp.value=i.getAttribute('data-value');
        wrap.appendChild(inp);
    });
}
function updateTriggerText(wrap){
    var sel=wrap.querySelectorAll('.ms-item.selected');
    var txt=wrap.querySelector('.ms-trigger-text');
    if(sel.length===0){txt.textContent='None selected';txt.classList.remove('has-sel')}
    else if(sel.length===1){txt.textContent=sel[0].getAttribute('data-value');txt.classList.add('has-sel')}
    else{txt.textContent=sel.length+' selected';txt.classList.add('has-sel')}
}
function updatePills(wrap){
    var name=wrap.getAttribute('data-name');
    var pills=document.getElementById('pills_'+name);
    if(!pills)return;
    pills.innerHTML='';
    var sel=wrap.querySelectorAll('.ms-item.selected');
    var max=3;
    for(var i=0;i<Math.min(sel.length,max);i++){
        var v=sel[i].getAttribute('data-value');
        var p=document.createElement('span');p.className='ms-pill';
        p.innerHTML='<span>'+v+'</span><span class="ms-pill-x" onclick="removePill(this,\''+v.replace(/'/g,"\\'")+'\')">&#215;</span>';
        pills.appendChild(p);
    }
    if(sel.length>max){
        var more=document.createElement('span');more.className='ms-pill ms-pill-more';
        more.textContent='+'+(sel.length-max)+' more';
        pills.appendChild(more);
    }
}
function removePill(x,val){
    var wrap=x.closest('.ms-wrap');
    wrap.querySelectorAll('.ms-item').forEach(function(i){
        if(i.getAttribute('data-value')===val)i.classList.remove('selected');
    });
    syncHiddenInputs(wrap);updateTriggerText(wrap);updatePills(wrap);
}

/* Close dropdowns on outside click */
document.addEventListener('click',function(e){
    if(!e.target.closest('.ms-wrap')){
        document.querySelectorAll('.ms-dropdown.open').forEach(function(d){d.classList.remove('open');d.closest('.ms-wrap').querySelector('.ms-trigger').classList.remove('open')});
    }
});

/* Init multi-selects */
document.querySelectorAll('.ms-wrap').forEach(function(w){updateTriggerText(w);updatePills(w)});

/* --- TABS --- */
function showTab(el,id){
    el.parentElement.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active')});
    el.classList.add('active');
    var card=el.closest('.card');
    card.querySelectorAll('.tab-content').forEach(function(c){c.classList.remove('active')});
    card.querySelector('#'+id).classList.add('active');
}

/* --- UPLOAD DRAG & DROP --- */
var dz=document.getElementById('dropZone');
if(dz){
    var fi=document.getElementById('fileInput');
    var fn=document.getElementById('fileName');
    ['dragenter','dragover'].forEach(function(e){dz.addEventListener(e,function(ev){ev.preventDefault();dz.classList.add('drag-over')})});
    ['dragleave','drop'].forEach(function(e){dz.addEventListener(e,function(ev){ev.preventDefault();dz.classList.remove('drag-over')})});
    dz.addEventListener('drop',function(e){if(e.dataTransfer.files.length){fi.files=e.dataTransfer.files;fn.style.display='block';fn.textContent=fi.files[0].name}});
    fi.addEventListener('change',function(){if(fi.files.length){fn.style.display='block';fn.textContent=fi.files[0].name}});
}
</script>
</body>
</html>"""


def md_to_html(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = text.replace('\n\n', '<br><br>').replace('\n-', '<br>•')
    return text


@app.route("/")
def index():
    if "df_json" not in _cached_data:
        return render_template_string(TEMPLATE, data_loaded=False)
    return redirect("/forecast")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return redirect("/")
    df = pd.read_excel(f)
    meta = {"Item_Name", "New MIS ITEM Group", "Metric"}
    month_cols = [c for c in df.columns if c not in meta]
    df[month_cols] = df[month_cols].fillna(0)
    for c in month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    _cached_data["df_json"] = df.to_json()
    _cached_data["month_cols"] = month_cols
    return redirect("/forecast")


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    if "df_json" not in _cached_data:
        return redirect("/")

    df = pd.read_json(io.StringIO(_cached_data["df_json"]))
    month_cols = _cached_data["month_cols"]
    forecast_months = compute_forecast_months(month_cols)

    all_groups = sorted([g for g in df["New MIS ITEM Group"].dropna().unique()])
    all_items = sorted(df["Item_Name"].dropna().unique().tolist())
    all_metrics = sorted(df["Metric"].dropna().unique().tolist())
    all_methods = list(FORECAST_METHODS.keys())

    if request.method == "POST":
        selected_groups = request.form.getlist("groups")
        selected_items = request.form.getlist("items")
        selected_metric = request.form.get("metric", all_metrics[0] if all_metrics else "")
        selected_methods = request.form.getlist("methods")
    else:
        selected_groups = all_groups
        selected_items = all_items[:1]
        selected_metric = all_metrics[0] if all_metrics else ""
        selected_methods = ["Seasonal Naive", "Holt-Winters (Multiplicative)",
                           "Linear Trend + Seasonality", "Ensemble (Best-of-3)"]

    if not selected_items:
        selected_items = all_items[:1]
    if not selected_methods:
        selected_methods = ["Ensemble (Best-of-3)"]

    results = []
    for item_name in selected_items[:10]:
        row = df[(df["Item_Name"] == item_name) & (df["Metric"] == selected_metric)]
        if row.empty:
            continue
        hist = row.iloc[0][month_cols].values.astype(float)
        series = pd.Series(hist, index=pd.date_range("2020-01-01", periods=len(month_cols), freq="MS"))
        group = row.iloc[0].get("New MIS ITEM Group", "—")

        forecasts = {}
        accuracy = []
        for m in selected_methods:
            fn = FORECAST_METHODS[m]
            try:
                fc = [float(v) for v in fn(series)[:9]]
                forecasts[m] = fc
            except Exception:
                forecasts[m] = [0.0] * 9
            mae, mape = backtest_method(hist, fn)
            rating = "—"
            if mape is not None:
                rating = "Good" if mape < 30 else ("Fair" if mape < 60 else "Poor")
            accuracy.append({"method": m, "mae": mae, "mape": mape, "rating": rating})

        chart_json = build_chart_json(item_name, selected_metric, hist, forecasts, month_cols, forecast_months)
        results.append({
            "item": item_name, "group": group,
            "chart_data": chart_json,
            "forecasts": forecasts,
            "accuracy": accuracy,
        })

    desc_html = {k: md_to_html(v) for k, v in METHOD_DESCRIPTIONS.items()}

    return render_template_string(TEMPLATE,
        data_loaded=True,
        all_groups=all_groups, selected_groups=selected_groups,
        all_items=all_items, selected_items=selected_items,
        all_metrics=all_metrics, selected_metric=selected_metric,
        all_methods=all_methods, selected_methods=selected_methods,
        forecast_months=forecast_months, results=results,
        total_items=len(all_items), last_month=month_cols[-1],
        n_months=len(month_cols),
        method_descriptions=desc_html,
    )


@app.route("/download")
def download():
    if "df_json" not in _cached_data:
        return redirect("/")
    df = pd.read_json(io.StringIO(_cached_data["df_json"]))
    month_cols = _cached_data["month_cols"]
    forecast_months = compute_forecast_months(month_cols)
    methods_str = request.args.get("methods", "Ensemble (Best-of-3)")
    selected_methods = [m.strip() for m in methods_str.split(",") if m.strip() in FORECAST_METHODS]
    if not selected_methods:
        selected_methods = ["Ensemble (Best-of-3)"]
    buf = create_excel_output(df, month_cols, forecast_months, selected_methods)
    return send_file(buf, download_name="Farmley_Forecast_Output.xlsx",
                     as_attachment=True, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
