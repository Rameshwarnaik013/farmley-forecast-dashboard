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
    holt_winters_components,
    linear_trend_forecast,
    seasonal_naive_forecast,
    seasonal_decomposition_forecast,
    seasonal_decomposition_components,
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
    fc_start_col = 4 + n_hist
    first_data_col = 4

    # Ensemble needs these 3 sheets to exist for cross-sheet MEDIAN references
    ensemble_deps = ["Seasonal Naive", "Holt-Winters (Multiplicative)", "Linear Trend + Seasonality"]
    if "Ensemble (Best-of-3)" in selected_methods:
        for dep in ensemble_deps:
            if dep not in selected_methods:
                selected_methods.append(dep)

    for method_name in selected_methods:
        method_fn = FORECAST_METHODS[method_name]
        ws = wb.create_sheet(title=method_name[:31])

        headers = ["Item_Name", "Group", "Metric"] + list(month_cols) + forecast_months + ["MAE", "MAPE (%)"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.fill = hfill
            cell.font = hfont
            cell.alignment = Alignment(horizontal="center")
            cell.border = border

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

            s = pd.Series(hist, index=pd.date_range("2020-01-01", periods=len(hist), freq="MS"))
            try:
                fc_values = [float(v) for v in method_fn(s)[:9]]
            except Exception:
                fc_values = [0.0] * 9

            _write_forecast_formulas(ws, r_idx, method_name, n_hist, first_data_col,
                                     fc_start_col, fc_values, hist, ffill, border)

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

    expl = wb.create_sheet(title="Method Explanations")
    expl.column_dimensions["A"].width = 30
    expl.column_dimensions["B"].width = 100
    expl.cell(row=1, column=1, value="Method").font = Font(bold=True, size=12)
    expl.cell(row=1, column=2, value="Explanation & Formula").font = Font(bold=True, size=12)
    for i, (mn, md) in enumerate(METHOD_DESCRIPTIONS.items(), 2):
        expl.cell(row=i, column=1, value=mn).font = Font(bold=True)
        expl.cell(row=i, column=2, value=md.strip())
        expl.row_dimensions[i].height = 80

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
        summ.cell(row=summ_row, column=1, value=f"-- {method_name} --").font = Font(bold=True, size=11, color="1F4E79")
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
    last_data_letter = get_column_letter(data_col + n_hist - 1)
    first_data_letter = get_column_letter(data_col)

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
        alpha = _find_best_alpha(hist)
        alpha_col = fc_col + 10
        ws.cell(row=r, column=alpha_col, value=round(alpha, 3))
        ws.cell(row=1, column=alpha_col, value="Alpha").font = Font(bold=True, size=9, color="888888")
        alpha_letter = get_column_letter(alpha_col)
        last_actual = get_column_letter(data_col + n_hist - 1)
        avg_letter_start = get_column_letter(data_col)
        avg_letter_end = get_column_letter(data_col + n_hist - 1)
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            cell.value = f"={alpha_letter}{r}*{last_actual}{r}+(1-{alpha_letter}{r})*AVERAGE({avg_letter_start}{r}:{avg_letter_end}{r})"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Linear Trend" in method_name:
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
            cell.value = f'=TREND({seq_start}{r}:{seq_end}{r},ROW(INDIRECT("1:{n_hist}")),{trend_x})*{si_col_letter}{r}'
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Holt-Winters" in method_name:
        seasonal_type = "mul" if "Multiplicativ" in method_name else "add"
        s = pd.Series(hist, index=pd.date_range("2020-01-01", periods=len(hist), freq="MS"))
        level, trend, season, period, stype = holt_winters_components(s, seasonal_type)
        # Helper columns: Level, Trend, Seasonal[0..period-1]
        helper_start = fc_col + 11
        ws.cell(row=r, column=helper_start, value=round(level, 4))
        ws.cell(row=r, column=helper_start + 1, value=round(trend, 4))
        for si_i, sv in enumerate(season):
            ws.cell(row=r, column=helper_start + 2 + si_i, value=round(sv, 4))
        if r == 2:
            ws.cell(row=1, column=helper_start, value="HW_Level").font = Font(size=8, color="888888")
            ws.cell(row=1, column=helper_start + 1, value="HW_Trend").font = Font(size=8, color="888888")
            for si_i in range(period):
                ws.cell(row=1, column=helper_start + 2 + si_i, value=f"HW_S{si_i+1}").font = Font(size=8, color="888888")
        lev_col = get_column_letter(helper_start)
        trn_col = get_column_letter(helper_start + 1)
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            h = fi + 1
            si = (n_hist + fi) % period
            s_col = get_column_letter(helper_start + 2 + si)
            if stype == "mul":
                cell.value = f"=({lev_col}{r}+{h}*{trn_col}{r})*{s_col}{r}"
            else:
                cell.value = f"={lev_col}{r}+{h}*{trn_col}{r}+{s_col}{r}"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Decomposition" in method_name:
        s = pd.Series(hist, index=pd.date_range("2020-01-01", periods=len(hist), freq="MS"))
        intercept, slope, seasonal, period = seasonal_decomposition_components(s)
        helper_start = fc_col + 11
        ws.cell(row=r, column=helper_start, value=round(intercept, 4))
        ws.cell(row=r, column=helper_start + 1, value=round(slope, 6))
        for si_i, sv in enumerate(seasonal):
            ws.cell(row=r, column=helper_start + 2 + si_i, value=round(sv, 4))
        if r == 2:
            ws.cell(row=1, column=helper_start, value="STL_Intcpt").font = Font(size=8, color="888888")
            ws.cell(row=1, column=helper_start + 1, value="STL_Slope").font = Font(size=8, color="888888")
            for si_i in range(period):
                ws.cell(row=1, column=helper_start + 2 + si_i, value=f"STL_S{si_i+1}").font = Font(size=8, color="888888")
        int_col = get_column_letter(helper_start)
        slp_col = get_column_letter(helper_start + 1)
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            future_t = n_hist + fi
            si = (n_hist + fi) % period
            s_col = get_column_letter(helper_start + 2 + si)
            cell.value = f"={int_col}{r}+{slp_col}{r}*{future_t}+{s_col}{r}"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    elif "Ensemble" in method_name:
        sn_sheet = "Seasonal Naive"[:31]
        hw_sheet = "Holt-Winters (Multiplicativ"[:31]
        lt_sheet = "Linear Trend + Seasonality"[:31]
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col)
            fc_letter = get_column_letter(col)
            cell.value = f"=MEDIAN('{sn_sheet}'!{fc_letter}{r},'{hw_sheet}'!{fc_letter}{r},'{lt_sheet}'!{fc_letter}{r})"
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'

    else:
        for fi in range(9):
            col = fc_col + fi
            cell = ws.cell(row=r, column=col, value=round(fc_values[fi], 2))
            cell.fill = ffill
            cell.border = border
            cell.number_format = '#,##0.00'


def _find_best_alpha(hist):
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
    d = get_column_letter(data_col)
    last = get_column_letter(data_col + n_hist - 1)
    if "Moving Average (3M)" in method_name:
        return "Each forecast = AVERAGE of previous 3 cells. First forecast: =AVERAGE(last 3 actuals). Subsequent: rolling window shifts right."
    elif "Moving Average (6M)" in method_name:
        return "Each forecast = AVERAGE of previous 6 cells. First forecast: =AVERAGE(last 6 actuals). Subsequent: rolling window shifts right."
    elif "Weighted Moving Average" in method_name:
        return "WMA = 0.10*fifth_last + 0.15*fourth_last + 0.20*third_last + 0.25*second_last + 0.30*last. Weights sum to 1.0, most recent gets highest weight."
    elif "Seasonal Naive" in method_name:
        return f"Each forecast month = same month from previous year. Apr forecast = Apr actual ({d}), May forecast = May actual, etc."
    elif "Exponential" in method_name:
        return "F = alpha * last_actual + (1-alpha) * avg(all_actuals). Alpha optimized by minimizing SSE. Stored in Alpha column."
    elif "Linear Trend" in method_name:
        return f"F = TREND(actuals, 1..{n_hist}, future_t) * Seasonal_Index[month]. TREND extrapolates linear fit. SI columns hold multiplicative seasonal indices."
    elif "Holt-Winters" in method_name:
        op = "*" if "Multiplicativ" in method_name else "+"
        return f"F = (HW_Level + h*HW_Trend) {op} HW_S[month]. Helper columns store optimized Level, Trend, and Seasonal factors. Click any yellow cell to see the formula."
    elif "Decomposition" in method_name:
        return "F = STL_Intercept + STL_Slope * t + STL_S[month]. Helper columns store trend intercept, slope, and seasonal offsets from centered MA decomposition."
    elif "Ensemble" in method_name:
        return "F = MEDIAN('Seasonal Naive'!cell, 'Holt-Winters (Multiplicativ'!cell, 'Linear Trend + Seasonality'!cell). Cross-sheet MEDIAN formula."
    return f"{method_name}: computed values from Python forecasting engine."


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Farmley Forecast</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:#f0f2f5;color:#1a1a2e;-webkit-font-smoothing:antialiased}

/* ── NAV ── */
.nav{background:#0f172a;padding:14px 32px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:50;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.nav-logo{width:36px;height:36px;background:#6366f1;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:1.1rem}
.nav h1{color:#fff;font-size:1.1rem;font-weight:700;letter-spacing:-.01em}
.nav-sub{color:#94a3b8;font-size:.75rem;margin-left:-6px}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:10px}
.nav-badge{background:rgba(99,102,241,.25);color:#a5b4fc;padding:4px 12px;border-radius:20px;font-size:.7rem;font-weight:600}

.wrap{max-width:1500px;margin:0 auto;padding:20px 24px 48px}

/* ── UPLOAD ── */
.upload-wrap{max-width:560px;margin:80px auto}
.upload-card{background:#fff;border-radius:16px;box-shadow:0 4px 24px rgba(0,0,0,.06);padding:48px 40px;text-align:center}
.upload-card h2{font-size:1.3rem;margin-bottom:4px}
.upload-card .sub{color:#64748b;font-size:.88rem;margin-bottom:28px}
.drop{border:2px dashed #cbd5e1;border-radius:12px;padding:36px 20px;background:#f8fafc;transition:.2s;cursor:pointer;position:relative}
.drop:hover,.drop.over{border-color:#6366f1;background:#eef2ff}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer}
.drop h3{font-size:.95rem;color:#334155;margin-bottom:4px}
.drop p{font-size:.8rem;color:#94a3b8}
#fileName{display:none;margin-top:10px;font-size:.85rem;color:#6366f1;font-weight:600}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:10px 22px;background:#6366f1;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:.88rem;font-weight:600;font-family:inherit;text-decoration:none;transition:.15s}
.btn:hover{background:#4f46e5}
.btn-block{width:100%}
.btn-green{background:#10b981}.btn-green:hover{background:#059669}
.btn-ghost{background:transparent;color:#6366f1;border:1px solid #e2e8f0}.btn-ghost:hover{background:#f8fafc}
.btn-sm{padding:4px 10px;font-size:.72rem;border-radius:6px}

/* ── LAYOUT ── */
.dash{display:grid;grid-template-columns:310px 1fr;gap:20px}
@media(max-width:960px){.dash{grid-template-columns:1fr}}

/* ── SIDEBAR ── */
.side{position:sticky;top:72px;max-height:calc(100vh - 90px);overflow-y:auto;display:flex;flex-direction:column;gap:12px}
.side::-webkit-scrollbar{width:3px}.side::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
@media(max-width:960px){.side{position:static;max-height:none}}
.panel{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.05);padding:16px;border:1px solid #e8ecf1}
.panel-title{font-size:.78rem;font-weight:700;color:#0f172a;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;display:flex;align-items:center;gap:6px}
.panel-title svg{width:14px;height:14px;color:#6366f1}

/* ── CHECKBOX FILTER ── */
.fgroup{margin-bottom:14px}
.fgroup-label{font-size:.72rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.fgroup-bar{display:flex;align-items:center;gap:4px;margin-bottom:6px}
.fgroup-bar input[type=text]{flex:1;padding:6px 10px;border:1px solid #e2e8f0;border-radius:6px;font-size:.78rem;font-family:inherit;outline:none;transition:.15s}
.fgroup-bar input[type=text]:focus{border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,.12)}
.cbox-list{max-height:180px;overflow-y:auto;border:1px solid #e8ecf1;border-radius:8px;background:#fafbfc;padding:4px 0}
.cbox-list::-webkit-scrollbar{width:4px}.cbox-list::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:4px}
.cbox-list label{display:flex;align-items:center;gap:8px;padding:5px 10px;font-size:.8rem;color:#334155;cursor:pointer;transition:.1s;user-select:none}
.cbox-list label:hover{background:#eef2ff}
.cbox-list label.hidden{display:none}
.cbox-list input[type=checkbox]{width:15px;height:15px;accent-color:#6366f1;cursor:pointer;flex-shrink:0}
.cbox-list .cname{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sel-info{font-size:.7rem;color:#94a3b8;margin-top:4px}

/* single select */
.sselect{margin-bottom:14px}
.sselect label{display:block;font-size:.72rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.sselect select{width:100%;padding:8px 10px;border:1px solid #e2e8f0;border-radius:6px;font-size:.82rem;font-family:inherit;background:#fff;outline:none;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' fill='%2394a3b8'%3E%3Cpath d='M5 7L1 3h8z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center}

/* ── STATS ── */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.st{background:#fff;border:1px solid #e8ecf1;border-radius:10px;padding:14px 16px}
.st-label{font-size:.68rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.st-val{font-size:1.15rem;font-weight:700;color:#0f172a;margin-top:2px}

/* ── CARDS ── */
.card{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.05);padding:20px;margin-bottom:16px;border:1px solid #e8ecf1}
.item-card{border-left:3px solid #6366f1}
.item-head{display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.item-head h2{font-size:1rem;font-weight:700;color:#0f172a;margin:0}
.tag{font-size:.68rem;font-weight:600;background:#f1f5f9;color:#64748b;padding:3px 9px;border-radius:20px}
.section-lbl{font-size:.8rem;font-weight:700;color:#0f172a;margin:18px 0 8px;display:flex;align-items:center;gap:6px}
.section-lbl svg{width:14px;height:14px;color:#6366f1}

/* ── TABLES ── */
table{width:100%;border-collapse:separate;border-spacing:0;font-size:.8rem}
thead th{background:#f8fafc;color:#64748b;padding:9px 12px;text-align:left;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid #e2e8f0;position:sticky;top:0;z-index:1}
tbody td{padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#334155}
tbody tr:hover{background:#f8fafc}
.fc-val{background:#fffbeb;font-weight:600;color:#92400e;font-variant-numeric:tabular-nums}
.scroll-tbl{overflow-x:auto;max-height:420px;overflow-y:auto;border:1px solid #e8ecf1;border-radius:8px}

/* ── BADGES ── */
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:600}
.badge-good{background:#ecfdf5;color:#065f46}
.badge-fair{background:#fffbeb;color:#92400e}
.badge-poor{background:#fef2f2;color:#991b1b}

/* ── METHOD TABS ── */
.tabs{display:flex;gap:4px;margin-bottom:14px;flex-wrap:wrap;padding:3px;background:#f1f5f9;border-radius:8px}
.tab{padding:7px 14px;cursor:pointer;font-size:.76rem;font-weight:500;color:#64748b;border-radius:6px;transition:.15s}
.tab:hover{color:#334155;background:#e8ecf1}
.tab.active{background:#fff;color:#6366f1;font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.tab-content{display:none}.tab-content.active{display:block}
.method-desc{line-height:1.7;font-size:.86rem;color:#475569}
.method-desc strong{color:#0f172a}
.method-desc code{background:#eef2ff;color:#6366f1;padding:2px 6px;border-radius:4px;font-size:.76rem}

.guide{margin-top:20px;padding:18px;background:#f8fafc;border-radius:10px;border:1px solid #e8ecf1}
.guide h3{font-size:.88rem;font-weight:700;color:#0f172a;margin-bottom:10px}
.guide-note{margin-top:12px;font-size:.78rem;color:#64748b;line-height:1.6}
.guide-note strong{color:#334155}

.empty{text-align:center;padding:60px 20px;color:#94a3b8}
.empty p{font-size:.9rem}.empty strong{color:#64748b}

footer{text-align:center;padding:28px 20px;color:#94a3b8;font-size:.72rem;border-top:1px solid #e8ecf1;margin-top:12px}
footer a{color:#6366f1;text-decoration:none}
</style>
</head>
<body>

<div class="nav">
    <div class="nav-logo">F</div>
    <h1>Farmley Forecast</h1>
    <span class="nav-sub">Sales Dashboard</span>
    {% if data_loaded %}
    <div class="nav-right"><span class="nav-badge">{{total_items}} Products</span></div>
    {% endif %}
</div>

{% if not data_loaded %}
<div class="upload-wrap">
    <div class="upload-card">
        <h2>Upload Sales Data</h2>
        <p class="sub">Excel file with item-wise monthly sales order data</p>
        <form method="post" enctype="multipart/form-data" action="/upload">
            <div class="drop" id="dropZone">
                <h3>Click to browse or drag & drop</h3>
                <p>.xlsx or .xls</p>
                <span id="fileName"></span>
                <input type="file" name="file" accept=".xlsx,.xls" required id="fileInput">
            </div>
            <button type="submit" class="btn btn-block" style="margin-top:18px">Upload & Analyze</button>
        </form>
    </div>
</div>
<script>
var dz=document.getElementById('dropZone'),fi=document.getElementById('fileInput'),fn=document.getElementById('fileName');
if(dz){
    ['dragenter','dragover'].forEach(function(e){dz.addEventListener(e,function(ev){ev.preventDefault();dz.classList.add('over')})});
    ['dragleave','drop'].forEach(function(e){dz.addEventListener(e,function(ev){ev.preventDefault();dz.classList.remove('over')})});
    dz.addEventListener('drop',function(e){if(e.dataTransfer.files.length){fi.files=e.dataTransfer.files;fn.style.display='block';fn.textContent=fi.files[0].name}});
    fi.addEventListener('change',function(){if(fi.files.length){fn.style.display='block';fn.textContent=fi.files[0].name}});
}
</script>

{% else %}

<div class="wrap">
<form method="post" action="/forecast" id="mainForm">
<div class="dash">

<!-- ── SIDEBAR ── -->
<div class="side">
    <div class="panel">
        <div class="panel-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
            Filters
        </div>

        <!-- Groups -->
        <div class="fgroup">
            <div class="fgroup-label">Product Group</div>
            <div class="fgroup-bar">
                <input type="text" placeholder="Search groups..." onkeyup="filterCB(this,'grp')">
                <button type="button" class="btn btn-sm btn-ghost" onclick="checkAll('grp')">All</button>
                <button type="button" class="btn btn-sm btn-ghost" onclick="uncheckAll('grp')">None</button>
            </div>
            <div class="cbox-list" id="grp">
            {% for g in all_groups %}
                <label><input type="checkbox" name="groups" value="{{g}}" {% if g in selected_groups %}checked{% endif %}><span class="cname">{{g}}</span></label>
            {% endfor %}
            </div>
            <div class="sel-info" id="grp_info"></div>
        </div>

        <!-- Items -->
        <div class="fgroup">
            <div class="fgroup-label">Item Name</div>
            <div class="fgroup-bar">
                <input type="text" placeholder="Search items..." onkeyup="filterCB(this,'itm')">
                <button type="button" class="btn btn-sm btn-ghost" onclick="checkAll('itm')">All</button>
                <button type="button" class="btn btn-sm btn-ghost" onclick="uncheckAll('itm')">None</button>
            </div>
            <div class="cbox-list" id="itm">
            {% for it in all_items %}
                <label><input type="checkbox" name="items" value="{{it}}" {% if it in selected_items %}checked{% endif %}><span class="cname">{{it}}</span></label>
            {% endfor %}
            </div>
            <div class="sel-info" id="itm_info"></div>
        </div>

        <!-- Metric -->
        <div class="sselect">
            <label>Metric</label>
            <select name="metric">
            {% for m in all_metrics %}
                <option value="{{m}}" {% if m == selected_metric %}selected{% endif %}>{{m}}</option>
            {% endfor %}
            </select>
        </div>

        <!-- Methods -->
        <div class="fgroup">
            <div class="fgroup-label">Forecast Methods</div>
            <div class="fgroup-bar">
                <button type="button" class="btn btn-sm btn-ghost" onclick="checkAll('mth')">All</button>
                <button type="button" class="btn btn-sm btn-ghost" onclick="uncheckAll('mth')">None</button>
            </div>
            <div class="cbox-list" id="mth" style="max-height:220px">
            {% for m in all_methods %}
                <label><input type="checkbox" name="methods" value="{{m}}" {% if m in selected_methods %}checked{% endif %}><span class="cname">{{m}}</span></label>
            {% endfor %}
            </div>
            <div class="sel-info" id="mth_info"></div>
        </div>

        <button type="submit" class="btn btn-block" style="margin-top:4px">Apply Filters</button>
    </div>

    <div class="panel">
        <div class="panel-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Export
        </div>
        <p style="font-size:.76rem;color:#94a3b8;margin-bottom:8px">Excel with real formulas, MAE/MAPE, one tab per method.</p>
        <a href="/download?methods={{selected_methods|join(',')}}" class="btn btn-green btn-block">Download Forecast Excel</a>
    </div>

    <div class="panel">
        <a href="/" class="btn btn-ghost btn-block">Upload New File</a>
    </div>
</div>

<!-- ── MAIN ── -->
<div>
    <div class="stats">
        <div class="st"><div class="st-label">Total Items</div><div class="st-val">{{total_items}}</div></div>
        <div class="st"><div class="st-label">Data Through</div><div class="st-val">{{last_month}}</div></div>
        <div class="st"><div class="st-label">Forecast Range</div><div class="st-val">{{forecast_months[0]}} - {{forecast_months[-1]}}</div></div>
        <div class="st"><div class="st-label">Months of Data</div><div class="st-val">{{n_months}}</div></div>
    </div>

    {% if overall_chart %}
    <div class="card" style="border-left:3px solid #10b981">
        <div class="item-head">
            <h2>Overall Aggregate</h2>
            <span class="tag" style="background:#ecfdf5;color:#065f46">{{results|length}} items combined</span>
            <span class="tag">{{selected_metric}}</span>
        </div>
        <div id="chart_overall" style="width:100%;height:460px"></div>
        <script>Plotly.newPlot('chart_overall',{{overall_chart|safe}}.data,{{overall_chart|safe}}.layout,{responsive:true})</script>
    </div>
    {% endif %}

    {% if group_charts %}
    {% for gc in group_charts %}
    <div class="card" style="border-left:3px solid #f59e0b">
        <div class="item-head">
            <h2>{{gc.group}}</h2>
            <span class="tag" style="background:#fffbeb;color:#92400e">{{gc.count}} items</span>
            <span class="tag">{{selected_metric}}</span>
            <span class="tag" style="background:#f5f3ff;color:#7c3aed">Group Total</span>
        </div>
        <div id="chart_grp_{{loop.index0}}" style="width:100%;height:440px"></div>
        <script>Plotly.newPlot('chart_grp_{{loop.index0}}',{{gc.chart_data|safe}}.data,{{gc.chart_data|safe}}.layout,{responsive:true})</script>
    </div>
    {% endfor %}
    {% endif %}

    {% for item_data in results %}
    <div class="card item-card">
        <div class="item-head">
            <h2>{{item_data.item}}</h2>
            <span class="tag">{{item_data.group}}</span>
            <span class="tag">{{selected_metric}}</span>
        </div>
        <div id="chart_{{loop.index0}}" style="width:100%;height:440px"></div>
        <script>Plotly.newPlot('chart_{{loop.index0}}',{{item_data.chart_data|safe}}.data,{{item_data.chart_data|safe}}.layout,{responsive:true})</script>

        <div class="section-lbl">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
            Forecast Values
        </div>
        <div class="scroll-tbl">
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
        <div class="section-lbl">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            Backtest Accuracy
            <span style="font-weight:400;font-size:.7rem;color:#94a3b8;margin-left:4px">(last 3 months held out)</span>
        </div>
        <div class="scroll-tbl">
        <table>
            <thead><tr><th>Method</th><th>MAE</th><th>MAPE (%)</th><th>Rating</th></tr></thead>
            <tbody>
            {% for acc in item_data.accuracy %}
            <tr>
                <td style="font-weight:500">{{acc.method}}</td>
                <td>{{acc.mae if acc.mae is not none else 'N/A'}}</td>
                <td>{{acc.mape if acc.mape is not none else 'N/A'}}</td>
                <td>{% if acc.rating == 'Good' %}<span class="badge badge-good">Good</span>
                    {% elif acc.rating == 'Fair' %}<span class="badge badge-fair">Fair</span>
                    {% elif acc.rating == 'Poor' %}<span class="badge badge-poor">Poor</span>
                    {% else %}-{% endif %}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        </div>
        {% endif %}
    </div>
    {% endfor %}

    {% if not results %}
    <div class="card"><div class="empty"><p>Select items and methods, then click <strong>Apply Filters</strong>.</p></div></div>
    {% endif %}

    <!-- METHOD EXPLANATIONS -->
    <div class="card">
        <h2 style="font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:14px">Forecasting Methods</h2>
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

        <div class="guide">
            <h3>Which Method to Use</h3>
            <div class="scroll-tbl">
            <table>
                <thead><tr><th>Sales Pattern</th><th>Recommended Method</th></tr></thead>
                <tbody>
                <tr><td>Strong seasonal peaks</td><td><strong>Holt-Winters (Multiplicative)</strong> or Ensemble</td></tr>
                <tr><td>Steady growth, no seasonality</td><td><strong>Exponential Smoothing</strong> or Moving Average</td></tr>
                <tr><td>Stable with mild seasonality</td><td><strong>Linear Trend + Seasonality</strong></td></tr>
                <tr><td>New product (few months)</td><td><strong>Moving Average (3M)</strong></td></tr>
                <tr><td>Unsure / mixed</td><td><strong>Ensemble (Best-of-3)</strong> - safest default</td></tr>
                </tbody>
            </table>
            </div>
            <div class="guide-note">
                <strong>MAE</strong> = avg absolute difference (lower is better, same units as data).<br>
                <strong>MAPE</strong> = avg % error. Below 30% = Good, 30-60% = Fair, above 60% = Poor.
            </div>
        </div>
    </div>
</div>
</div>
</form>
</div>

{% endif %}

<footer>
    Farmley Forecast Dashboard v5.0 &bull; Flask + Plotly &bull; <a href="https://github.com/Rameshwarnaik013/farmley-forecast-dashboard" target="_blank">GitHub</a>
</footer>

<script>
function filterCB(input, listId){
    var q=input.value.toLowerCase();
    var labels=document.getElementById(listId).querySelectorAll('label');
    labels.forEach(function(lb){
        var t=lb.querySelector('.cname').textContent.toLowerCase();
        lb.classList.toggle('hidden', t.indexOf(q)===-1);
    });
}
function checkAll(listId){
    document.getElementById(listId).querySelectorAll('label:not(.hidden) input[type=checkbox]').forEach(function(cb){cb.checked=true});
    updateInfo(listId);
}
function uncheckAll(listId){
    document.getElementById(listId).querySelectorAll('input[type=checkbox]').forEach(function(cb){cb.checked=false});
    updateInfo(listId);
}
function updateInfo(listId){
    var total=document.getElementById(listId).querySelectorAll('input[type=checkbox]').length;
    var checked=document.getElementById(listId).querySelectorAll('input[type=checkbox]:checked').length;
    var el=document.getElementById(listId+'_info');
    if(el) el.textContent=checked+' of '+total+' selected';
}
function showTab(el,id){
    el.parentElement.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active')});
    el.classList.add('active');
    var card=el.closest('.card');
    card.querySelectorAll('.tab-content').forEach(function(c){c.classList.remove('active')});
    card.querySelector('#'+id).classList.add('active');
}
// Init counts
['grp','itm','mth'].forEach(function(id){
    var el=document.getElementById(id);
    if(el){
        updateInfo(id);
        el.addEventListener('change', function(){updateInfo(id)});
    }
});
</script>
</body>
</html>"""


def md_to_html(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = text.replace('\n\n', '<br><br>').replace('\n-', '<br>*')
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
        selected_items = all_items
        selected_metric = all_metrics[0] if all_metrics else ""
        selected_methods = list(FORECAST_METHODS.keys())

    if not selected_groups:
        selected_groups = all_groups
    if not selected_methods:
        selected_methods = list(FORECAST_METHODS.keys())

    group_items = df[df["New MIS ITEM Group"].isin(selected_groups)]["Item_Name"].unique().tolist()
    filtered_items = sorted(set(all_items) & set(group_items))
    selected_items = [it for it in selected_items if it in filtered_items]
    if not selected_items:
        selected_items = filtered_items[:5]

    results = []
    for item_name in selected_items[:20]:
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

    # Build overall aggregate chart (sum across all selected items)
    overall_chart = None
    if results:
        agg_hist = np.zeros(len(month_cols))
        agg_forecasts = {m: np.zeros(9) for m in selected_methods}
        for rd in results:
            row = df[(df["Item_Name"] == rd["item"]) & (df["Metric"] == selected_metric)]
            if not row.empty:
                agg_hist += row.iloc[0][month_cols].values.astype(float)
            for m in selected_methods:
                agg_forecasts[m] += np.array(rd["forecasts"].get(m, [0.0]*9))
        agg_fc_dict = {m: agg_forecasts[m].tolist() for m in selected_methods}
        overall_chart = build_chart_json(
            f"All Selected Items ({len(results)})", selected_metric,
            agg_hist, agg_fc_dict, month_cols, forecast_months
        )

    # Build group-level aggregate charts
    group_charts = []
    if results:
        groups_in_results = {}
        for rd in results:
            g = rd["group"]
            if g not in groups_in_results:
                groups_in_results[g] = {"hist": np.zeros(len(month_cols)),
                                        "fc": {m: np.zeros(9) for m in selected_methods},
                                        "count": 0}
            row = df[(df["Item_Name"] == rd["item"]) & (df["Metric"] == selected_metric)]
            if not row.empty:
                groups_in_results[g]["hist"] += row.iloc[0][month_cols].values.astype(float)
            for m in selected_methods:
                groups_in_results[g]["fc"][m] += np.array(rd["forecasts"].get(m, [0.0]*9))
            groups_in_results[g]["count"] += 1
        for gname, gdata in sorted(groups_in_results.items()):
            if gdata["count"] < 2:
                continue  # skip groups with only 1 item (already shown in item view)
            gc_dict = {m: gdata["fc"][m].tolist() for m in selected_methods}
            gc_json = build_chart_json(
                f"{gname} ({gdata['count']} items)", selected_metric,
                gdata["hist"], gc_dict, month_cols, forecast_months
            )
            group_charts.append({"group": gname, "count": gdata["count"], "chart_data": gc_json})

    desc_html = {k: md_to_html(v) for k, v in METHOD_DESCRIPTIONS.items()}

    return render_template_string(TEMPLATE,
        data_loaded=True,
        all_groups=all_groups, selected_groups=selected_groups,
        all_items=filtered_items, selected_items=selected_items,
        all_metrics=all_metrics, selected_metric=selected_metric,
        all_methods=all_methods, selected_methods=selected_methods,
        forecast_months=forecast_months, results=results,
        overall_chart=overall_chart, group_charts=group_charts,
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
