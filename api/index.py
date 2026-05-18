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
        line=dict(color="#1f77b4", width=3), marker=dict(size=7),
    ))
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff6600"]
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
                  line=dict(color="gray", width=2, dash="dot"))
    fig.add_annotation(x=last_idx, y=1.05, yref="paper", text="← Actual | Forecast →",
                       showarrow=False, font=dict(size=11, color="gray"))
    fig.update_layout(
        title=f"{item_name} — {metric}", xaxis_title="Month", yaxis_title=metric,
        hovermode="x unified", height=480, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35), margin=dict(b=100),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_excel_output(df, month_cols, forecast_months, selected_methods):
    wb = openpyxl.Workbook()
    hfill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    hfont = Font(color="FFFFFF", bold=True, size=11)
    ffill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    afill = PatternFill(start_color="D6EAF8", end_color="D6EAF8", fill_type="solid")
    border = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"), bottom=Side(style="thin"))

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
            for ci, m in enumerate(month_cols, 4):
                v = float(row[m])
                cell = ws.cell(row=r_idx, column=ci, value=v)
                cell.fill = afill
                cell.border = border
                cell.number_format = '#,##0.00'
                hist.append(v)

            s = pd.Series(hist, index=pd.date_range("2020-01-01", periods=len(hist), freq="MS"))
            try:
                fc = [float(v) for v in method_fn(s)[:9]]
            except Exception:
                fc = [0.0] * 9
            mae, mape = backtest_method(np.array(hist), method_fn)
            fc_start = 4 + len(month_cols)
            for fi, fv in enumerate(fc):
                cell = ws.cell(row=r_idx, column=fc_start + fi, value=round(fv, 2))
                cell.fill = ffill
                cell.border = border
                cell.number_format = '#,##0.00'
            mae_col = fc_start + len(forecast_months)
            if mae is not None:
                ws.cell(row=r_idx, column=mae_col, value=mae).border = border
            if mape is not None:
                ws.cell(row=r_idx, column=mae_col + 1, value=mape).border = border

        for c in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 16
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    expl = wb.create_sheet(title="Method Explanations")
    expl.column_dimensions["A"].width = 30
    expl.column_dimensions["B"].width = 100
    expl.cell(row=1, column=1, value="Method").font = Font(bold=True, size=12)
    expl.cell(row=1, column=2, value="Explanation").font = Font(bold=True, size=12)
    for i, (mn, md) in enumerate(METHOD_DESCRIPTIONS.items(), 2):
        expl.cell(row=i, column=1, value=mn).font = Font(bold=True)
        expl.cell(row=i, column=2, value=md.strip())
        expl.row_dimensions[i].height = 60

    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Farmley Sales Forecast Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f8f9fa;color:#1a1a2e}
.header{background:linear-gradient(135deg,#1f4e79,#2980b9);color:#fff;padding:24px 32px}
.header h1{font-size:1.8rem;font-weight:700}
.header p{opacity:.85;margin-top:4px;font-size:.95rem}
.container{max-width:1400px;margin:0 auto;padding:20px}
.card{background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:24px;margin-bottom:20px}
.card h2{font-size:1.2rem;color:#1f4e79;margin-bottom:12px;border-bottom:2px solid #e8f0fe;padding-bottom:8px}
.grid{display:grid;gap:20px}
.grid-2{grid-template-columns:280px 1fr}
.upload-area{border:2px dashed #ccd;border-radius:10px;padding:40px 20px;text-align:center;background:#f8f9fc;transition:.2s}
.upload-area:hover{border-color:#2980b9;background:#eef5ff}
.upload-area input[type=file]{margin-top:12px}
.btn{display:inline-block;padding:10px 20px;background:#1f4e79;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:.9rem;text-decoration:none;transition:.2s}
.btn:hover{background:#16395a}
.btn-sm{padding:6px 14px;font-size:.8rem}
.btn-outline{background:transparent;color:#1f4e79;border:1px solid #1f4e79}
.btn-outline:hover{background:#1f4e79;color:#fff}
label{display:block;font-weight:600;margin-bottom:4px;font-size:.85rem;color:#444}
select,input[type=text]{width:100%;padding:8px 10px;border:1px solid #ccd;border-radius:6px;font-size:.85rem;margin-bottom:12px}
select[multiple]{height:180px}
.filter-section{margin-bottom:16px}
.filter-actions{display:flex;gap:6px;margin-bottom:6px}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{background:#1f4e79;color:#fff;padding:10px 12px;text-align:left;position:sticky;top:0}
td{padding:8px 12px;border-bottom:1px solid #eee}
tr:hover{background:#f0f7ff}
.fc-val{background:#fff8e1;font-weight:600}
.metric-cards{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.metric-card{background:#f0f7ff;border-radius:8px;padding:12px 20px;min-width:140px}
.metric-card .label{font-size:.75rem;color:#666;text-transform:uppercase;letter-spacing:.5px}
.metric-card .value{font-size:1.1rem;font-weight:700;color:#1f4e79;margin-top:2px}
.tabs{display:flex;gap:0;border-bottom:2px solid #e8f0fe;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:10px 18px;cursor:pointer;font-size:.85rem;color:#666;border-bottom:2px solid transparent;margin-bottom:-2px;transition:.2s}
.tab:hover{color:#1f4e79}
.tab.active{color:#1f4e79;border-bottom-color:#1f4e79;font-weight:600}
.tab-content{display:none}
.tab-content.active{display:block}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:600}
.badge-good{background:#d4edda;color:#155724}
.badge-fair{background:#fff3cd;color:#856404}
.badge-poor{background:#f8d7da;color:#721c24}
.method-desc{line-height:1.7;font-size:.9rem}
.method-desc strong{color:#1f4e79}
.method-desc code{background:#e8f0fe;padding:2px 6px;border-radius:4px;font-size:.8rem}
.scroll-table{overflow-x:auto;max-height:500px;overflow-y:auto}
.sidebar{position:sticky;top:20px;max-height:calc(100vh - 40px);overflow-y:auto;padding-right:10px}
@media(max-width:900px){.grid-2{grid-template-columns:1fr}.sidebar{position:static;max-height:none}}
</style>
</head>
<body>
<div class="header">
    <h1>Farmley Sales Forecast Dashboard</h1>
    <p>Upload item-wise sales order data &bull; Auto-detects data range &bull; Forecasts next 9 months</p>
</div>
<div class="container">

{% if not data_loaded %}
<div class="card">
    <h2>Upload Data</h2>
    <form method="post" enctype="multipart/form-data" action="/upload">
        <div class="upload-area">
            <p style="font-size:1.1rem;color:#555">Drop your Excel file here or click to browse</p>
            <input type="file" name="file" accept=".xlsx,.xls" required>
            <p style="margin-top:12px"><button type="submit" class="btn">Upload & Analyze</button></p>
        </div>
    </form>
</div>
{% else %}

<form method="post" action="/forecast" id="mainForm">
<div class="grid grid-2">

<!-- SIDEBAR -->
<div class="sidebar">
    <div class="card">
        <h2>Filters</h2>

        <div class="filter-section">
            <label>Product Group</label>
            <div class="filter-actions">
                <button type="button" class="btn btn-sm btn-outline" onclick="selAll('groups')">All</button>
                <button type="button" class="btn btn-sm btn-outline" onclick="clrAll('groups')">Clear</button>
            </div>
            <select name="groups" id="groups" multiple>
            {% for g in all_groups %}
                <option value="{{g}}" {% if g in selected_groups %}selected{% endif %}>{{g}}</option>
            {% endfor %}
            </select>
        </div>

        <div class="filter-section">
            <label>Item Name</label>
            <div class="filter-actions">
                <button type="button" class="btn btn-sm btn-outline" onclick="selAll('items')">All</button>
                <button type="button" class="btn btn-sm btn-outline" onclick="clrAll('items')">Clear</button>
            </div>
            <input type="text" id="itemSearch" placeholder="Search items..." oninput="filterSelect('items','itemSearch')">
            <select name="items" id="items" multiple>
            {% for it in all_items %}
                <option value="{{it}}" {% if it in selected_items %}selected{% endif %}>{{it}}</option>
            {% endfor %}
            </select>
        </div>

        <div class="filter-section">
            <label>Metric</label>
            <select name="metric">
            {% for m in all_metrics %}
                <option value="{{m}}" {% if m == selected_metric %}selected{% endif %}>{{m}}</option>
            {% endfor %}
            </select>
        </div>

        <div class="filter-section">
            <label>Forecast Methods</label>
            <div class="filter-actions">
                <button type="button" class="btn btn-sm btn-outline" onclick="selAll('methods')">All</button>
                <button type="button" class="btn btn-sm btn-outline" onclick="clrAll('methods')">Clear</button>
            </div>
            <select name="methods" id="methods" multiple>
            {% for m in all_methods %}
                <option value="{{m}}" {% if m in selected_methods %}selected{% endif %}>{{m}}</option>
            {% endfor %}
            </select>
        </div>

        <button type="submit" class="btn" style="width:100%;margin-top:8px">Apply Filters</button>
    </div>
    <div class="card" style="margin-top:12px">
        <h2>Download</h2>
        <p style="font-size:.85rem;color:#666;margin-bottom:10px">Excel with forecasts for all items, MAE/MAPE, formula comments.</p>
        <a href="/download?methods={{selected_methods|join(',')}}" class="btn" style="width:100%;text-align:center;display:block">Download Forecast Excel</a>
    </div>
    <div class="card" style="margin-top:12px">
        <a href="/" class="btn btn-outline" style="width:100%;text-align:center;display:block">Upload New File</a>
    </div>
</div>

<!-- MAIN CONTENT -->
<div>
    <div class="card">
        <div class="metric-cards">
            <div class="metric-card"><div class="label">Items</div><div class="value">{{total_items}}</div></div>
            <div class="metric-card"><div class="label">Data Through</div><div class="value">{{last_month}}</div></div>
            <div class="metric-card"><div class="label">Forecast</div><div class="value">{{forecast_months[0]}} → {{forecast_months[-1]}}</div></div>
            <div class="metric-card"><div class="label">Months of Data</div><div class="value">{{n_months}}</div></div>
        </div>
    </div>

    {% for item_data in results %}
    <div class="card">
        <h2>{{item_data.item}} <span style="font-weight:400;font-size:.85rem;color:#888">| {{item_data.group}} | {{selected_metric}}</span></h2>
        <div id="chart_{{loop.index0}}" style="width:100%;height:480px"></div>
        <script>Plotly.newPlot('chart_{{loop.index0}}',{{item_data.chart_data|safe}}.data,{{item_data.chart_data|safe}}.layout,{responsive:true})</script>

        <h3 style="margin:16px 0 8px;font-size:1rem;color:#1f4e79">Forecast Values</h3>
        <div class="scroll-table">
        <table>
            <tr><th>Month</th>{% for m in selected_methods %}<th>{{m}}</th>{% endfor %}</tr>
            {% for i in range(forecast_months|length) %}
            <tr><td><strong>{{forecast_months[i]}}</strong></td>
            {% for m in selected_methods %}
                <td class="fc-val">{{item_data.forecasts.get(m, [0]*9)[i]|round(2)|string|replace('.0','') if item_data.forecasts.get(m) else 'N/A'}}</td>
            {% endfor %}</tr>
            {% endfor %}
        </table>
        </div>

        {% if item_data.accuracy %}
        <h3 style="margin:16px 0 8px;font-size:1rem;color:#1f4e79">Backtest Accuracy <span style="font-weight:400;font-size:.8rem;color:#888">(trained on all-but-last-3 months, tested on last 3)</span></h3>
        <table>
            <tr><th>Method</th><th>MAE</th><th>MAPE (%)</th><th>Rating</th></tr>
            {% for acc in item_data.accuracy %}
            <tr>
                <td>{{acc.method}}</td>
                <td>{{acc.mae if acc.mae is not none else 'N/A'}}</td>
                <td>{{acc.mape if acc.mape is not none else 'N/A'}}</td>
                <td>{% if acc.rating == 'Good' %}<span class="badge badge-good">Good</span>
                    {% elif acc.rating == 'Fair' %}<span class="badge badge-fair">Fair</span>
                    {% elif acc.rating == 'Poor' %}<span class="badge badge-poor">Poor</span>
                    {% else %}—{% endif %}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endfor %}

    {% if not results %}
    <div class="card"><p style="color:#888;text-align:center;padding:40px">Select items and methods, then click <strong>Apply Filters</strong>.</p></div>
    {% endif %}

    <!-- METHOD EXPLANATIONS -->
    <div class="card">
        <h2>Forecasting Methods — Logic & Approach</h2>
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

        <div style="margin-top:20px;padding:16px;background:#f0f7ff;border-radius:8px">
            <h3 style="font-size:1rem;color:#1f4e79;margin-bottom:8px">Quick Guide — Which Method to Use</h3>
            <table>
                <tr><th>Sales Pattern</th><th>Recommended Method</th></tr>
                <tr><td>Strong seasonal peaks (festive months)</td><td>Holt-Winters (Multiplicative) or Ensemble</td></tr>
                <tr><td>Steady growth, no seasonality</td><td>Exponential Smoothing or Moving Average</td></tr>
                <tr><td>Stable with mild seasonality</td><td>Linear Trend + Seasonality</td></tr>
                <tr><td>New product (few months of data)</td><td>Moving Average (3M)</td></tr>
                <tr><td>Unsure / mixed pattern</td><td><strong>Ensemble (Best-of-3)</strong> — safest default</td></tr>
            </table>
            <p style="margin-top:12px;font-size:.85rem;color:#555">
                <strong>MAE</strong> (Mean Absolute Error): avg absolute difference. Lower = better. Same units as data.<br>
                <strong>MAPE</strong> (Mean Absolute % Error): avg % error. &lt;30% = Good, 30-60% = Fair, &gt;60% = Poor.
            </p>
        </div>
    </div>

</div>
</div>
</form>

{% endif %}
</div>

<footer style="text-align:center;padding:20px;color:#999;font-size:.8rem">
Farmley Sales Forecast Dashboard v3.0 &bull; Flask + Plotly
</footer>

<script>
function selAll(id){var s=document.getElementById(id);for(var i=0;i<s.options.length;i++)if(s.options[i].style.display!=='none')s.options[i].selected=true}
function clrAll(id){var s=document.getElementById(id);for(var i=0;i<s.options.length;i++)s.options[i].selected=false}
function filterSelect(selId,inputId){
    var q=document.getElementById(inputId).value.toLowerCase();
    var s=document.getElementById(selId);
    for(var i=0;i<s.options.length;i++){
        s.options[i].style.display=s.options[i].text.toLowerCase().includes(q)?'':'none';
    }
}
function showTab(el,id){
    el.parentElement.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    el.classList.add('active');
    var card=el.closest('.card');
    card.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
    card.querySelector('#'+id).classList.add('active');
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
