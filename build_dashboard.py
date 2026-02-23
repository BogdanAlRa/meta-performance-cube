#!/usr/bin/env python3
"""
Meta Performance Cube — Dashboard v2
Senate-directed redesign: all metrics surfaced, every axis labeled with units,
progressive disclosure via Plotly dropdown toggles.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

OUTPUT_DIR = Path(__file__).parent / "output"
DASH_PATH = Path(__file__).parent / "dashboard.html"

# ─── Design System ────────────────────────────────────────────────────────────

C = {
    "bg": "#0a0a0f",
    "card": "#12121a",
    "border": "#1e1e2e",
    "text": "#e0e0e0",
    "muted": "#6b7280",
    "indigo": "#6366f1",
    "cyan": "#06b6d4",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "green": "#10b981",
    "pink": "#ec4899",
    "purple": "#8b5cf6",
    "teal": "#14b8a6",
}

PAL = ["#6366f1", "#06b6d4", "#f59e0b", "#ef4444", "#10b981",
       "#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#64748b"]

GRID = "rgba(255,255,255,0.05)"
ZERO = "rgba(255,255,255,0.08)"


def lay(fig, title="", xlab="", ylab="", extra=None, h=None):
    """Apply layout to any figure with explicit axis labels."""
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color=C["text"], size=12),
        margin=dict(l=60, r=30, t=60, b=55),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(title=xlab, gridcolor=GRID, zerolinecolor=ZERO),
        yaxis=dict(title=ylab, gridcolor=GRID, zerolinecolor=ZERO),
    )
    if title:
        d["title"] = dict(text=title, font=dict(size=15))
    if h:
        d["height"] = h
    if extra:
        d.update(extra)
    fig.update_layout(**d)


def to_html(fig, height=420):
    return fig.to_html(
        full_html=False, include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
        default_height=f"{height}px",
    )


# ─── Unit Formatters ──────────────────────────────────────────────────────────

def fmt(val, unit):
    if pd.isna(val) or val == 0:
        return "—"
    if unit == "$":
        return f"${val:,.2f}" if val < 100 else f"${val:,.0f}"
    if unit == "x":
        return f"{val:.2f}x"
    if unit == "%":
        return f"{val:.2f}%"
    if unit == "#":
        return f"{val:,.0f}"
    return f"{val:,.2f}"


METRIC_DEF = {
    "spend":          ("Spend", "$"),
    "impressions":    ("Impressions", "#"),
    "clicks":         ("Clicks", "#"),
    "reach":          ("Reach", "#"),
    "frequency":      ("Frequency", "#"),
    "cpm":            ("CPM", "$"),
    "cpc":            ("CPC", "$"),
    "ctr":            ("CTR", "%"),
    "purchase":       ("Purchases", "#"),
    "value_purchase": ("Revenue", "$"),
    "roas":           ("ROAS", "x"),
    "cpa":            ("CPA", "$"),
    "aov":            ("AOV", "$"),
}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_account(name):
    d = OUTPUT_DIR / name
    data = {}
    for f in d.glob("*.csv"):
        try:
            data[f.stem] = pd.read_csv(f, low_memory=False)
        except Exception:
            pass
    return data


def safe(df, col, default=0):
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)


# ─── KPI Cards ────────────────────────────────────────────────────────────────

def kpi(label, value, unit, sub="", color=C["indigo"]):
    v = fmt(value, unit)
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{v}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def detect_date_range(totals):
    """Extract date range and day count from the totals slice."""
    if totals is None or totals.empty or "date_start" not in totals.columns:
        return None, None, 0
    dates = pd.to_datetime(totals["date_start"])
    d_min = dates.min()
    d_max = dates.max()
    n_days = (d_max - d_min).days + 1
    return d_min, d_max, n_days


def build_time_banner(d_min, d_max, n_days):
    """Prominent temporal context bar."""
    if d_min is None:
        return ""
    d1 = d_min.strftime("%b %d, %Y")
    d2 = d_max.strftime("%b %d, %Y")
    weeks = n_days / 7
    if n_days <= 7:
        span_label = f"{n_days} days"
    elif n_days <= 35:
        span_label = f"{weeks:.1f} weeks ({n_days} days)"
    else:
        months = n_days / 30.44
        span_label = f"{months:.1f} months ({n_days} days)"

    return f"""<div class="time-banner">
        <span class="time-range">{d1} &rarr; {d2}</span>
        <span class="time-span">{span_label}</span>
    </div>"""


def build_kpis(totals):
    if totals is None or totals.empty:
        return ""

    d_min, d_max, n_days = detect_date_range(totals)
    n_days = max(n_days, 1)

    s = totals["spend"].sum()
    imp = totals["impressions"].sum()
    clk = safe(totals, "clicks").sum()
    pur = safe(totals, "purchase").sum()
    rev = safe(totals, "value_purchase").sum()
    roas = rev / s if s > 0 else 0
    cpa_v = s / pur if pur > 0 else 0
    aov = rev / pur if pur > 0 else 0
    freq = safe(totals, "frequency").mean()
    cpm = (s / imp * 1000) if imp > 0 else 0
    cpc = s / clk if clk > 0 else 0
    ctr = (clk / imp * 100) if imp > 0 else 0

    # Daily averages
    d_spend = s / n_days
    d_rev = rev / n_days
    d_pur = pur / n_days
    d_imp = imp / n_days
    d_clk = clk / n_days

    time_banner = build_time_banner(d_min, d_max, n_days)

    row1 = f"""{time_banner}
    <div class="kpi-row">
        {kpi("Total Spend", s, "$", f"{fmt(d_spend, '$')}/day &middot; {imp/1e6:.1f}M impr", C['indigo'])}
        {kpi("Revenue", rev, "$", f"{fmt(d_rev, '$')}/day &middot; {pur:,.0f} purchases", C['green'])}
        {kpi("ROAS", roas, "x", "revenue / spend", C['cyan'] if roas >= 1 else C['red'])}
        {kpi("CPA", cpa_v, "$", "spend / purchases", C['amber'])}
        {kpi("AOV", aov, "$", "revenue / purchases", C['purple'])}
        {kpi("Frequency", freq, "#", "avg impressions per user", C['teal'])}
        {kpi("CPM", cpm, "$", "cost per 1K impressions", C['pink'])}
        {kpi("CTR", ctr, "%", f"CPC: {fmt(cpc, '$')}", C['indigo'])}
    </div>
    <div class="kpi-row kpi-row-secondary">
        {kpi("Daily Spend", d_spend, "$", f"over {n_days} days", C['indigo'])}
        {kpi("Daily Revenue", d_rev, "$", f"{d_pur:.1f} purchases/day", C['green'])}
        {kpi("Daily Impressions", d_imp, "#", f"{d_clk:,.0f} clicks/day", C['cyan'])}
        {kpi("Clicks", clk, "#", f"{fmt(cpc, '$')} per click", C['amber'])}
    </div>"""
    return row1


# ─── Chart: Age × Gender Heatmap with Metric Toggle ──────────────────────────

def build_demo_heatmaps(df):
    """Heatmap with dropdown to toggle between spend/ROAS/CPA/AOV/purchases/CTR."""
    if df is None or df.empty:
        return ""
    df = df[~df["age"].isin(["Unknown"])].copy()
    df = df[~df["gender"].isin(["unknown", "Unknown"])].copy()
    if df.empty:
        return ""

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    metrics = []
    for col, (label, unit) in METRIC_DEF.items():
        if col in df.columns and df[col].sum() > 0:
            metrics.append((col, label, unit))

    if not metrics:
        return ""

    fig = go.Figure()
    buttons = []
    for i, (col, label, unit) in enumerate(metrics):
        pivot = df.pivot_table(index="age", columns="gender", values=col, aggfunc="sum").fillna(0)
        pivot = pivot.reindex([a for a in age_order if a in pivot.index])

        prefix = "$" if unit == "$" else ""
        suffix = "x" if unit == "x" else ("%" if unit == "%" else "")
        text_matrix = [[f"{prefix}{v:,.0f}{suffix}" if unit in ("$", "#") else f"{v:.2f}{suffix}" for v in row] for row in pivot.values]

        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0, "#0f0f1a"], [0.25, "#1e1b4b"], [0.5, "#4338ca"], [0.75, "#6366f1"], [1, "#a78bfa"]],
            text=text_matrix, texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            hovertemplate=f"Age: %{{y}}<br>Gender: %{{x}}<br>{label}: %{{text}}<extra></extra>",
            colorbar=dict(title=dict(text=f"{label} ({unit})", font=dict(color=C["muted"], size=11)), tickfont=dict(color=C["muted"])),
            visible=(i == 0),
            name=label,
        ))

        vis = [False] * len(metrics)
        vis[i] = True
        buttons.append(dict(
            label=f"{label} ({unit})",
            method="update",
            args=[{"visible": vis}, {"title": dict(text=f"Demographics: {label} by Age x Gender", font=dict(size=15))}],
        ))

    lay(fig, title=f"Demographics: {metrics[0][1]} by Age x Gender",
        xlab="Gender", ylab="Age Group",
        extra=dict(
            xaxis=dict(side="top", title="Gender", gridcolor=GRID),
            yaxis=dict(autorange="reversed", title="Age Group", gridcolor=GRID),
            updatemenus=[dict(
                type="dropdown", direction="down",
                x=1.0, xanchor="right", y=1.18, yanchor="top",
                bgcolor=C["card"], bordercolor=C["border"],
                font=dict(color=C["text"], size=11),
                buttons=buttons,
            )],
        ))
    return to_html(fig, 400)


# ─── Chart: Age × Gender Grouped Bars (ROAS + CPA dual axis) ─────────────────

def build_demo_bars(df):
    """Grouped bars: ROAS by age×gender with breakeven line."""
    if df is None or df.empty:
        return ""
    df = df[~df["age"].isin(["Unknown"])].copy()
    df = df[~df["gender"].isin(["unknown", "Unknown"])].copy()

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    gender_colors = {"female": C["pink"], "male": C["indigo"]}

    # Build with metric toggle: ROAS, CPA, AOV, CTR
    toggle_metrics = []
    for col, label, unit in [("roas", "ROAS", "x"), ("cpa", "CPA", "$"), ("aov", "AOV", "$"), ("ctr", "CTR", "%")]:
        if col in df.columns and df[col].sum() > 0:
            toggle_metrics.append((col, label, unit))

    if not toggle_metrics:
        return ""

    fig = go.Figure()
    buttons = []
    trace_idx = 0

    for mi, (col, label, unit) in enumerate(toggle_metrics):
        for gender in ["female", "male"]:
            gdf = df[df["gender"] == gender].copy()
            gdf = gdf[gdf[col] > 0]
            gdf["age"] = pd.Categorical(gdf["age"], categories=age_order, ordered=True)
            gdf = gdf.sort_values("age")

            prefix = "$" if unit == "$" else ""
            suffix = "x" if unit == "x" else ("%" if unit == "%" else "")
            text = gdf[col].apply(lambda v: f"{prefix}{v:,.2f}{suffix}")

            fig.add_trace(go.Bar(
                x=gdf["age"], y=gdf[col].round(2),
                name=f"{gender.title()}",
                marker_color=gender_colors.get(gender, C["cyan"]),
                text=text, textposition="outside", textfont=dict(size=10),
                hovertemplate=f"%{{x}} {gender.title()}<br>{label}: {prefix}%{{y:,.2f}}{suffix}<br>"
                              f"Spend: ${'{0}'.format('%{customdata[0]:,.0f}')}<br>"
                              f"Purchases: %{{customdata[1]:,.0f}}<extra></extra>",
                customdata=list(zip(gdf.get("spend", [0]*len(gdf)), gdf.get("purchase", [0]*len(gdf)))),
                visible=(mi == 0),
                legendgroup=gender,
                showlegend=(mi == 0),
            ))
            trace_idx += 1

        n_traces = len(toggle_metrics) * 2
        vis = [False] * n_traces
        vis[mi * 2] = True
        vis[mi * 2 + 1] = True
        buttons.append(dict(
            label=f"{label} ({unit})",
            method="update",
            args=[{"visible": vis},
                  {"title": dict(text=f"{label} by Age x Gender", font=dict(size=15)),
                   "yaxis.title": f"{label} ({unit})"}],
        ))

    ref_line = 1.0 if toggle_metrics[0][0] == "roas" else None
    if ref_line is not None:
        fig.add_hline(y=ref_line, line_dash="dash", line_color=C["amber"], opacity=0.6,
                      annotation_text="Breakeven (1.0x)", annotation_font_color=C["amber"])

    first_label, first_unit = toggle_metrics[0][1], toggle_metrics[0][2]
    lay(fig, title=f"{first_label} by Age x Gender",
        xlab="Age Group", ylab=f"{first_label} ({first_unit})",
        extra=dict(
            barmode="group",
            updatemenus=[dict(
                type="dropdown", direction="down",
                x=1.0, xanchor="right", y=1.18, yanchor="top",
                bgcolor=C["card"], bordercolor=C["border"],
                font=dict(color=C["text"], size=11),
                buttons=buttons,
            )],
        ))
    return to_html(fig, 420)


# ─── Chart: Demographics Efficiency Scatter ───────────────────────────────────

def build_demo_scatter(df):
    """Scatter: CPA ($) vs ROAS (x), bubble size = spend ($), color by gender."""
    if df is None or df.empty:
        return ""
    needed = ["cpa", "roas", "spend"]
    if not all(c in df.columns for c in needed):
        return ""
    df = df[~df["age"].isin(["Unknown"])].copy()
    df = df[~df["gender"].isin(["unknown", "Unknown"])].copy()
    df = df[(df["cpa"] > 0) & (df["roas"] > 0)].copy()
    if df.empty:
        return ""

    df["label"] = df["age"] + " " + df["gender"].str.title()
    gender_colors = {"female": C["pink"], "male": C["indigo"]}

    fig = go.Figure()
    for gender in ["female", "male"]:
        gdf = df[df["gender"] == gender]
        fig.add_trace(go.Scatter(
            x=gdf["cpa"], y=gdf["roas"],
            mode="markers+text",
            name=gender.title(),
            marker=dict(
                size=gdf["spend"] ** 0.45 * 2.5,
                color=gender_colors[gender], opacity=0.75,
                line=dict(width=1.5, color="rgba(255,255,255,0.2)"),
            ),
            text=gdf["age"],
            textposition="top center",
            textfont=dict(size=9, color=C["muted"]),
            customdata=list(zip(gdf["spend"], gdf["purchase"], gdf["value_purchase"], gdf["aov"])),
            hovertemplate=(
                "<b>%{text} " + gender.title() + "</b><br>"
                "CPA: $%{x:,.0f}<br>"
                "ROAS: %{y:.2f}x<br>"
                "Spend: $%{customdata[0]:,.0f}<br>"
                "Purchases: %{customdata[1]:,.0f}<br>"
                "Revenue: $%{customdata[2]:,.0f}<br>"
                "AOV: $%{customdata[3]:,.0f}<extra></extra>"
            ),
        ))

    fig.add_hline(y=1, line_dash="dash", line_color=C["amber"], opacity=0.4,
                  annotation_text="ROAS Breakeven", annotation_font_color=C["amber"])

    lay(fig, title="Efficiency: CPA vs ROAS by Age x Gender",
        xlab="CPA — Cost per Purchase ($)", ylab="ROAS — Return on Ad Spend (x)")
    return to_html(fig, 420)


# ─── Chart: Delivery Treemap ─────────────────────────────────────────────────

def build_delivery_treemap(df):
    if df is None or df.empty:
        return ""
    df = df[df["spend"] > 0].copy()
    if df.empty:
        return ""

    labels, parents, values, colors_list, hover_texts = [], [], [], [], []
    labels.append("All Placements")
    parents.append("")
    values.append(0)
    colors_list.append(C["card"])
    hover_texts.append("")

    platforms = df.groupby("publisher_platform")["spend"].sum().sort_values(ascending=False)
    plat_colors = dict(zip(platforms.index, PAL))

    for plat in platforms.index:
        pdf = df[df["publisher_platform"] == plat]
        ps = pdf["spend"].sum()
        pi = pdf["impressions"].sum()
        pc = safe(pdf, "clicks").sum()
        pp = safe(pdf, "purchase").sum()
        pr = safe(pdf, "value_purchase").sum()
        roas = pr / ps if ps > 0 else 0
        cpm = ps / pi * 1000 if pi > 0 else 0
        ctr = pc / pi * 100 if pi > 0 else 0

        labels.append(plat.title())
        parents.append("All Placements")
        values.append(ps)
        colors_list.append(plat_colors[plat])
        hover_texts.append(
            f"<b>{plat.title()}</b><br>"
            f"Spend: ${ps:,.0f}<br>Revenue: ${pr:,.0f}<br>"
            f"ROAS: {roas:.2f}x<br>Purchases: {pp:,.0f}<br>"
            f"CPM: ${cpm:.2f}<br>CTR: {ctr:.2f}%<br>"
            f"Impressions: {pi:,.0f}"
        )

        for pos in pdf.groupby("platform_position")["spend"].sum().sort_values(ascending=False).index:
            posdf = pdf[pdf["platform_position"] == pos]
            s = posdf["spend"].sum()
            i = posdf["impressions"].sum()
            ck = safe(posdf, "clicks").sum()
            p = safe(posdf, "purchase").sum()
            r = safe(posdf, "value_purchase").sum()
            ro = r / s if s > 0 else 0
            cm = s / i * 1000 if i > 0 else 0
            ct = ck / i * 100 if i > 0 else 0
            cpa = s / p if p > 0 else 0

            label = pos.replace("_", " ").title()
            labels.append(label)
            parents.append(plat.title())
            values.append(s)
            colors_list.append(plat_colors[plat])
            hover_texts.append(
                f"<b>{plat.title()} / {label}</b><br>"
                f"Spend: ${s:,.0f}<br>Revenue: ${r:,.0f}<br>"
                f"ROAS: {ro:.2f}x<br>CPA: {fmt(cpa, '$')}<br>"
                f"Purchases: {p:,.0f}<br>"
                f"CPM: ${cm:.2f}<br>CTR: {ct:.2f}%<br>"
                f"Clicks: {ck:,.0f}<br>Impressions: {i:,.0f}"
            )

    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colors_list, line=dict(color=C["bg"], width=2)),
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>$%{value:,.0f}",
        hovertext=hover_texts, hoverinfo="text",
        textfont=dict(size=12),
    ))
    lay(fig, title="Spend ($) by Placement — Hover for Full Metrics",
        extra=dict(margin=dict(l=10, r=10, t=60, b=10)))
    return to_html(fig, 500)


# ─── Chart: Delivery ROAS + CPM Bars ─────────────────────────────────────────

def build_delivery_bars(df):
    if df is None or df.empty:
        return ""
    df = df[df["spend"] > 30].copy()
    if df.empty:
        return ""

    df["label"] = df.apply(
        lambda r: f"{r.get('publisher_platform','')} / {str(r.get('platform_position','')).replace('_',' ')} / {r.get('device_platform','')}",
        axis=1)

    # Compute all metrics
    df["calc_cpm"] = df["spend"] / df["impressions"] * 1000
    df["calc_ctr"] = safe(df, "clicks") / df["impressions"] * 100
    df["calc_cpa"] = df["spend"] / safe(df, "purchase").replace(0, float("nan"))
    if "roas" not in df.columns:
        df["roas"] = safe(df, "value_purchase") / df["spend"].replace(0, float("nan"))

    toggle = [
        ("roas", "ROAS", "x", 1.0, "Breakeven"),
        ("calc_cpm", "CPM", "$", None, None),
        ("calc_ctr", "CTR", "%", None, None),
        ("calc_cpa", "CPA", "$", None, None),
    ]
    toggle = [(c, l, u, ref, rl) for c, l, u, ref, rl in toggle if c in df.columns and df[c].sum() > 0]
    if not toggle:
        return ""

    fig = go.Figure()
    buttons = []

    for mi, (col, label, unit, ref_val, ref_label) in enumerate(toggle):
        sdf = df.dropna(subset=[col]).sort_values(col, ascending=True).tail(20)
        prefix = "$" if unit == "$" else ""
        suffix = "x" if unit == "x" else ("%" if unit == "%" else "")

        bar_colors = []
        if ref_val is not None:
            bar_colors = [C["green"] if v >= ref_val else C["red"] for v in sdf[col]]
        else:
            bar_colors = [C["indigo"]] * len(sdf)

        fig.add_trace(go.Bar(
            x=sdf[col].round(2), y=sdf["label"], orientation="h",
            marker_color=bar_colors,
            text=sdf.apply(lambda r: f"{prefix}{r[col]:,.2f}{suffix}  |  Spend: ${r['spend']:,.0f}", axis=1),
            textposition="outside", textfont=dict(size=9),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{label}: {prefix}%{{x:,.2f}}{suffix}<br>"
                "Spend: $%{customdata[0]:,.0f}<br>"
                "Impressions: %{customdata[1]:,.0f}<br>"
                "Clicks: %{customdata[2]:,.0f}<extra></extra>"
            ),
            customdata=list(zip(sdf["spend"], sdf["impressions"], safe(sdf, "clicks"))),
            visible=(mi == 0),
        ))

        vis = [False] * len(toggle)
        vis[mi] = True
        buttons.append(dict(
            label=f"{label} ({unit})",
            method="update",
            args=[{"visible": vis},
                  {"title": dict(text=f"{label} by Placement", font=dict(size=15)),
                   "xaxis.title": f"{label} ({unit})"}],
        ))

    first = toggle[0]
    if first[3] is not None:
        fig.add_vline(x=first[3], line_dash="dash", line_color=C["amber"], opacity=0.5)

    h = max(400, min(len(df), 20) * 28 + 100)
    lay(fig, title=f"{first[1]} by Placement",
        xlab=f"{first[1]} ({first[2]})", ylab="Placement",
        extra=dict(
            height=h,
            updatemenus=[dict(
                type="dropdown", direction="down",
                x=1.0, xanchor="right", y=1.12, yanchor="top",
                bgcolor=C["card"], bordercolor=C["border"],
                font=dict(color=C["text"], size=11),
                buttons=buttons,
            )],
        ))
    return to_html(fig, h)


# ─── Chart: Geography Choropleth ──────────────────────────────────────────────

ISO2_TO_ISO3 = {
    "AF":"AFG","AL":"ALB","DZ":"DZA","AS":"ASM","AD":"AND","AO":"AGO","AG":"ATG","AR":"ARG","AM":"ARM",
    "AU":"AUS","AT":"AUT","AZ":"AZE","BS":"BHS","BH":"BHR","BD":"BGD","BB":"BRB","BY":"BLR","BE":"BEL",
    "BZ":"BLZ","BJ":"BEN","BM":"BMU","BT":"BTN","BO":"BOL","BA":"BIH","BW":"BWA","BR":"BRA","BN":"BRN",
    "BG":"BGR","BF":"BFA","BI":"BDI","KH":"KHM","CM":"CMR","CA":"CAN","CV":"CPV","CF":"CAF","TD":"TCD",
    "CL":"CHL","CN":"CHN","CO":"COL","KM":"COM","CG":"COG","CD":"COD","CR":"CRI","CI":"CIV","HR":"HRV",
    "CU":"CUB","CY":"CYP","CZ":"CZE","DK":"DNK","DJ":"DJI","DM":"DMA","DO":"DOM","EC":"ECU","EG":"EGY",
    "SV":"SLV","GQ":"GNQ","ER":"ERI","EE":"EST","ET":"ETH","FJ":"FJI","FI":"FIN","FR":"FRA","GA":"GAB",
    "GM":"GMB","GE":"GEO","DE":"DEU","GH":"GHA","GR":"GRC","GL":"GRL","GD":"GRD","GT":"GTM","GN":"GIN",
    "GW":"GNB","GY":"GUY","HT":"HTI","HN":"HND","HK":"HKG","HU":"HUN","IS":"ISL","IN":"IND","ID":"IDN",
    "IR":"IRN","IQ":"IRQ","IE":"IRL","IL":"ISR","IT":"ITA","JM":"JAM","JP":"JPN","JO":"JOR","KZ":"KAZ",
    "KE":"KEN","KI":"KIR","KP":"PRK","KR":"KOR","KW":"KWT","KG":"KGZ","LA":"LAO","LV":"LVA","LB":"LBN",
    "LS":"LSO","LR":"LBR","LY":"LBY","LI":"LIE","LT":"LTU","LU":"LUX","MO":"MAC","MK":"MKD","MG":"MDG",
    "MW":"MWI","MY":"MYS","MV":"MDV","ML":"MLI","MT":"MLT","MH":"MHL","MR":"MRT","MU":"MUS","MX":"MEX",
    "FM":"FSM","MD":"MDA","MC":"MCO","MN":"MNG","ME":"MNE","MA":"MAR","MZ":"MOZ","MM":"MMR","NA":"NAM",
    "NR":"NRU","NP":"NPL","NL":"NLD","NZ":"NZL","NI":"NIC","NE":"NER","NG":"NGA","NO":"NOR","OM":"OMN",
    "PK":"PAK","PW":"PLW","PA":"PAN","PG":"PNG","PY":"PRY","PE":"PER","PH":"PHL","PL":"POL","PT":"PRT",
    "QA":"QAT","RO":"ROU","RU":"RUS","RW":"RWA","KN":"KNA","LC":"LCA","VC":"VCT","WS":"WSM","SM":"SMR",
    "ST":"STP","SA":"SAU","SN":"SEN","RS":"SRB","SC":"SYC","SL":"SLE","SG":"SGP","SK":"SVK","SI":"SVN",
    "SB":"SLB","SO":"SOM","ZA":"ZAF","ES":"ESP","LK":"LKA","SD":"SDN","SR":"SUR","SZ":"SWZ","SE":"SWE",
    "CH":"CHE","SY":"SYR","TW":"TWN","TJ":"TJK","TZ":"TZA","TH":"THA","TL":"TLS","TG":"TGO","TO":"TON",
    "TT":"TTO","TN":"TUN","TR":"TUR","TM":"TKM","TV":"TUV","UG":"UGA","UA":"UKR","AE":"ARE","GB":"GBR",
    "US":"USA","UY":"URY","UZ":"UZB","VU":"VUT","VE":"VEN","VN":"VNM","YE":"YEM","ZM":"ZMB","ZW":"ZWE",
    "PS":"PSE","XK":"XKX","SS":"SSD","CW":"CUW","SX":"SXM","PR":"PRI","GU":"GUM","VI":"VIR","RE":"REU",
    "GP":"GLP","MQ":"MTQ","GF":"GUF","YT":"MYT","NC":"NCL","PF":"PYF",
}


def build_geo_map(df):
    if df is None or df.empty or "country" not in df.columns:
        return ""
    df = df[df["country"] != "unknown"].copy()
    if df.empty or len(df) < 2:
        return ""

    # Convert alpha-2 to alpha-3 for Plotly choropleth
    df["country_iso3"] = df["country"].map(ISO2_TO_ISO3)
    df = df.dropna(subset=["country_iso3"])

    # Build rich hover
    hover = df.apply(lambda r: (
        f"<b>{r['country']}</b><br>"
        f"Spend: ${r['spend']:,.0f}<br>"
        f"Impressions: {r['impressions']:,.0f}<br>"
        f"Clicks: {safe(df, 'clicks').loc[r.name]:,.0f}<br>"
        + (f"Purchases: {r['purchase']:,.0f}<br>" if 'purchase' in df.columns and r.get('purchase', 0) > 0 else "")
        + (f"Revenue: ${r['value_purchase']:,.0f}<br>" if 'value_purchase' in df.columns and r.get('value_purchase', 0) > 0 else "")
        + (f"ROAS: {r['roas']:.2f}x<br>" if 'roas' in df.columns and pd.notna(r.get('roas')) and r.get('roas', 0) > 0 else "")
        + (f"CPA: ${r['cpa']:,.0f}<br>" if 'cpa' in df.columns and pd.notna(r.get('cpa')) and r.get('cpa', 0) > 0 else "")
    ), axis=1)

    fig = go.Figure(data=go.Choropleth(
        locations=df["country_iso3"], z=df["spend"],
        text=hover, hoverinfo="text",
        colorscale=[[0, "#0f0f1a"], [0.3, "#1e1b4b"], [0.6, "#4338ca"], [1, "#6366f1"]],
        marker_line_color=C["border"], marker_line_width=0.5,
        colorbar=dict(
            title=dict(text="Spend ($)", font=dict(color=C["muted"], size=11)),
            tickfont=dict(color=C["muted"]), tickprefix="$",
        ),
    ))
    lay(fig, title="Spend ($) by Country — Hover for Full Metrics",
        extra=dict(
            geo=dict(
                bgcolor="rgba(0,0,0,0)", lakecolor=C["bg"], landcolor="#0f0f1a",
                showframe=False, showcoastlines=True, coastlinecolor="#1e1e2e",
                projection_type="natural earth",
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        ))
    return to_html(fig, 440)


# ─── Chart: Geography Table ──────────────────────────────────────────────────

def build_geo_table(df):
    if df is None or df.empty or "country" not in df.columns:
        return ""
    df = df[df["country"] != "unknown"].copy()
    if df.empty:
        return ""

    df = df.sort_values("spend", ascending=False).head(25)

    cols = ["country", "spend", "impressions", "clicks", "purchase", "value_purchase", "roas", "cpa"]
    cols = [c for c in cols if c in df.columns]
    headers = []
    cells = []
    for col in cols:
        label, unit = METRIC_DEF.get(col, (col, ""))
        headers.append(f"{label} ({unit})" if unit else label)
        vals = df[col].tolist()
        if unit == "$":
            cells.append([fmt(v, "$") for v in vals])
        elif unit == "x":
            cells.append([fmt(v, "x") for v in vals])
        elif unit == "%":
            cells.append([fmt(v, "%") for v in vals])
        elif unit == "#":
            cells.append([f"{v:,.0f}" if pd.notna(v) else "—" for v in vals])
        else:
            cells.append([str(v) for v in vals])

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color=C["card"], font=dict(color=C["text"], size=12),
            align="left", line_color=C["border"],
        ),
        cells=dict(
            values=cells,
            fill_color=C["bg"], font=dict(color=C["text"], size=11),
            align="left", line_color=C["border"],
        ),
    ))
    lay(fig, title="Geography Breakdown — All Metrics",
        extra=dict(margin=dict(l=10, r=10, t=60, b=10)))
    return to_html(fig, max(300, len(df) * 28 + 80))


# ─── Chart: Cross-Cube Scatter ────────────────────────────────────────────────

def build_cross_scatter(df):
    if df is None or df.empty:
        return ""
    df = df[df["est_spend"] > 5].copy()
    if "est_roas" not in df.columns or df["est_roas"].sum() == 0:
        return ""

    df = df[df["est_roas"] > 0].copy()
    if df.empty:
        return ""

    df["label"] = df.apply(
        lambda r: f"{r.get('age','?')} {str(r.get('gender','?'))[:1].upper()} | "
                  f"{r.get('publisher_platform','?')} {str(r.get('platform_position','?')).replace('_',' ')} | "
                  f"{r.get('country','?')}",
        axis=1)

    has_purchases = "est_purchases" in df.columns and df["est_purchases"].sum() > 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["est_spend"], y=df["est_roas"],
        mode="markers",
        marker=dict(
            size=df["est_spend"].clip(upper=df["est_spend"].quantile(0.95)) ** 0.5 * 3,
            color=df["est_roas"],
            colorscale=[[0, "#ef4444"], [0.15, "#f59e0b"], [0.4, "#10b981"], [1, "#06b6d4"]],
            opacity=0.7,
            line=dict(width=1, color="rgba(255,255,255,0.12)"),
            colorbar=dict(
                title=dict(text="ROAS (x)", font=dict(color=C["muted"], size=11)),
                tickfont=dict(color=C["muted"]),
            ),
        ),
        text=df["label"],
        customdata=list(zip(
            df.get("est_purchases", [0]*len(df)),
            df.get("est_purchase_value", [0]*len(df)),
            df.get("est_cpa", [0]*len(df)),
        )),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Est. Spend: $%{x:,.0f}<br>"
            "Est. ROAS: %{y:.2f}x<br>"
            + ("Est. Purchases: %{customdata[0]:,.1f}<br>" if has_purchases else "")
            + ("Est. Revenue: $%{customdata[1]:,.0f}<br>" if has_purchases else "")
            + ("Est. CPA: $%{customdata[2]:,.0f}<br>" if has_purchases else "")
            + "<extra></extra>"
        ),
    ))

    fig.add_hline(y=1, line_dash="dash", line_color=C["amber"], opacity=0.4,
                  annotation_text="ROAS Breakeven (1.0x)", annotation_font_color=C["amber"])

    lay(fig, title="N-Dimensional Cube — Every Cell: Age x Gender x Platform x Position x Device x Country",
        xlab="Estimated Spend ($) — log scale", ylab="Estimated ROAS (x) — log scale",
        extra=dict(xaxis=dict(type="log", title="Estimated Spend ($) — log scale", gridcolor=GRID),
                   yaxis=dict(type="log", title="Estimated ROAS (x) — log scale", gridcolor=GRID)))
    return to_html(fig, 550)


# ─── Chart: Delivery Efficiency Scatter ───────────────────────────────────────

def build_delivery_scatter(df):
    """CPM vs CTR by placement, bubble size = spend."""
    if df is None or df.empty:
        return ""
    df = df[df["spend"] > 30].copy()
    if df.empty:
        return ""

    df["calc_cpm"] = df["spend"] / df["impressions"] * 1000
    df["calc_ctr"] = safe(df, "clicks") / df["impressions"] * 100
    df["label"] = df.apply(
        lambda r: f"{r.get('publisher_platform','')} / {str(r.get('platform_position','')).replace('_',' ')}",
        axis=1)

    df = df[(df["calc_cpm"] > 0) & (df["calc_ctr"] > 0)].copy()
    if df.empty:
        return ""

    has_roas = "roas" in df.columns

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["calc_cpm"], y=df["calc_ctr"],
        mode="markers+text",
        marker=dict(
            size=df["spend"] ** 0.4 * 3,
            color=[C["green"] if r > 1 else C["red"] for r in df.get("roas", [0]*len(df))] if has_roas else C["indigo"],
            opacity=0.75,
            line=dict(width=1.5, color="rgba(255,255,255,0.2)"),
        ),
        text=df["label"],
        textposition="top center",
        textfont=dict(size=8, color=C["muted"]),
        customdata=list(zip(
            df["spend"], df["impressions"], safe(df, "clicks"),
            safe(df, "purchase"), df.get("roas", [0]*len(df)),
        )),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "CPM: $%{x:.2f}<br>"
            "CTR: %{y:.2f}%<br>"
            "Spend: $%{customdata[0]:,.0f}<br>"
            "Impressions: %{customdata[1]:,.0f}<br>"
            "Clicks: %{customdata[2]:,.0f}<br>"
            "Purchases: %{customdata[3]:,.0f}<br>"
            "ROAS: %{customdata[4]:.2f}x<extra></extra>"
        ),
    ))
    lay(fig, title="Placement Efficiency: CPM vs CTR (bubble size = spend)",
        xlab="CPM — Cost per 1,000 Impressions ($)", ylab="CTR — Click-Through Rate (%)")
    return to_html(fig, 450)


# ─── Chart: Frequency vs ROAS ─────────────────────────────────────────────────

def build_frequency_chart(df):
    """Bar chart: average frequency by age group."""
    if df is None or df.empty or "frequency" not in df.columns:
        return ""
    df = df[~df["age"].isin(["Unknown"])].copy()
    df = df[~df["gender"].isin(["unknown", "Unknown"])].copy()
    df = df[df["frequency"] > 0].copy()
    if df.empty:
        return ""

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    gender_colors = {"female": C["pink"], "male": C["indigo"]}

    fig = go.Figure()
    for gender in ["female", "male"]:
        gdf = df[df["gender"] == gender].copy()
        gdf["age"] = pd.Categorical(gdf["age"], categories=age_order, ordered=True)
        agg = gdf.groupby("age", observed=True).agg(
            freq=("frequency", "mean"), spend=("spend", "sum")
        ).reset_index()

        fig.add_trace(go.Bar(
            x=agg["age"], y=agg["freq"].round(2),
            name=gender.title(),
            marker_color=gender_colors[gender],
            text=agg["freq"].apply(lambda v: f"{v:.1f}"),
            textposition="outside", textfont=dict(size=10),
            hovertemplate=f"%{{x}} {gender.title()}<br>Avg Frequency: %{{y:.2f}}<br>Spend: $%{{customdata[0]:,.0f}}<extra></extra>",
            customdata=list(zip(agg["spend"])),
        ))

    lay(fig, title="Average Ad Frequency by Age x Gender",
        xlab="Age Group", ylab="Frequency (avg impressions per user)",
        extra=dict(barmode="group"))
    return to_html(fig, 380)


# ─── Account Section Builder ─────────────────────────────────────────────────

def build_account(name, data):
    totals = data.get("slice_totals")
    demo = data.get("pivot_age_gender")
    deliv = data.get("pivot_delivery")
    geo = data.get("pivot_geography")
    cube = data.get("cross_cube_summary")

    parts = []

    # KPIs
    parts.append(build_kpis(totals))

    # Section: Demographics
    h1 = build_demo_heatmaps(demo)
    h2 = build_demo_bars(demo)
    if h1 or h2:
        parts.append('<h3 class="section-label">Demographics</h3>')
        parts.append('<div class="chart-row">')
        if h1:
            parts.append(f'<div class="chart-card">{h1}</div>')
        if h2:
            parts.append(f'<div class="chart-card">{h2}</div>')
        parts.append('</div>')

    h3 = build_demo_scatter(demo)
    h_freq = build_frequency_chart(demo)
    if h3 or h_freq:
        parts.append('<div class="chart-row">')
        if h3:
            parts.append(f'<div class="chart-card">{h3}</div>')
        if h_freq:
            parts.append(f'<div class="chart-card">{h_freq}</div>')
        parts.append('</div>')

    # Section: Placements
    h4 = build_delivery_treemap(deliv)
    h5 = build_delivery_bars(deliv)
    if h4 or h5:
        parts.append('<h3 class="section-label">Placements &amp; Delivery</h3>')
        parts.append('<div class="chart-row">')
        if h4:
            parts.append(f'<div class="chart-card">{h4}</div>')
        if h5:
            parts.append(f'<div class="chart-card">{h5}</div>')
        parts.append('</div>')

    h_ds = build_delivery_scatter(deliv)
    if h_ds:
        parts.append(f'<div class="chart-row"><div class="chart-card full">{h_ds}</div></div>')

    # Section: Geography
    h6 = build_geo_map(geo)
    h7 = build_geo_table(geo)
    if h6 or h7:
        parts.append('<h3 class="section-label">Geography</h3>')
        parts.append('<div class="chart-row">')
        if h6:
            parts.append(f'<div class="chart-card">{h6}</div>')
        if h7:
            parts.append(f'<div class="chart-card">{h7}</div>')
        parts.append('</div>')

    # Section: Cross-dimensional cube
    h8 = build_cross_scatter(cube)
    if h8:
        parts.append('<h3 class="section-label">Cross-Dimensional Cube</h3>')
        parts.append(f'<div class="chart-row"><div class="chart-card full">{h8}</div></div>')

    display_name = name.replace("_", " ")
    inner = "\n".join(parts)
    return f"""
    <section class="account-section" id="{name}">
        <h2 class="account-title">{display_name}</h2>
        {inner}
    </section>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    accounts = ["Rolling_Square_2025", "Hanso_US", "Hanso_US_2nd"]
    account_data = {}
    for name in accounts:
        if (OUTPUT_DIR / name).exists():
            account_data[name] = load_account(name)

    nav = "".join(f'<a href="#{n}" class="nav-link">{n.replace("_", " ")}</a>' for n in account_data)

    # Detect global date range across all accounts
    global_min, global_max, global_days = None, None, 0
    for data in account_data.values():
        t = data.get("slice_totals")
        if t is not None and "date_start" in t.columns:
            d_min, d_max, nd = detect_date_range(t)
            if d_min is not None:
                if global_min is None or d_min < global_min:
                    global_min = d_min
                if global_max is None or d_max > global_max:
                    global_max = d_max
    if global_min and global_max:
        global_days = (global_max - global_min).days + 1
        date_str = f"{global_min.strftime('%b %d')} &ndash; {global_max.strftime('%b %d, %Y')} ({global_days} days)"
    else:
        date_str = "Date range unknown"

    sections = ""
    for name, data in account_data.items():
        print(f"Building {name}...")
        sections += build_account(name, data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meta Performance Cube</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: {C['bg']}; color: {C['text']};
    font-family: 'Inter', -apple-system, sans-serif; line-height: 1.5;
}}
.header {{
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    border-bottom: 1px solid {C['border']}; padding: 40px 40px 30px; text-align: center;
}}
.header h1 {{
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, {C['indigo']}, {C['cyan']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}}
.header .sub {{ color: {C['muted']}; font-size: 0.95rem; }}
.nav {{
    display: flex; justify-content: center; gap: 8px; padding: 14px 40px;
    background: {C['card']}; border-bottom: 1px solid {C['border']};
    position: sticky; top: 0; z-index: 100; backdrop-filter: blur(12px);
}}
.nav-link {{
    color: {C['muted']}; text-decoration: none; padding: 8px 20px;
    border-radius: 8px; font-size: 0.85rem; font-weight: 500;
    transition: all 0.2s; border: 1px solid transparent;
}}
.nav-link:hover {{ color: {C['text']}; background: rgba(99,102,241,0.1); border-color: {C['indigo']}; }}
.container {{ max-width: 1600px; margin: 0 auto; padding: 30px 40px; }}
.account-section {{ margin-bottom: 70px; scroll-margin-top: 80px; }}
.account-title {{
    font-size: 1.5rem; font-weight: 600; margin-bottom: 24px;
    padding-bottom: 12px; border-bottom: 2px solid {C['indigo']}; display: inline-block;
}}
.section-label {{
    font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: {C['muted']}; margin: 28px 0 12px 4px;
    padding-left: 12px; border-left: 3px solid {C['indigo']};
}}
.kpi-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px; margin-bottom: 28px;
}}
.kpi-card {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; padding: 20px; text-align: center; transition: border-color 0.2s;
}}
.kpi-card:hover {{ border-color: {C['indigo']}; }}
.kpi-label {{
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: {C['muted']}; margin-bottom: 6px;
}}
.kpi-value {{ font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }}
.kpi-sub {{ font-size: 0.75rem; color: {C['muted']}; margin-top: 3px; }}
.kpi-row-secondary .kpi-card {{
    background: transparent; border: 1px dashed {C['border']};
    padding: 14px;
}}
.kpi-row-secondary .kpi-value {{ font-size: 1.3rem; }}
.time-banner {{
    display: flex; align-items: center; justify-content: center; gap: 20px;
    background: linear-gradient(90deg, rgba(99,102,241,0.08) 0%, rgba(6,182,212,0.08) 100%);
    border: 1px solid {C['border']}; border-radius: 10px;
    padding: 14px 28px; margin-bottom: 20px; text-align: center;
}}
.time-range {{
    font-size: 1.05rem; font-weight: 600; color: {C['text']};
    letter-spacing: 0.01em;
}}
.time-span {{
    font-size: 0.85rem; font-weight: 500; color: {C['indigo']};
    background: rgba(99,102,241,0.12); padding: 4px 14px;
    border-radius: 20px;
}}
.chart-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;
}}
.chart-card {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; padding: 20px; overflow: hidden;
}}
.chart-card.full {{ grid-column: 1 / -1; }}
@media (max-width: 1000px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
    .container {{ padding: 16px; }}
    .kpi-row {{ grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }}
}}
.footer {{
    text-align: center; padding: 30px; color: {C['muted']};
    font-size: 0.8rem; border-top: 1px solid {C['border']};
}}
</style>
</head>
<body>
<div class="header">
    <h1>Meta Performance Cube</h1>
    <div class="sub">{date_str} &mdash; Age x Gender x Platform x Position x Device x Country<br>Metrics: Spend, Revenue, ROAS, CPA, AOV, CPM, CPC, CTR, Frequency, Impressions, Clicks, Purchases</div>
</div>
<nav class="nav">{nav}</nav>
<div class="container">{sections}</div>
<div class="footer">
    Meta Marketing API &mdash; Multi-slice cube reconstruction via ad-level bridge &mdash; Dropdown toggles for metric switching
</div>
</body>
</html>"""

    DASH_PATH.write_text(html)
    print(f"\nDashboard: {DASH_PATH}")
    print(f"Size: {DASH_PATH.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
