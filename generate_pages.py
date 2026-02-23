#!/usr/bin/env python3
"""Generate separate per-account dashboard pages for deployment."""

import sys
sys.path.insert(0, ".")
from build_dashboard import *

def generate_single_account(name, data, out_path):
    d_min, d_max, n_days = None, None, 0
    t = data.get("slice_totals")
    if t is not None and "date_start" in t.columns:
        d_min, d_max, n_days = detect_date_range(t)
    if d_min and d_max:
        date_str = f"{d_min.strftime('%b %d')} &ndash; {d_max.strftime('%b %d, %Y')} ({n_days} days)"
    else:
        date_str = "Date range from extraction"

    display = name.replace("_", " ")
    section = build_account(name, data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{display} — Performance Cube</title>
<meta property="og:title" content="{display} — Meta Performance Cube">
<meta property="og:description" content="N-dimensional ad performance: Age x Gender x Platform x Position x Device x Country. {date_str.replace('&ndash;', '-').replace('&mdash;', '-')}">
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
    background: transparent; border: 1px dashed {C['border']}; padding: 14px;
}}
.kpi-row-secondary .kpi-value {{ font-size: 1.3rem; }}
.time-banner {{
    display: flex; align-items: center; justify-content: center; gap: 20px;
    background: linear-gradient(90deg, rgba(99,102,241,0.08) 0%, rgba(6,182,212,0.08) 100%);
    border: 1px solid {C['border']}; border-radius: 10px;
    padding: 14px 28px; margin-bottom: 20px; text-align: center;
}}
.time-range {{ font-size: 1.05rem; font-weight: 600; color: {C['text']}; }}
.time-span {{
    font-size: 0.85rem; font-weight: 500; color: {C['indigo']};
    background: rgba(99,102,241,0.12); padding: 4px 14px; border-radius: 20px;
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
    <h1>{display}</h1>
    <div class="sub">{date_str} &mdash; Age x Gender x Platform x Position x Device x Country<br>Metrics: Spend, Revenue, ROAS, CPA, AOV, CPM, CPC, CTR, Frequency, Impressions, Clicks, Purchases</div>
</div>
<div class="container">
    {section}
</div>
<div class="footer">
    Meta Marketing API &mdash; Multi-slice cube reconstruction via ad-level bridge &mdash; Dropdown toggles for metric switching
</div>
</body>
</html>"""

    out_path.write_text(html)
    print(f"  {out_path.name}: {out_path.stat().st_size / 1024:.0f} KB")


def main():
    deploy_dir = Path(__file__).parent / "deploy"
    deploy_dir.mkdir(exist_ok=True)

    targets = {
        "Rolling_Square_2025": "rolling-square",
        "Hanso_US": "hanso-us",
    }

    # Index page
    links = ""
    for name, slug in targets.items():
        display = name.replace("_", " ")
        data = load_account(name)
        out = deploy_dir / f"{slug}.html"
        print(f"Generating {display}...")
        generate_single_account(name, data, out)
        links += f'<a href="{slug}.html" class="page-link">{display}</a>\n'

    index = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meta Performance Cube</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: {C['bg']}; color: {C['text']};
    font-family: 'Inter', -apple-system, sans-serif;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
}}
.container {{ text-align: center; }}
h1 {{
    font-size: 2.5rem; font-weight: 700; margin-bottom: 12px;
    background: linear-gradient(135deg, {C['indigo']}, {C['cyan']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.sub {{ color: {C['muted']}; margin-bottom: 40px; }}
.page-link {{
    display: block; color: {C['text']}; text-decoration: none;
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; padding: 24px 48px; margin: 12px auto;
    max-width: 400px; font-size: 1.1rem; font-weight: 500;
    transition: all 0.2s;
}}
.page-link:hover {{
    border-color: {C['indigo']}; background: rgba(99,102,241,0.08);
    transform: translateY(-2px);
}}
</style>
</head>
<body>
<div class="container">
    <h1>Meta Performance Cube</h1>
    <div class="sub">N-Dimensional Ad Performance Dashboards</div>
    {links}
</div>
</body>
</html>"""

    (deploy_dir / "index.html").write_text(index)
    print(f"\nDeploy directory: {deploy_dir}")
    print(f"Files: {list(f.name for f in deploy_dir.glob('*.html'))}")


if __name__ == "__main__":
    main()
