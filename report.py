"""
report.py — generate a self-contained static HTML dashboard (report.html)
showing the current US equity bubble risk (gauge + 8 features + S&P/Nasdaq
history) and the Buy-&-Hold vs Bubble-DCA backtest.  No web server required:
just open report.html in any browser.

This is the "open-and-read" companion to the live Streamlit app (app.py).
It reuses pipeline.py + backtest.py, so the numbers are identical.

Run:
    python report.py            # uses cached / live data
    python report.py --refresh  # force re-fetch of score + price history
"""

from __future__ import annotations

import argparse
import html as _html

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pipeline as pipe
import backtest as bt

REF_2000 = 83.2   # historical reference (Dalio-style framing)
REF_2021 = 92.1

BANDS = [  # (lo, hi, color, label)
    (0, 40, "forestgreen", "Low / Cooling"),
    (40, 60, "gold", "Normal"),
    (60, 80, "darkorange", "Watch"),
    (80, 90, "crimson", "Elevated"),
    (90, 100, "darkred", "Bubble Warning"),
]

# Plain CSS (NOT an f-string, so the braces stay literal).
_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#222}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 4px}
.cap{color:#777;font-size:13px;margin-bottom:18px}
.top{display:flex;gap:18px;flex-wrap:wrap;align-items:stretch}
.box{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;flex:1;min-width:260px}
.status{font-size:20px;font-weight:700;margin:8px 0}
.kv{font-size:13px;color:#555;line-height:1.7}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:6px}
.card{background:#fafafa;border:1px solid #ddd;border-radius:10px;padding:12px;text-align:center}
.lbl{font-size:12px;color:#555}
.val{font-size:30px;font-weight:700}
.w{font-size:11px;color:#888}
.tbl{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}
.tbl th,.tbl td{border:1px solid #e5e7eb;padding:8px 10px;text-align:left}
.tbl th{background:#f0f2f5}
h2{margin-top:28px;font-size:18px}
.warn{background:#fff7ed;border:1px solid #fdba74;color:#9a3412;padding:10px 14px;border-radius:8px;margin:10px 0;font-size:13px}
/* phones: shrink padding, stack the top row, let cards reflow */
@media (max-width:640px){
  .wrap{padding:14px}
  .top{flex-direction:column}
  .box{min-width:0}
  .grid{grid-template-columns:repeat(auto-fit,minmax(135px,1fr))}
  h1{font-size:20px}
}
"""


def band_color(score: float) -> str:
    if pd.isna(score):
        return "gray"
    for lo, hi, c, _ in BANDS:
        if lo <= score < hi:
            return c
    return "darkred"


def gauge_div(score: float) -> str:
    steps = [{"range": [lo, hi], "color": c} for lo, hi, c, _ in BANDS]
    v = 0.0 if pd.isna(score) else float(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=v,
        number={"font": {"size": 56}, "suffix": " / 100"},
        delta={"reference": REF_2021, "increasing": {"color": "crimson"},
               "decreasing": {"color": "forestgreen"}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1},
               "bar": {"color": band_color(score)},
               "steps": steps,
               "threshold": {"line": {"color": "black", "width": 3}, "value": v}},
        title={"text": "US Equity Bubble Risk Score", "font": {"size": 20}},
    ))
    fig.add_annotation(x=0.5, y=-0.05,
                       text=f"Refs — 2000: {REF_2000}  |  2021: {REF_2021}",
                       showarrow=False, font={"size": 12, "color": "gray"})
    fig.update_layout(height=380, margin={"t": 50, "b": 30, "l": 30, "r": 30})
    # First plotly div carries the CDN script; later divs set include_plotlyjs=False.
    # responsive:True makes the chart resize with the viewport (phone + desktop).
    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"responsive": True})


def history_div(scores: pd.Series, spx, ndx) -> str:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("S&P 500 / Nasdaq Composite (log)", "Bubble Risk Score"),
    )
    if spx is not None:
        fig.add_trace(go.Scatter(x=spx.index, y=spx.values, name="S&P 500",
                                 line={"color": "steelblue"}), row=1, col=1)
    if ndx is not None:
        fig.add_trace(go.Scatter(x=ndx.index, y=ndx.values, name="Nasdaq",
                                 line={"color": "purple"}), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    sc = scores.dropna()
    if not sc.empty:
        fig.add_trace(go.Scatter(x=sc.index, y=sc.values, name="Bubble Score",
                                 line={"color": "crimson", "width": 2},
                                 fill="tozeroy", fillcolor="rgba(220,20,60,0.08)"),
                      row=2, col=1)
        fig.add_hrect(y0=80, y1=100, line_width=0, fillcolor="rgba(220,20,60,0.12)",
                      row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="crimson",
                      annotation_text="High-risk > 80", row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified", showlegend=True,
                      margin={"t": 40, "b": 30, "l": 50, "r": 30})
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True})


def feature_cards_html(state: dict) -> str:
    cells = ""
    for key, f in state["features"].items():
        sc = f["score"]
        if sc is None:
            color, txt = "gray", "n/a"
        else:
            color = band_color(sc)
            txt = f"{sc:.0f}"
        avail = "live" if f["available"] else "no data"
        cells += (
            f'<div class="card">'
            f'<div class="lbl">{_html.escape(f["label"])}</div>'
            f'<div class="val" style="color:{color}">{txt}</div>'
            f'<div class="w">weight {f["weight"]:.2f} · {avail}</div>'
            f'</div>'
        )
    return f'<div class="grid">{cells}</div>'


def backtest_html(res: dict) -> str:
    mb = res["benchmark"]
    ms = res["strategy"]
    rows = [
        ("Cumulative", f"{mb['cum_return']*100:.1f}%", f"{ms['cum_return']*100:.1f}%"),
        ("CAGR", f"{mb['cagr']*100:.1f}%", f"{ms['cagr']*100:.1f}%"),
        ("Max Drawdown", f"{mb['max_drawdown']*100:.1f}%", f"{ms['max_drawdown']*100:.1f}%"),
        ("Sharpe", f"{mb['sharpe']:.2f}", f"{ms['sharpe']:.2f}"),
        ("Calmar", f"{mb['calmar']:.2f}", f"{ms['calmar']:.2f}"),
        ("Ending Value", f"${mb['end_value']:,.0f}", f"${ms['end_value']:,.0f}"),
    ]
    body = "".join(
        f"<tr><td>{n}</td><td>{b}</td><td>{s}</td></tr>" for n, b, s in rows
    )
    tops = res.get("tops", {})
    trows = ""
    for name, t in tops.items():
        if t:
            trows += (
                f"<tr><td>{_html.escape(name)}</td>"
                f"<td>{t['bench_mdd']*100:.1f}%</td>"
                f"<td>{t['strat_mdd']*100:.1f}%</td>"
                f"<td>{t['avoided_pp']:.1f} pp</td></tr>"
            )
    return (
        '<table class="tbl"><tr><th>Metric</th><th>Benchmark</th>'
        '<th>Bubble-DCA</th></tr>' + body + '</table>'
        "<h3>Drawdown during classic tops (strategy vs benchmark)</h3>"
        '<table class="tbl"><tr><th>Top</th><th>Bench MDD</th>'
        '<th>Strategy MDD</th><th>Avoided</th></tr>' + trows + '</table>'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="force re-fetch of score + price history")
    args = ap.parse_args()

    scores, meta = pipe.get_monthly_scores(refresh=args.refresh)
    state = pipe.get_latest_state(refresh=args.refresh)
    spx = pipe.get_price_series("^GSPC", start=pipe.LIVE_START)
    ndx = pipe.get_price_series("^IXIC", start=pipe.LIVE_START)
    res = bt.main(refresh=args.refresh)

    score = state["score"]
    asof = state["as_of"]
    asof_str = pd.Timestamp(asof).date() if asof else "n/a"
    src = meta.get("source", "unknown")

    warn = (
        '<div class="warn">⚠️ Live data unavailable — showing a deterministic '
        'synthetic series for layout/demo only. Set FRED_API_KEY in .env and '
        're-run with --refresh.</div>'
    ) if src == "synthetic" else ""

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>US Equity Bubble Risk Report</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>📈 US Equity Bubble Risk — Dalio-style Report</h1>
<div class="cap">8-feature percentile model · rolling 20-year window · generated {pd.Timestamp.now().date()}</div>
{warn}
<div class="top">
  <div class="box">{gauge_div(score)}</div>
  <div class="box">
    <div class="status">{_html.escape(state['status'])}</div>
    <div class="kv"><b>Score:</b> {score:.1f} / 100 (as of {asof_str})</div>
    <div class="kv"><b>DCA rule:</b> 0–40: 2.0× · 40–60: 1.5× · 60–80: 1.0× · 80–90: 0.5× · 90–100: 0×</div>
    <div class="kv"><b>Data source:</b> {src} · features live: {meta.get('available_count','?')}/8</div>
  </div>
</div>
<h2>Eight Risk Features (current percentile)</h2>
{feature_cards_html(state)}
<h2>Historical Trend</h2>
{history_div(scores, spx, ndx)}
<h2>Backtest: Buy &amp; Hold vs Bubble-DCA ({res['freq']})</h2>
{backtest_html(res)}
</div></body></html>"""

    out = "report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Wrote {out}  (source={src}, score={score:.1f}, as_of={asof_str})")


if __name__ == "__main__":
    main()
