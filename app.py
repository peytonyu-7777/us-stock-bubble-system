"""
app.py — Streamlit dashboard for the Dalio-style US Equity Bubble Risk Score.

Sections
   1. Top   : refined gauge (current score) + live Status Card (strategy action
              badge, coverage, elevated-feature callouts).
   2. Middle: 8 feature cards (dynamic colour by percentile band, crimson pulse
              when in the >80 danger zone, greyed "Pending" when missing).
   3. Bottom: dual-axis S&P 500 / Nasdaq main chart (log/linear toggle) linked
              to a 30-day-EMA-smoothed Bubble Risk Score history with risk-zone
              shading and a shared, zoom-linked time axis (hovermode="x unified").
   4. Backtest: interactive Bubble Risk-Adjusted DCA vs Buy & Hold, with
              user-tunable sliders and de-risk markers. The engine lives in
              backtest.py and returns (metrics_df, chart_fig) or None — the
              panel shows a friendly card if data is unavailable.

Deploy: `streamlit run app.py`  (Render / HuggingFace Spaces ready).
Live data needs a FRED_API_KEY env var; without it the app still runs on cache
or, as a last resort, a clearly-flagged synthetic series.
"""

from __future__ import annotations

import os
import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional

import pipeline as pipe
import backtest as bt

st.set_page_config(page_title="US Equity Bubble Risk", page_icon="📈",
                   layout="wide", initial_sidebar_state="auto")

# --- Modern finance-terminal styling (soft-shadow cards, responsive) --------
st.markdown("""
<style>
/* fluid main container (caps at 1320 on wide desktops, shrinks on mobile) */
.main .block-container{max-width:1320px;padding-top:1.4rem;padding-bottom:2rem}

/* ----- responsive layout for small / mid screens ----- */
@media (max-width:720px){
  .stHorizontalBlock{flex-direction:column !important}
  .stHorizontalBlock>div{width:100% !important;min-width:0 !important}
  .main .block-container{padding-left:.5rem;padding-right:.5rem}
  .status-card{padding:14px 14px}
  .feat-grid{grid-template-columns:repeat(2,1fr)}
}
@media (max-width:480px){
  .feat-grid{grid-template-columns:1fr 1fr}
  .feat-card{padding:10px 6px}
  .feat-val{font-size:24px}
}

/* Trim oversized Plotly tooltips/menus on touch devices */
@media (max-width:720px){
  .modebar-group{display:none}
  .js-plotly-plot .plotly .main-svg{font-size:11px}
}

/* Scrollable tables on mobile (backtest metrics can be wide) */
.stDataFrame>div{overflow-x:auto}

/* Make sure legend / modebar don't collide with the chart on small viewports */
.legend{font-size:11px}

/* 8 feature cards: rounded, soft shadow, hover lift */
.feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin-top:8px}
.feat-card{border:2px solid #e2e8f0;border-radius:12px;padding:14px 10px;
  background:#ffffff;text-align:center;
  box-shadow:0 4px 14px rgba(15,23,42,0.06);
  transition:transform .15s ease, box-shadow .15s ease}
.feat-card:hover{transform:translateY(-3px);box-shadow:0 8px 22px rgba(15,23,42,0.10)}
.feat-card.missing{opacity:.55;background:#f8fafc;border-style:dashed}
.feat-card.alert{animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{
  0%,100%{box-shadow:0 0 0 0 rgba(193,18,31,0.35)}
  50%{box-shadow:0 0 0 9px rgba(193,18,31,0)}
}
.feat-lbl{font-size:12px;color:#475569;font-weight:600;line-height:1.2}
.feat-val{font-size:30px;font-weight:800;margin:4px 0;letter-spacing:-.5px}
.feat-w{font-size:11px;color:#94a3b8}

/* status card */
.status-card{border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px;
  background:linear-gradient(135deg,#ffffff,#f8fafc);
  box-shadow:0 6px 20px rgba(15,23,42,0.08);height:100%;
  line-height:1.55}
.status-head{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
  color:#64748b;font-weight:700}
.badge{display:inline-block;margin:10px 0;padding:11px 16px;border-radius:11px;
  color:#fff;font-weight:800;font-size:18px;letter-spacing:.01em;
  box-shadow:0 4px 12px rgba(15,23,42,0.18)}
.status-score{font-size:17px;color:#0f172a;font-weight:700}
.status-note{font-size:13px;color:#475569;margin-top:8px;line-height:1.6}
.status-cov{font-size:12px;color:#94a3b8;margin-top:8px}
.status-elev{font-size:12px;color:#c1121f;font-weight:700;margin-top:8px}

/* gauge wrapper */
.gauge-wrap{border:1px solid #e2e8f0;border-radius:14px;padding:6px 6px 0;
  box-shadow:0 6px 20px rgba(15,23,42,0.08);background:#fff}
</style>
""", unsafe_allow_html=True)

from pipeline import RISK_BANDS as BANDS  # risk-level bands (0-40 .. 90-100)
from pipeline import MODULE_WEIGHTS       # 5-module weights for the radar/labels


def band_color(score: float) -> str:
    if pd.isna(score):
        return "gray"
    for lo, hi, c, _ in BANDS:
        if lo <= score < hi:
            return c
    return "#991b1b"


def status_action(score: float):
    """Return (badge_text, badge_color, note) for the Status Card (risk bands)."""
    if pd.isna(score):
        return ("UNKNOWN", "#64748b", "No data")
    if score < 40:
        return ("CHEAP / FEAR", "#1a9850", "Below-median risk — market fear")
    if score < 60:
        return ("NORMAL", "#2166ac", "Balanced — no excess")
    if score < 75:
        return ("EXPENSIVE", "#f4a01c", "Elevated valuation building")
    if score < 90:
        return ("BUBBLE RISK", "#e4572e", "Risk accumulation — trim exposure")
    return ("EXTREME BUBBLE", "#c1121f", "De-risk / defensive")


@st.cache_data(ttl=3600)
def load_scores(refresh: bool, tail_boost: bool):
    return pipe.get_monthly_scores(refresh=refresh, tail_boost=tail_boost)


@st.cache_data(ttl=3600)
def load_daily_scores(refresh: bool, tail_boost: bool):
    return pipe.get_daily_scores(refresh=refresh, tail_boost=tail_boost)


@st.cache_data(ttl=3600)
def load_prices():
    """DAILY price series for the combined chart (cache-first). The daily
    cache is refreshed incrementally by the pipeline's auto-refresh, so the
    S&P 500 / Nasdaq lines reach the latest trading day — not the last
    month-end."""
    spx = pipe.get_daily_price("^GSPC")
    ndx = pipe.get_daily_price("^IXIC")
    if spx is None:
        spx = pipe.get_price_series("^GSPC", start=pipe.LIVE_START)
    if ndx is None:
        ndx = pipe.get_price_series("^IXIC", start=pipe.LIVE_START)
    return spx, ndx


@st.cache_data(ttl=3600)
def load_spy():
    """SPY monthly close for the backtest — CACHE-ONLY (never triggers network
    from a user request). Falls back to the S&P 500 index level from the same
    cache (FRED-backed) when the SPY column is absent, so the backtest panel
    stays usable even during a total Yahoo/Stooq outage."""
    try:
        s = pipe.get_price_series("SPY", start=pipe.LIVE_START)
        if s is not None and not s.dropna().empty:
            return s
        s = pipe.get_price_series("^GSPC", start=pipe.LIVE_START)
        if s is not None and not s.dropna().empty:
            return s
    except Exception:
        pass
    return None


def gauge_fig(score: float) -> go.Figure:
    """CNN Fear & Greed style semicircular gauge.

    Layout is fully de-coupled from the needle geometry: the coloured arc lives
    in the half-disk *above* the pivot hub (cy = 0.25) while the big numeric
    read-out and the status label are pinned *below* the hub via Plotly
    annotations, so the needle can sweep 0..100 without ever touching the text.

    Arc: outer radius 1.0, inner radius 0.70. Needle pivot (0.5, 0.25),
    length 0.55 (tip stays inside the band). Angle θ = π·(1 − Score/100):
    0 → 180° (left), 50 → 90° (up), 100 → 0° (right).
    """
    if pd.isna(score):
        score = 0.0
    score = max(0.0, min(100.0, score))

    cx, cy = 0.5, 0.25          # pivot hub
    R_out, R_in = 1.0, 0.70     # outer / inner arc radii
    R_needle = 0.55             # needle length (< R_in -> tip inside the band)

    # --- 5-step color bands aligned with the risk-level bands ---------------
    STEPS = [
        (0, 40, "#10b981", "Cheap / Fear"),
        (40, 60, "#3b82f6", "Normal"),
        (60, 75, "#f59e0b", "Expensive"),
        (75, 90, "#ef4444", "Bubble Risk"),
        (90, 100, "#991b1b", "Extreme Bubble"),
    ]

    fig = go.Figure()

    # --- filled annular arc segments ---------------------------------------
    n = 60
    for lo, hi, color, _ in STEPS:
        xs, ys = [], []
        for i in range(n + 1):                      # outer arc, lo -> hi
            v = lo + (hi - lo) * i / n
            th = math.pi * (1 - v / 100.0)
            xs.append(cx + R_out * math.cos(th))
            ys.append(cy + R_out * math.sin(th))
        for i in range(n + 1):                      # inner arc, hi -> lo
            v = hi + (lo - hi) * i / n
            th = math.pi * (1 - v / 100.0)
            xs.append(cx + R_in * math.cos(th))
            ys.append(cy + R_in * math.sin(th))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", fillcolor=color,
            line=dict(width=0), hoverinfo="skip", showlegend=False))

    # --- tick numbers: inside the inner radius, gray, clear of the needle ---
    for v in (0, 20, 40, 60, 80, 100):
        th = math.pi * (1 - v / 100.0)
        tx = cx + 0.66 * math.cos(th)
        ty = cy + 0.66 * math.sin(th)
        fig.add_annotation(x=tx, y=ty, text=str(v), showarrow=False,
                           font=dict(size=11, color="#94a3b8"))

    # --- needle (tapered triangle) -----------------------------------------
    th = math.pi * (1 - score / 100.0)
    tipx = cx + R_needle * math.cos(th)
    tipy = cy + R_needle * math.sin(th)
    px = -math.sin(th) * 0.018          # perpendicular base half-width
    py = math.cos(th) * 0.018
    fig.add_trace(go.Scatter(
        x=[cx - px, cx + px, tipx], y=[cy - py, cy + py, tipy],
        fill="toself", fillcolor="#1e293b", line=dict(width=0),
        hoverinfo="skip", showlegend=False))

    # --- pivot hub: dark disc with white rim -------------------------------
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy], mode="markers",
        marker=dict(size=15, color="#1e293b", line=dict(color="#ffffff", width=2)),
        hoverinfo="skip", showlegend=False))

    # --- numeric read-out + status, anchored BELOW the hub -----------------
    action, color, _ = status_action(score)
    fig.add_annotation(x=0.5, y=0.135, text=f"{score:.1f}", showarrow=False,
                       font=dict(size=46, color="#0f172a",
                                 family="Arial Black, Arial, sans-serif"))
    fig.add_annotation(x=0.5, y=0.055, text=action, showarrow=False,
                       font=dict(size=15, color=color,
                                 family="Arial, sans-serif"))
    fig.add_annotation(x=0.5, y=0.005, text="/ 100", showarrow=False,
                       font=dict(size=12, color="#94a3b8"))
    fig.add_annotation(x=0.5, y=1.345,
                       text="US Equity Bubble Risk Score", showarrow=False,
                       font=dict(size=16, color="#0f172a",
                                 family="Arial, sans-serif"))

    # --- square-scaled, axis-less, generous margins ------------------------
    fig.update_layout(
        height=400, margin=dict(t=30, b=8, l=8, r=8),
        paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
        xaxis=dict(visible=False, range=[-0.58, 1.58],
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-0.10, 1.44]),
    )
    return fig


def status_card(state: dict, meta: dict, src_label: str, tail_boost: bool = True):
    score = state["score"]
    action, color, note = status_action(score)
    avail = meta.get("available_count", "?")
    modules = state.get("modules", {}) or {}
    hist_pct = state.get("hist_pct", np.nan)
    # Top contributing modules (for the "Compared with history" callout)
    mod_rank = sorted(((m, v) for m, v in modules.items() if v is not None),
                      key=lambda x: x[1], reverse=True)
    mod_html = " · ".join(
        f"{m.capitalize()} {v:.0f}" for m, v in mod_rank[:3])
    hist_txt = (f"higher than {hist_pct:.0f}% of history"
                if pd.notna(hist_pct) else "n/a")
    asof = (pd.Timestamp(state["as_of"]).date() if state.get("as_of") else "n/a")
    st.markdown(f"""
    <div class="status-card">
      <div class="status-head">Current Bubble Risk Index</div>
      <div style="display:flex;align-items:center;gap:12px;margin:4px 0 2px">
        <div class="badge" style="background:{color};margin:0">{action}</div>
        <div class="status-score" style="margin:0"><b>{score:.1f}</b> / 100</div>
      </div>
      <div class="status-cov" style="margin-top:2px">as of {asof} · {src_label} data · coverage {avail}/8</div>
      <div class="status-note">{note}</div>
      <table style="width:100%;font-size:12.5px;color:#475569;border-collapse:collapse;margin-top:6px">
        <tr><td style="padding:3px 0">vs history</td>
            <td style="padding:3px 0;text-align:right;font-weight:600">{hist_txt}</td></tr>
        <tr><td style="padding:3px 0">top modules</td>
            <td style="padding:3px 0;text-align:right;font-weight:600">{mod_html or 'n/a'}</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)


MODULE_LABELS = {
    "valuation": "Valuation",
    "sentiment": "Sentiment",
    "leverage": "Leverage",
    "structure": "Structure",
    "macro": "Macro",
}


def radar_fig(modules: dict) -> go.Figure:
    """5-axis radar of the module risk scores (0-100 each)."""
    cats = [MODULE_LABELS[m] for m in MODULE_WEIGHTS]
    vals = [modules.get(m, 0.0) or 0.0 for m in MODULE_WEIGHTS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(193,18,31,0.18)", line=dict(color="#c1121f", width=2),
        marker=dict(size=6, color="#c1121f"),
        hovertemplate="%{theta}: %{r:.0f}<extra></extra>"))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=10),
                                    gridcolor="#e2e8f0"),
                    angularaxis=dict(tickfont=dict(size=12, color="#0f172a"))),
        height=360, margin=dict(t=30, b=20, l=40, r=40),
        paper_bgcolor="white", showlegend=False)
    return fig


def module_cards(modules: dict):
    """Five module score cards (weight-bearing) below the radar."""
    cells = ""
    for m in MODULE_WEIGHTS:
        sc = modules.get(m)
        if sc is None:
            border, color, txt, cls = "#d1d5db", "gray", "—", " missing"
        else:
            color = band_color(sc)
            border = color
            txt = f"{sc:.0f}"
            cls = " alert" if sc >= 75 else ""
        cells += (
            f'<div class="feat-card{cls}" style="border-color:{border}">'
            f'<div class="feat-lbl">{MODULE_LABELS[m]}</div>'
            f'<div class="feat-val" style="color:{color}">{txt}</div>'
            f'<div class="feat-w">w {MODULE_WEIGHTS[m]:.2f} · risk score</div>'
            f'</div>'
        )
    st.markdown(f'<div class="feat-grid">{cells}</div>', unsafe_allow_html=True)


def drivers_panel(state: dict):
    """Explain what moved the index vs last month (module deltas)."""
    drivers = state.get("drivers", []) or []
    score = state.get("score", np.nan)
    if not drivers:
        st.info("Module detail updates monthly; no month-over-month change yet.")
        return
    rows = "".join(
        f'<tr><td style="padding:4px 10px"><b>{MODULE_LABELS.get(d["module"], d["module"])}</b></td>'
        f'<td style="padding:4px 10px;text-align:right;color:'
        f'{"#c1121f" if d["delta"] > 0 else "#1a9850"}">'
        f'{"+" if d["delta"] >= 0 else ""}{d["delta"]:.1f}</td>'
        f'<td style="padding:4px 10px;text-align:right;color:#64748b">w {d["weight"]:.2f}</td></tr>'
        for d in drivers[:5])
    st.markdown(f"""
    <div class="status-card" style="margin-top:10px">
      <div class="status-head">Monthly Drivers (vs prior month)</div>
      <table style="width:100%;font-size:13px;color:#0f172a">
        <tr><th style="text-align:left;padding:2px 10px">Module</th>
        <th style="text-align:right;padding:2px 10px">Δ</th>
        <th style="text-align:right;padding:2px 10px">weight</th></tr>
        {rows}
      </table>
      <div class="status-note" style="margin-top:6px">Index measures risk
      accumulation — not a crash forecast. A rising score means price is
      diverging further from fundamentals / leverage &amp; euphoria are building.</div>
    </div>
    """, unsafe_allow_html=True)


def guidance_panel(state: dict, daily: pd.Series, monthly: pd.Series = None):
    """Actionable guidance: current zone -> posture, anchored to the detected
    historical risk climaxes and accumulation windows (the 'what do I do now'
    panel, incl. the late-2022-style deep-fear buying opportunity)."""
    score = state.get("score", np.nan)
    if pd.isna(score):
        return
    if score < 35:
        posture, color, note = (
            "ACCUMULATE AGGRESSIVELY · 3× base this month", "#1a9850",
            "Deep-fear zone — historically the STRONGEST forward 12–24m return "
            "window (2002-09, 2009-03, 2020-03, 2022-10). Invest 3× your base.")
    elif score < 40:
        posture, color, note = (
            "ACCUMULATE · 3× base this month", "#1a9850",
            "Cheap / fear zone — risk is below median; invest 3× your base.")
    elif score < 50:
        posture, color, note = (
            "STEADY ACCUMULATION · 1.5× base this month", "#2166ac",
            "Below-median risk — keep buying, put in 1.5× your base.")
    elif score < 80:
        posture, color, note = (
            "BASE PACE · 1.0× DCA", "#f4a01c",
            "Valuation elevated — invest your base amount only, no new aggression.")
    elif score < 95:
        posture, color, note = (
            "TAPER · 0.5× base this month", "#e4572e",
            "Bubble-risk zone — invest half your base; skip the rest.")
    else:
        posture, color, note = (
            "DE-RISK · 0× + 15% to cash sleeve", "#c1121f",
            "Extreme-bubble zone — prioritise capital preservation.")

    # Last detected risk climax + accumulation window from the daily series.
    last_risk = last_opp = None
    try:
        ev = pipe.detect_events(daily) if daily is not None else {"risk": [], "opportunity": []}
        if ev.get("risk"):
            d, v = ev["risk"][-1]
            last_risk = f"{pd.Timestamp(d).strftime('%Y-%m')} · score {v:.0f}"
        if ev.get("opportunity"):
            d, v = ev["opportunity"][-1]
            last_opp = f"{pd.Timestamp(d).strftime('%Y-%m')} · score {v:.0f}"
    except Exception:
        pass

    # Confirmation-style signals (sustained / rapid) + historical grounding.
    sig_html = ""
    try:
        if monthly is not None and not monthly.dropna().empty:
            sig = pipe.detect_signals(monthly.dropna())
            sts = pipe.signal_stats(monthly.dropna())
            _b = sts.get("_benchmark", {})
            bench12 = _b.get("fwd12")
            rows = []
            if sig["sell"]:
                d, v = sig["sell"][-1]
                rows.append(("最近卖出信号", f"{pd.Timestamp(d).strftime('%Y-%m')} · {v:.0f}"))
            if sig["buy"]:
                d, v, k = sig["buy"][-1]
                lbl = "快速下滑" if k == "rapid" else "持续低位"
                rows.append(("最近买入信号", f"{pd.Timestamp(d).strftime('%Y-%m')} · {v:.0f} ({lbl})"))
            def _pct(x):
                return "—" if x is None else f"{x*100:+.1f}%"
            stats_line = (f"历史回测: 卖出信号后12个月 {_pct(sts.get('sell',{}).get('fwd12'))} "
                          f"(全期基准 {_pct(bench12)}) · 买入后12个月 "
                          f"{_pct(sts.get('buy_sustained',{}).get('fwd12'))}~{_pct(sts.get('buy_rapid',{}).get('fwd12'))}")
            rows.append(("信号历史统计", stats_line))
            rows_html = "".join(
                f'<tr><td style="padding:3px 10px">{k}</td>'
                f'<td style="padding:3px 10px;text-align:right;font-weight:600">{v}</td></tr>'
                for k, v in rows)
            sig_html = (f'<table style="width:100%;font-size:12.5px;color:#0f172a;'
                        f'border-collapse:collapse;margin-top:6px">{rows_html}</table>')
    except Exception:
        pass

    st.markdown(f"""
    <div class="status-card" style="margin-top:4px">
      <div class="status-head">🧭 Current Guidance 当前操作指引</div>
      <div style="font-size:20px;font-weight:800;color:{color};
                  margin:6px 0 4px">{posture}</div>
      <div class="status-note">{note}</div>
      <table style="width:100%;font-size:13px;color:#0f172a;margin-top:8px">
        <tr><td style="padding:3px 10px">Last risk climax 最近风险事件</td>
            <td style="padding:3px 10px;text-align:right">{last_risk or "—"}</td></tr>
        <tr><td style="padding:3px 10px">Last buying window 最近买入机会</td>
            <td style="padding:3px 10px;text-align:right">{last_opp or "—"}</td></tr>
      </table>
      {sig_html}
      <div class="status-note" style="margin-top:6px">When the index falls into
      the deep-fear zone (≤35) it has historically marked the best accumulation
      windows — e.g. the <b>late-2022 bear bottom</b>. When it pushes above 80 it
      has preceded every major drawdown. The plan keeps your monthly outflow
      constant and only re-times deployment: stockpile at highs, deploy at lows.
      This is risk-accumulation guidance, not a crash prediction.</div>
    </div>
    """, unsafe_allow_html=True)


def history_fig(scores: pd.Series, spx, ndx, log_scale: bool = True,
                view: str = "all") -> go.Figure:
    """Combined Bubble Index chart.

    Single canvas: Bubble Index (left axis, 0-100 with coloured risk bands)
    overlaid on S&P 500 and Nasdaq Composite (right axis, rebased to 100 at
    the START of the selected window so both indices are directly
    comparable). Time range is supplied pre-sliced by the caller — this
    function does NOT add its own period selector.
    """
    _ = log_scale  # kept in signature for API stability; the combined view uses
    #               a linear rebased right axis that already makes log moot.

    sc = scores.dropna()
    # Rebase SPX/NDX to 100 at the FIRST date of the (already sliced) window so
    # both indices start from the same baseline — makes their relative runs
    # directly comparable.
    def _rebase(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return s
        return s / float(s.iloc[0]) * 100.0

    spx_r = _rebase(spx) if spx is not None and not spx.empty else None
    ndx_r = _rebase(ndx) if ndx is not None and not ndx.empty else None

    # No forced common grid: Plotly overlays mixed frequencies fine, and the
    # daily Bubble Index keeps its full daily detail.
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ---- Coloured risk bands on the LEFT (Bubble Index) axis -------------
    # add_hrect binds to the subplot's PRIMARY axis (yref="y" = the 0-100
    # Bubble Index axis). exclude_empty_subplots=False is CRITICAL: the
    # default True silently DROPS the shape when no trace exists yet
    # (Plotly 6.x behaviour — this made the bands invisible).
    band_tints = [
        (0, 40,  "rgba(16,185,129,0.28)", "Cheap / Fear"),
        (40, 60, "rgba(59,130,246,0.16)", "Normal"),
        (60, 75, "rgba(245,158,11,0.22)", "Expensive"),
        (75, 90, "rgba(239,68,68,0.22)",  "Bubble Risk"),
        (90,100, "rgba(153,27,27,0.30)",  "Extreme Bubble"),
    ]
    for lo, hi, color, _ in band_tints:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0,
                      layer="below", exclude_empty_subplots=False)

    # ---- Rebased prices (right axis) ----
    if spx_r is not None and not spx_r.empty:
        fig.add_trace(go.Scatter(
            x=spx_r.index, y=spx_r.values, name="S&P 500 (rebased=100)",
            line={"color": "#1f4e79", "width": 2},
            hovertemplate="S&P 500 · %{y:.1f} (rebased)<extra></extra>",
        ), secondary_y=True)
    if ndx_r is not None and not ndx_r.empty:
        fig.add_trace(go.Scatter(
            x=ndx_r.index, y=ndx_r.values, name="Nasdaq (rebased=100)",
            line={"color": "#6a1b9a", "width": 2},
            hovertemplate="Nasdaq · %{y:.1f} (rebased)<extra></extra>",
        ), secondary_y=True)

    # ---- Bubble Index (left axis) ----
    if not sc.empty:
        fig.add_trace(go.Scatter(
            x=sc.index, y=sc.values,
            name="Bubble Index", line={"color": "#c1121f", "width": 2.6},
            fill="tozeroy", fillcolor="rgba(193,18,31,0.05)",
            hovertemplate="Bubble Index · %{y:.1f}<extra></extra>",
        ), secondary_y=False)

    # ---- Layout ----
    # Title top-left, legend top-right — they never collide. The x/y ranges
    # autorange to the sliced data so the chart auto-stretches on range change.
    fig.update_layout(
        height=520, hovermode="x unified", showlegend=True,
        margin={"t": 50, "b": 50, "l": 70, "r": 90},
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right",
                    bgcolor="rgba(255,255,255,0.6)", bordercolor="#e2e8f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text="Bubble Index (left) · S&P 500 / Nasdaq rebased to 100 (right)",
                   x=0.01, xanchor="left", font=dict(size=13, color="#0f172a")),
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#e2e8f0",
                     ticks="outside", row=1, col=1,
                     autorange=True,
                     dtick="M1", tickformat="%Y-%m",
                     hoverformat="%Y-%m-%d")
    fig.update_yaxes(showgrid=False, range=[0, 100],
                     title_text="Bubble Index (0-100)",
                     row=1, col=1, secondary_y=False, title_font=dict(size=11))
    fig.update_yaxes(showgrid=False, title_text="Index (100 = window start)",
                     row=1, col=1, secondary_y=True, title_font=dict(size=11),
                     autorange=True)
    return fig


def backtest_panel(scores: pd.Series, params: dict, meta: dict = None):
    """Interactive Bubble-DCA vs Buy&Hold backtest. Delegates to the crash-proof
    engine in backtest.py (run_backtest returns (metrics_df, chart_fig) or None).
    Metrics are HONEST for a DCA (money-weighted return / total invested /
    final value / max drawdown) — never a naive end/start equity ratio.
    Never raises: any engine error degrades to a friendly warning so the rest
    of the dashboard keeps rendering.
    """
    st.subheader("📊 Strategy Backtest — Bubble-Risk DCA vs Buy & Hold")

    # Synthetic score series would produce a meaningless backtest — skip it
    # explicitly instead of rendering numbers that look authoritative.
    if meta and meta.get("source") == "synthetic":
        st.info("⏸️ Backtest is paused while the score series is in **synthetic** "
                "fallback mode — a DCA backtest against fabricated risk data would "
                "be misleading. It resumes automatically once real FRED data is "
                "available (fix `FRED_API_KEY` on Render and redeploy).")
        return

    # --- User-selectable time range (years) -----------------------------------
    valid_years = sorted({int(d.year) for d in scores.dropna().index})
    if valid_years:
        c1, c2 = st.columns([1, 1])
        start_year = c1.select_slider(
            "Backtest start", options=valid_years,
            value=st.session_state.get("bt_start", min(valid_years)),
            key="bt_start_widget",
            help="Backtest begins in January of this year.")
        end_year = c2.select_slider(
            "Backtest end", options=valid_years,
            value=st.session_state.get("bt_end", max(valid_years)),
            key="bt_end_widget",
            help="Backtest ends in December of this year.")
        if start_year > end_year:
            st.warning("Start year must be ≤ end year — auto-swapped.")
            start_year, end_year = end_year, start_year
        st.session_state["bt_start"] = start_year
        st.session_state["bt_end"] = end_year
        mask = scores.dropna().index
        mask = mask[(mask.year >= start_year) & (mask.year <= end_year)]
        scores_bt = scores.loc[mask]
    else:
        scores_bt = scores

    if params.get("recycle"):
        st.caption("Benchmark: fixed ${:,}/mo into SPY (buy & hold). Strategy: the "
                   "SAME ${:,}/mo outflow — the index only decides deployment: "
                   "taper & stockpile cash at high risk (≥80), deploy the reserve "
                   "at low readings (<40/50), de-risk at extremes (≥95). Same "
                   "total invested, so IRR isolates pure timing skill. The lower "
                   "subplot shows the cash reserve accumulating/depleting."
                   .format(int(params["base_monthly"]), int(params["base_monthly"])))
    else:
        st.caption("Benchmark: fixed ${:,}/mo into SPY (buy & hold). Strategy "
                   "(default): scale the monthly contribution by the index band "
                   "— 3× at score <40, 1.5× at 40–50, 1× at 50–80, 0.5× at "
                   "80–95, 0× + 15% cash sleeve ≥95. Deploys more at lows and "
                   "less at highs → visibly diverges from DCA, beats it in $ "
                   "Final Value, IRR stays comparable. The lower subplot shows "
                   "the strategy's cash sleeve (mostly empty in multiplier "
                   "mode — dollars go straight to SPY)."
                   .format(int(params["base_monthly"])))

    # Recompute ONLY when the apply button changed the params or the time range
    # (or first load): unrelated widget reruns do NOT re-run the backtest.
    key = json.dumps({"p": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in params.items()},
                      "rng": [start_year, end_year]}, sort_keys=True)
    if st.session_state.get("bt_result_key") != key:
        with st.spinner("Running backtest ({}–{})...".format(start_year, end_year)):
            try:
                spy = load_spy()
                # Slice SPY to the same range so the inner-join doesn't lose data
                spy_bt = spy.loc[(spy.index.year >= start_year)
                                  & (spy.index.year <= end_year)] \
                    if spy is not None else None
                res = bt.run_backtest(scores_bt, spy_bt, params)
                if res is None:
                    st.session_state["bt_error"] = (
                        "Backtest engine returned no result — need SPY price "
                        "history + a scored series. If you just deployed, the "
                        "baked cache may still be building; the page shows "
                        "cached data until the next refresh.")
                else:
                    st.session_state["bt_result"] = res
                    st.session_state.pop("bt_error", None)
            except Exception as exc:
                st.session_state["bt_error"] = repr(exc)
        st.session_state["bt_result_key"] = key

    if st.session_state.get("bt_error"):
        st.error(f"回测引擎报错（页面其余部分不受影响）：`{st.session_state['bt_error']}`\n\n"
                 "请在侧边栏点击「✅ 应用参数并运行回测」重试；若持续失败，打开 "
                 "🔧 数据诊断 确认数据源状态。")
        return

    res = st.session_state.get("bt_result")
    if res is None:
        st.warning("Backtest needs SPY price history + a scored series (cache). "
                   "Run the scoring first / ensure network connectivity.")
        return

    metrics_df, chart_fig = res
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.plotly_chart(chart_fig, use_container_width=True)

    # Per-dollar efficiency (Final Value / Total Invested) — the real measure
    # of "alpha per $ deployed": > 1.0 = the strategy multiplies dollars better
    # than DCA did; < 1.0 = DCA did. Together with the table above, this lets
    # the user see whether the $ advantage came from "more shares at lower
    # prices" (good) or just "more dollars" (neutral).
    try:
        ti_b = float(metrics_df.loc[metrics_df["Metric"] == "Total Invested",
                                     "Benchmark (Buy & Hold DCA)"].iloc[0]
                      .replace("$", "").replace(",", ""))
        fv_b = float(metrics_df.loc[metrics_df["Metric"] == "Final Value",
                                     "Benchmark (Buy & Hold DCA)"].iloc[0]
                      .replace("$", "").replace(",", ""))
        ti_s = float(metrics_df.loc[metrics_df["Metric"] == "Total Invested",
                                     "Bubble-DCA Strategy"].iloc[0]
                      .replace("$", "").replace(",", ""))
        fv_s = float(metrics_df.loc[metrics_df["Metric"] == "Final Value",
                                     "Bubble-DCA Strategy"].iloc[0]
                      .replace("$", "").replace(",", ""))
        eff_b = fv_b / ti_b
        eff_s = fv_s / ti_s
        delta = (eff_s / eff_b - 1) * 100
        c1, c2 = st.columns(2)
        c1.metric("DCA: $1 变成",
                  f"${eff_b:.2f}",
                  help="基准定投的每美元回报 (Final Value / Total Invested)")
        c2.metric("策略: $1 变成",
                  f"${eff_s:.2f}",
                  delta=f"{delta:+.1f}% vs 定投",
                  help="策略的每美元回报。**正值 = 策略在每一美元上比定投更高效** "
                       "（同样投入，回报更高）。",
                  delta_color=("normal" if delta >= 0 else "inverse"))
    except Exception:
        pass


def historical_comparison(scores: pd.Series, state: dict):
    """Overlay the live index history with the canonical bubble episodes and a
    'Compared with historical bubbles' summary table."""
    st.subheader("🕰️ Compared with Historical Bubbles & Buying Windows")
    bm = pipe.historical_benchmarks(scores)
    opp = pipe.opportunity_benchmarks(scores)
    today = state.get("score", np.nan)
    hist_pct = state.get("hist_pct", np.nan)

    # summary line
    if pd.notna(today) and pd.notna(hist_pct):
        peak2000 = bm.get("dotcom_2000", {}).get("score", np.nan)
        vs = (f"Current {today:.0f} is higher than {hist_pct:.0f}% of history"
              + (f" and below the 2000 dot-com peak ({peak2000:.0f})."
                 if pd.notna(peak2000) else "."))
        st.markdown(f"<div class='status-note' style='margin-bottom:8px'>{vs}</div>",
                    unsafe_allow_html=True)

    # table: episode | date | score  (risk climaxes AND accumulation troughs)
    risk_rows = "".join(
        f'<tr><td style="padding:4px 10px">🔴 {label}</td>'
        f'<td style="padding:4px 10px;text-align:right">{d["date"]}</td>'
        f'<td style="padding:4px 10px;text-align:right;font-weight:700;color:#c1121f">{d["score"]:.0f}</td></tr>'
        for label, d in [("Dot-com 2000", bm.get("dotcom_2000", {})),
                         ("GFC 2007", bm.get("gfc_2007", {})),
                         ("COVID pre-2020", bm.get("covid_pre", {})),
                         ("2021 liquidity", bm.get("bubble_2021", {}))]
        if d)
    opp_rows = "".join(
        f'<tr><td style="padding:4px 10px">🟢 {label}</td>'
        f'<td style="padding:4px 10px;text-align:right">{d["date"]}</td>'
        f'<td style="padding:4px 10px;text-align:right;font-weight:700;color:#1a9850">{d["score"]:.0f}</td></tr>'
        for label, d in [("Dot-com trough 2002", opp.get("dotcom_trough_2002", {})),
                         ("GFC trough 2009", opp.get("gfc_trough_2009", {})),
                         ("COVID trough 2020", opp.get("covid_trough_2020", {})),
                         ("2022 bear trough (big buy)", opp.get("bear_trough_2022", {}))]
        if d)
    cur_row = (
        f'<tr><td style="padding:4px 10px">📍 Current</td>'
        f'<td style="padding:4px 10px;text-align:right">'
        f'{pd.Timestamp(state.get("as_of")).strftime("%Y-%m") if state.get("as_of") else "—"}</td>'
        f'<td style="padding:4px 10px;text-align:right;font-weight:700">{today:.0f}</td></tr>'
        if pd.notna(today) else "")
    st.markdown(f"""
    <div class="status-card" style="margin-bottom:10px">
      <div class="status-head">Index at canonical episodes — risk climaxes 🔴 vs accumulation windows 🟢</div>
      <table style="width:100%;font-size:13px;color:#0f172a">
        <tr><th style="text-align:left;padding:2px 10px">Episode</th>
        <th style="text-align:right;padding:2px 10px">Date</th>
        <th style="text-align:right;padding:2px 10px">Score</th></tr>
        {risk_rows}{opp_rows}{cur_row}
      </table>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.title("📈 US Equity Bubble Risk Index")
    st.caption("5-module macro framework · Valuation 30% · Sentiment 20% "
               "· Leverage 20% · Structure 15% · Macro 15% · every indicator is a "
               "trailing robust-Z score (20y MAD, no look-ahead) → weighted module "
               "blend → fixed-gain calibration (0-100) → canonical daily index "
               "(50% monthly macro anchor + 50% daily price/VIX regime, stress-aware "
               "daily clamp) · confirmation-style buy/sell signals grounded in history · "
               "cache-first, crash-proof loading · free/open data (FRED, yfinance/Stooq)")

    # ---- Sidebar: refresh controls ---------------------------------------
    # NOTE: the pipeline AUTO-refreshes whenever the on-disk cache is older
    # than CACHE_MAX_AGE_HOURS (incremental, bounded, crash-proof) — so data
    # reaches the latest trading day without user action. This checkbox forces
    # a refresh even when the cache is fresh.
    refresh = st.sidebar.checkbox(
        "Force refresh live data", value=False,
        help="默认自动：缓存超过 6 小时会在打开页面时自动增量更新到最新交易日。"
             "勾选此项可绕过缓存强制立即刷新。")
    if st.sidebar.button("Re-run scoring"):
        st.cache_data.clear()
        refresh = True

    log_scale = st.sidebar.checkbox("Log scale (price chart)", value=False,
                                    help="Only affects the standalone price sub-chart "
                                         "in the V2/V3 layout. The combined view "
                                         "below always rebases prices to 100.")

    # ---- Sidebar: valuation acceleration toggle -------------------------
    tail_boost = st.sidebar.checkbox(
        "Valuation acceleration curve", value=pipe.TAIL_BOOST_ON,
        help="Legacy toggle: the V3 robust-Z scoring no longer applies a "
             "separate acceleration curve, so this switch has no effect on "
             "current scores (kept for dashboard compatibility).")

    # ---- Sidebar: interactive backtest parameters (applied on button) -----
    # Widgets live in a FORM so tweaking them does NOT recompute the whole
    # backtest on every keystroke — only the "应用参数并运行回测" submit runs it.
    if "bt_params" not in st.session_state:
        st.session_state["bt_params"] = dict(
            base_monthly=1000, low_mult=3.0, high_mult=0.5,
            derisk_threshold=95, derisk_cash=0.15, cash_yield=4.0, recycle=False)
    with st.sidebar.expander("🎛️ Backtest Parameters", expanded=False):
        with st.form("bt_form"):
            base_monthly = st.number_input("Base Monthly DCA ($)", min_value=0,
                                           max_value=10000, value=1000, step=100)
            recycle = st.checkbox(
                "Reserve-recycle mode (same $ outflow as DCA)", value=False,
                help="OFF (默认): 倍率模式 — 低风险多投、高风险少投，曲线明显与定投分离，"
                     "最终收益在定投基础上叠加（但总投入更多）。"
                     "ON: 资金回收模式 — 每月投入和定投相同，只在择时上优化，"
                     "曲线几乎与定投重合（IRR 衡量纯择时 alpha）。")
            low_mult = st.slider("Deep-Value Deploy Cap (Score < 40)", 1.0, 4.0, 3.0, 0.1,
                                 help="At deep-value readings deploy the base "
                                      "amount PLUS up to this multiple drawn "
                                      "from the reserve.")
            high_mult = st.slider("Taper Fraction (80 ≤ Score < 95)",
                                  0.0, 1.0, 0.5, 0.05,
                                  help="Fraction of the monthly amount deployed "
                                       "at high risk; the rest is stockpiled.")
            derisk_threshold = st.slider("De-Risk Threshold Score", 85, 99, 95, 1)
            derisk_cash = st.slider("De-Risk Cash Allocation", 0.0, 0.5, 0.15, 0.05)
            cash_yield = st.number_input("Cash Yield (Annualized %)", min_value=0.0,
                                         max_value=10.0, value=4.0, step=0.5)
            submitted = st.form_submit_button(
                "✅ 应用参数并运行回测", use_container_width=True)
            if submitted:
                st.session_state["bt_params"] = dict(
                    base_monthly=base_monthly, low_mult=low_mult,
                    high_mult=high_mult, derisk_threshold=derisk_threshold,
                    derisk_cash=derisk_cash, cash_yield=cash_yield,
                    recycle=recycle)
    params = st.session_state["bt_params"]

    # ---- Single data pass (cache-first; the pipeline never raises, but keep
    #      a last-resort guard so the page can never hard-crash) -------------
    with st.spinner("加载气泡指数中（本地缓存优先，必要时增量更新）..."):
        try:
            scores, meta = load_scores(refresh=refresh, tail_boost=tail_boost)
            state = pipe.get_latest_state(refresh=False, tail_boost=tail_boost)
        except Exception as exc:
            st.error("数据加载遇到问题（已触发保护，页面未中断）。请稍后重试或点击 "
                     f"“Force refresh live data”。详情：{exc!r}")
            st.stop()

    src = meta.get("source", "unknown")
    src_label = {"live": "Real-time", "cache": "Cached", "synthetic": "Synthetic"}.get(src, src)
    if src == "synthetic":
        # Switch into the RESILIENT mode: drop the synthetic path and serve a
        # real-time yfinance-only index (price / VIX / credit / yield). This
        # is real data (not the deterministic synthetic), it just uses a
        # different feature set than the validated V3 monthly composite.
        try:
            ri = pipe.compute_resilient_index()
            if not ri.empty:
                scores = ri
                meta["source"] = "resilient"
                meta["resilient_only"] = True
                st.info(
                    "ℹ️ FRED egress blocked on this Render instance — switched "
                    "to **resilient mode**: real-time Bubble Index computed "
                    "from yfinance (price / VIX / credit / 10y yield). "
                    "Validation is coarser than the FRED-based composite; "
                    "the gauge / signals / chart now reflect actual market "
                    "conditions rather than a synthetic series. Once FRED is "
                    "reachable again the validated V3 composite returns.")
        except Exception:
            pass
        if meta.get("source") == "synthetic":
            # Fall back to the old hint if even the resilient index failed.
            st.warning("⚠️ Live data unavailable — showing a **deterministic synthetic** "
                       "series for layout/demo only. On Render, set `FRED_API_KEY` as an "
                       "**environment variable for BOTH build and runtime** (Dashboard → "
                       "Environment → Add, no quotes), then trigger **Manual Deploy → "
                       "Clear build cache & deploy** so the baked parquet cache is rebuilt "
                       "with real FRED data.")
    elif src == "cache":
        st.info("ℹ️ Showing cached data (set refresh to pull live).")

    # ---- Canonical DAILY index (single source of truth for display) --------
    # The gauge, status card and guidance all read this DAILY series — the same
    # red line the chart shows — so the headline number can never disagree with
    # the chart. In resilient mode `scores` already IS the daily series. The
    # monthly macro composite stays the basis for the backtest / signals /
    # module drivers (its natural monthly cadence).
    daily = scores if meta.get("source") == "resilient" else None
    if daily is None:
        try:
            daily = load_daily_scores(refresh=False, tail_boost=tail_boost)
        except Exception:
            daily = pd.Series(dtype=float)
    if daily is not None and not daily.dropna().empty:
        dvals = daily.dropna()
        dscore = float(dvals.iloc[-1])
        state["score"] = dscore
        state["status"] = pipe.risk_level(dscore)
        state["as_of"] = dvals.index[-1]
        try:
            state["hist_pct"] = float((dvals <= dscore).mean() * 100.0)
        except Exception:
            pass

    # ---- Staleness guard: warn if the score's latest reading is old --------
    # The 6h auto-refresh keeps data current when the network works. If the
    # served score is still more than a week stale, surface it clearly so the
    # user knows the guidance is not based on the latest tape.
    try:
        cinfo = pipe.cache_info()
        sc_asof = cinfo.get("score_as_of")
        age = cinfo.get("age_hours")
        if sc_asof is not None:
            stale_days = (pd.Timestamp.today() - pd.Timestamp(sc_asof)).days
            if stale_days > 7:
                st.warning(
                    f"⚠️ 指数读数已 {stale_days} 天未更新（最新 {pd.Timestamp(sc_asof).strftime('%Y-%m-%d')}）。"
                    "自动刷新可能因网络受限失败——请点击侧边栏「Re-run scoring」强制刷新，"
                    "或检查「🔧 数据诊断」里的连通性探测。")
        elif age is not None and age > 48:
            st.warning("⚠️ 数据缓存超过 48 小时未写入——自动刷新可能失败，请点击侧边栏"
                       "「Re-run scoring」或检查连通性。")
    except Exception:
        pass

    # ---- Data diagnostics (auto-open when in synthetic fallback) ----------
    # Turns the "刷新不出数据" black box into concrete answers: which upstream
    # endpoints are reachable from THIS container, whether FRED_API_KEY is
    # present, and which raw series / features came back. No log access needed.
    with st.expander("🔧 数据诊断 (Data diagnostics)",
                     expanded=(src == "synthetic")):
        feats = meta.get("features", {}) or {}
        diag = {
            "source": src,
            "available_features": meta.get("available_count"),
            "FRED_API_KEY set (runtime)": bool(os.getenv("FRED_API_KEY")),
            "fetch deadline (s)": pipe.FETCH_DEADLINE,
            "per-request timeout (s)": pipe.FETCH_TIMEOUT,
        }
        st.json(diag)
        if feats:
            st.caption("Raw series / feature availability (Y = data present):")
            st.json(feats)
        # Episode calibration readout — verifies the fixed-gain calibration
        # against the named episodes with whatever data this container has.
        try:
            _s, _ = load_scores(refresh=False, tail_boost=tail_boost)
            bm = pipe.historical_benchmarks(_s)
            _last = _s.dropna()
            _today = ({"date": _last.index[-1].strftime("%Y-%m"),
                       "score": float(_last.iloc[-1])} if not _last.empty else {})
            st.caption("Episode calibration (peak score inside each window):")
            st.json({
                label: {"date": d["date"], "score": round(d["score"], 1)}
                for label, d in [("dotcom_2000", bm.get("dotcom_2000", {})),
                                 ("gfc_2007", bm.get("gfc_2007", {})),
                                 ("covid_pre_2020", bm.get("covid_pre", {})),
                                 ("bubble_2021", bm.get("bubble_2021", {})),
                                 ("latest", _today)]
                if d})
        except Exception:
            pass
        # Buy/sell signals + historical grounding (grounded in real history)
        try:
            _s, _ = load_scores(refresh=False, tail_boost=tail_boost)
            sig = pipe.detect_signals(_s.dropna())
            sts = pipe.signal_stats(_s.dropna())
            st.caption("Signals (sell: score >78 for ≥3m · buy: <45 for ≥2m or "
                       "≥15pt fall in ≤3m) + forward-return stats:")
            st.json({
                "sell_signals": [f"{pd.Timestamp(d).strftime('%Y-%m')} ({v:.0f})"
                                 for d, v in sig["sell"][-6:]],
                "buy_signals": [f"{pd.Timestamp(d).strftime('%Y-%m')} ({v:.0f}, {k})"
                                for d, v, k in sig["buy"][-6:]],
                "fwd12m_after_sell": sts.get("sell", {}).get("fwd12"),
                "fwd12m_after_buy": [sts.get("buy_sustained", {}).get("fwd12"),
                                     sts.get("buy_rapid", {}).get("fwd12")],
                "fwd12m_benchmark": sts.get("_benchmark", {}).get("fwd12"),
            })
        except Exception:
            pass
        if st.button("🌐 Run connectivity probe (FRED / Stooq / Yahoo)"):
            with st.spinner("Probing upstream endpoints from this container..."):
                probe = pipe.probe_connectivity()
            st.write(f"FRED_API_KEY set: **{probe['fred_api_key_set']}**")
            st.dataframe(pd.DataFrame(probe["targets"]),
                         use_container_width=True, hide_index=True)

    # ---- Top: gauge (left) + status card (right) -------------------------
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.markdown('<div class="gauge-wrap">', unsafe_allow_html=True)
        st.plotly_chart(gauge_fig(state["score"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # Last-updated stamp right under the gauge (仪表盘下方)
        try:
            cinfo = pipe.cache_info()
            wa = cinfo.get("written_at")
            if wa is not None:
                t = pd.Timestamp(wa)
                age = cinfo.get("age_hours")
                age_txt = (f"（{age:.1f} 小时前自动更新）" if age is not None else "")
                st.markdown(
                    f"<div style='font-size:12px;color:#64748b;text-align:center;"
                    f"margin-top:-6px'>🕐 最近更新: {t.strftime('%Y-%m-%d %H:%M')} "
                    f"{age_txt}<br>"
                    f"<span style='font-size:11px;color:#94a3b8'>数据自动增量刷新 "
                    f"(缓存超 6 小时即更新)</span></div>",
                    unsafe_allow_html=True)
        except Exception:
            pass
    with c2:
        status_card(state, meta, src_label, tail_boost)

    # ---- Actionable guidance (zone -> posture, anchored to past events) ---
    # Uses the SAME canonical daily index computed above (gauge/status/chart
    # all agree), with the monthly macro composite passed for the validated
    # confirmation signals.
    guidance_panel(state, daily, scores)

    # ---- Module radar + module cards -------------------------------------
    st.subheader("Five Risk Modules (current)")
    rc1, rc2 = st.columns([1, 1.25])
    with rc1:
        st.plotly_chart(radar_fig(state.get("modules", {})), use_container_width=True)
    with rc2:
        module_cards(state.get("modules", {}))
        drivers_panel(state)

    # ---- Historical comparison ------------------------------------------
    # Compare the CURRENT canonical (daily) index against the same daily series
    # at canonical episodes, so the "vs history" reading is fully self-consistent.
    historical_comparison(daily, state)

    # ---- Combined chart: Bubble Index + rebased S&P 500 / Nasdaq ------------
    st.subheader("Bubble Index vs Market (S&P 500 / Nasdaq rebased to 100)")

    # --- User-selectable DATE range (controls the combined chart) ----------
    # Two date pickers (start / end) bounded by the available data range, plus
    # a row of preset buttons for one-click common windows. The data is
    # sliced to the exact [start, end] dates, not by year.
    # `daily` is the canonical index (resilient series in resilient mode, the
    # blended daily index otherwise) — the same series the gauge reads.
    daily_for_chart = daily
    if not daily_for_chart.empty:
        d_min = daily_for_chart.index.min().date()
        d_max = daily_for_chart.index.max().date()

        # Preset range buttons (one-click: rewrite the session_state dates
        # and rerun). 'key' must be unique; we mutate session_state and ask
        # Streamlit to re-run.
        def _apply_preset(days_back):
            from datetime import timedelta
            end = d_max
            start = max(d_min, end - timedelta(days=days_back))
            st.session_state["chart_start_date"] = start
            st.session_state["chart_end_date"] = end

        presets = st.columns([1, 1, 1, 1, 1, 1.5])
        if presets[0].button("1M", use_container_width=True):
            _apply_preset(31); st.rerun()
        if presets[1].button("3M", use_container_width=True):
            _apply_preset(93); st.rerun()
        if presets[2].button("6M", use_container_width=True):
            _apply_preset(186); st.rerun()
        if presets[3].button("1Y", use_container_width=True):
            _apply_preset(365); st.rerun()
        if presets[4].button("2Y", use_container_width=True):
            _apply_preset(730); st.rerun()
        if presets[5].button("📆 全部 (All)", use_container_width=True):
            st.session_state["chart_start_date"] = d_min
            st.session_state["chart_end_date"] = d_max
            st.rerun()

        c1, c2, c3 = st.columns([1, 1, 1.2])
        default_start = st.session_state.get("chart_start_date", max(d_min, d_max - pd.Timedelta(days=365*2)))
        default_end   = st.session_state.get("chart_end_date", d_max)
        chart_start = c1.date_input(
            "📅 开始日期", value=default_start,
            min_value=d_min, max_value=d_max,
            key="chart_start_date",
            help="图表起始日期（精确到日）。")
        chart_end = c2.date_input(
            "📅 结束日期", value=default_end,
            min_value=d_min, max_value=d_max,
            key="chart_end_date",
            help="图表结束日期（精确到日）。")
        if chart_start > chart_end:
            st.warning("开始日期需 ≤ 结束日期——已自动交换。")
            chart_start, chart_end = chart_end, chart_start

        # Slice the DAILY index to the exact date range
        daily_for_chart = daily_for_chart.loc[
            (daily_for_chart.index.date >= chart_start)
            & (daily_for_chart.index.date <= chart_end)]
        n_days = len(daily_for_chart)
        c3.markdown(
            f"<div style='font-size:12px;color:#475569;padding-top:1.6rem'>"
            f"📅 显示区间: <b>{chart_start} → {chart_end}</b> "
            f"({n_days} 个交易日)</div>",
            unsafe_allow_html=True)

    try:
        spx, ndx = load_prices()
    except Exception:
        spx, ndx = None, None
    # Slice price series to the same exact dates so the rebase start aligns
    if not daily_for_chart.empty and spx is not None and not spx.empty:
        spx = spx.loc[(spx.index.date >= chart_start)
                     & (spx.index.date <= chart_end)]
    if not daily_for_chart.empty and ndx is not None and not ndx.empty:
        ndx = ndx.loc[(ndx.index.date >= chart_start)
                     & (ndx.index.date <= chart_end)]
    st.plotly_chart(history_fig(daily_for_chart if not daily_for_chart.empty else scores,
                                spx, ndx, log_scale=False, view="all"),
                    use_container_width=True)

    # ---- Data freshness footer (proves the charts reach the latest day) ----
    try:
        cinfo = pipe.cache_info()
        parts = []
        dl = daily_for_chart.dropna()
        if not dl.empty:
            parts.append(f"Bubble Index 截至 <b>{dl.index[-1].strftime('%Y-%m-%d')}</b>（日度信号）")
        px = spx.dropna() if spx is not None else None
        if px is not None and not px.empty:
            parts.append(f"S&P 500 截至 <b>{px.index[-1].strftime('%Y-%m-%d')}</b>")
        nx = ndx.dropna() if ndx is not None else None
        if nx is not None and not nx.empty:
            parts.append(f"Nasdaq 截至 <b>{nx.index[-1].strftime('%Y-%m-%d')}</b>")
        wa = cinfo.get("written_at")
        if wa is not None:
            age = cinfo.get("age_hours")
            age_txt = (f"{age:.1f} 小时前更新" if age is not None else "")
            parts.append(f"缓存 {pd.Timestamp(wa).strftime('%m-%d %H:%M')} ({age_txt})")
        st.caption(f"🔍 数据时间线: {' · '.join(parts)}", unsafe_allow_html=True)
    except Exception:
        pass

    # ---- Strategy backtest (interactive) --------------------------------
    backtest_panel(scores, params, meta)


if __name__ == "__main__":
    main()
