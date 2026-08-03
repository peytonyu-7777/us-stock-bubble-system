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
/* fluid main container */
.main .block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:2rem}

/* stack horizontal column blocks on small screens */
@media (max-width:720px){
  .stHorizontalBlock{flex-direction:column !important}
  .stHorizontalBlock>div{width:100% !important;min-width:0 !important}
}

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
  box-shadow:0 6px 20px rgba(15,23,42,0.08);height:100%}
.status-head{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
  color:#64748b;font-weight:700}
.badge{display:inline-block;margin:10px 0;padding:11px 16px;border-radius:11px;
  color:#fff;font-weight:800;font-size:18px;letter-spacing:.01em;
  box-shadow:0 4px 12px rgba(15,23,42,0.18)}
.status-score{font-size:15px;color:#0f172a;margin-top:2px}
.status-note{font-size:13px;color:#475569;margin-top:6px}
.status-cov{font-size:12px;color:#94a3b8;margin-top:10px}
.status-elev{font-size:12px;color:#c1121f;font-weight:700;margin-top:8px}

/* gauge wrapper */
.gauge-wrap{border:1px solid #e2e8f0;border-radius:14px;padding:6px 6px 0;
  box-shadow:0 6px 20px rgba(15,23,42,0.08);background:#fff}
</style>
""", unsafe_allow_html=True)

from pipeline import RISK_BANDS as BANDS  # V2 risk-level bands (0-40 .. 90-100)
from pipeline import MODULE_WEIGHTS       # 5-module weights for the radar/labels


def band_color(score: float) -> str:
    if pd.isna(score):
        return "gray"
    for lo, hi, c, _ in BANDS:
        if lo <= score < hi:
            return c
    return "#991b1b"


def status_action(score: float):
    """Return (badge_text, badge_color, note) for the Status Card (V2 bands)."""
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
    spx = pipe.get_price_series("^GSPC", start=pipe.LIVE_START)
    ndx = pipe.get_price_series("^IXIC", start=pipe.LIVE_START)
    return spx, ndx


@st.cache_data(ttl=3600)
def load_spy():
    return pipe.get_price_series("SPY", start=pipe.LIVE_START)


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

    # --- 5-step color bands aligned with the V2 risk-level bands -----------
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
    hist_txt = (f"Higher than {hist_pct:.0f}% of history"
                if pd.notna(hist_pct) else "n/a")
    asof = (pd.Timestamp(state["as_of"]).date() if state.get("as_of") else "n/a")
    st.markdown(f"""
    <div class="status-card">
      <div class="status-head">Current Bubble Risk Index</div>
      <div class="badge" style="background:{color}">{action}</div>
      <div class="status-score">Score <b>{score:.1f}</b> / 100 &nbsp;·&nbsp; as of {asof}</div>
      <div class="status-note">{note}</div>
      <div class="status-cov">Risk accumulation indicator — not a crash forecast</div>
      <div class="status-cov">Data: {src_label} &nbsp;·&nbsp; coverage {avail}/8 &nbsp;·&nbsp; {hist_txt}</div>
      <div class="status-cov">Top modules: {mod_html or 'n/a'}</div>
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


def guidance_panel(state: dict, daily: pd.Series):
    """Actionable guidance: current zone -> posture, anchored to the detected
    historical risk climaxes and accumulation windows (the 'what do I do now'
    panel, incl. the late-2022-style deep-fear buying opportunity)."""
    score = state.get("score", np.nan)
    if pd.isna(score):
        return
    if score < 35:
        posture, color, note = (
            "ACCUMULATE AGGRESSIVELY · 2.0x DCA", "#1a9850",
            "Deep-fear zone — historically the STRONGEST forward 12–24m return "
            "window (2002-09, 2009-03, 2020-03, 2022-10).")
    elif score < 40:
        posture, color, note = (
            "ACCUMULATE · 2.0x DCA", "#1a9850",
            "Cheap / fear zone — risk is below median; keep buying the dip.")
    elif score < 60:
        posture, color, note = (
            "STEADY ACCUMULATION · 1.5x DCA", "#2166ac",
            "Balanced zone — no speculative excess; stay on plan.")
    elif score < 75:
        posture, color, note = (
            "BASE PACE · 1.0x DCA", "#f4a01c",
            "Valuation elevated — no new aggression, no panic either.")
    elif score < 90:
        posture, color, note = (
            "TRIM / RAISE CASH · 0.5x DCA", "#e4572e",
            "Bubble-risk zone — risk is accumulating; scale down exposure.")
    else:
        posture, color, note = (
            "DE-RISK · 0x DCA + cash sleeve", "#c1121f",
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
      <div class="status-note" style="margin-top:6px">When the index falls into
      the deep-fear zone (≤35) it has historically marked the best accumulation
      windows — e.g. the <b>late-2022 bear bottom</b>. When it pushes above 75 it
      has preceded every major drawdown. This is risk-accumulation guidance,
      not a crash prediction.</div>
    </div>
    """, unsafe_allow_html=True)


def history_fig(scores: pd.Series, spx, ndx, log_scale: bool = True,
                view: str = "all") -> go.Figure:
    if view == "3y":
        START = pd.Timestamp("2023-07-31")
    else:
        START = pd.Timestamp("2000-01-01")
    ytype = "log" if log_scale else "linear"

    # --- strict alignment: crop to >=START, then intersect the two prices so
    #     the dual axis shares an identical x-grid (no 2001-vs-1995 drift), and
    #     force the score onto that same intersection so zoom/hover link. ---
    if spx is not None:
        spx = spx[spx.index >= START]
    if ndx is not None:
        ndx = ndx[ndx.index >= START]
    if spx is not None and ndx is not None and not spx.empty and not ndx.empty:
        common = spx.index.intersection(ndx.index)
        spx = spx.reindex(common)
        ndx = ndx.reindex(common)

    price_lo = None
    if spx is not None and not spx.empty:
        price_lo = spx.index.min()
    elif ndx is not None and not ndx.empty:
        price_lo = ndx.index.min()

    sc = scores.dropna()
    sc = sc[sc.index >= START]
    if price_lo is not None:
        sc = sc[sc.index >= price_lo]   # keep the hover line aligned with prices

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.65, 0.35],
        subplot_titles=("S&P 500 (left axis)  ·  Nasdaq (right axis)",
                        "Bubble Risk Index (stability-filtered)"),
        specs=[[{"secondary_y": True}], [{}]],
    )

    if spx is not None and not spx.empty:
        fig.add_trace(go.Scatter(x=spx.index, y=spx.values, name="S&P 500",
                                 line={"color": "#1f4e79", "width": 1.5}),
                      row=1, col=1, secondary_y=False)
    if ndx is not None and not ndx.empty:
        fig.add_trace(go.Scatter(x=ndx.index, y=ndx.values, name="Nasdaq",
                                 line={"color": "#6a1b9a", "width": 1.5}),
                      row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="S&P 500", type=ytype, row=1, col=1,
                     secondary_y=False)
    fig.update_yaxes(title_text="Nasdaq", type=ytype, row=1, col=1,
                     secondary_y=True)

    # Row 2: smoothed score with V2 risk-zone background shading
    if not sc.empty:
        fig.add_trace(go.Scatter(x=sc.index, y=sc.values, name="Bubble Index",
                                 line={"color": "#c1121f", "width": 2},
                                 fill="tozeroy", fillcolor="rgba(193,18,31,0.06)"),
                      row=2, col=1)
        fig.add_hrect(y0=0, y1=40, fillcolor="rgba(16,185,129,0.10)", line_width=0,
                      row=2, col=1)
        fig.add_hrect(y0=40, y1=60, fillcolor="rgba(59,130,246,0.08)", line_width=0,
                      row=2, col=1)
        fig.add_hrect(y0=60, y1=75, fillcolor="rgba(245,158,11,0.10)", line_width=0,
                      row=2, col=1)
        fig.add_hrect(y0=75, y1=90, fillcolor="rgba(239,68,68,0.10)", line_width=0,
                      row=2, col=1)
        fig.add_hrect(y0=90, y1=100, fillcolor="rgba(153,27,27,0.14)", line_width=0,
                      row=2, col=1)
        fig.add_hline(y=75, line_dash="dash", line_color="#ef4444",
                      annotation_text="Bubble Risk > 75",
                      annotation_position="top left", row=2, col=1)

        # --- Event markers: risk climaxes (red ▼) + accumulation troughs (green ▲)
        #     Data-driven from the plotted series — highlights each risk episode
        #     and the deep-fear buying windows (e.g. the late-2022 bottom).
        try:
            events = pipe.detect_events(sc)
        except Exception:
            events = {"risk": [], "opportunity": []}
        risk_pts = events.get("risk", [])
        opp_pts = events.get("opportunity", [])
        if risk_pts:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in risk_pts], y=[v for _, v in risk_pts],
                mode="markers", name="Risk event (trim)",
                marker={"color": "#c1121f", "size": 9, "symbol": "triangle-down",
                        "line": {"color": "white", "width": 1}},
                hovertemplate="Risk climax %{x|%Y-%m} · score %{y:.0f}<extra></extra>"),
                row=2, col=1)
        if opp_pts:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in opp_pts], y=[v for _, v in opp_pts],
                mode="markers", name="Buying opportunity",
                marker={"color": "#1a9850", "size": 9, "symbol": "triangle-up",
                        "line": {"color": "white", "width": 1}},
                hovertemplate="Accumulation zone %{x|%Y-%m} · score %{y:.0f}<extra></extra>"),
                row=2, col=1)

    fig.update_layout(height=660, hovermode="x unified", showlegend=True,
                      margin={"t": 50, "b": 30, "l": 62, "r": 62},
                      legend=dict(orientation="h", y=1.05, x=0),
                      plot_bgcolor="white", paper_bgcolor="white")
    if view == "3y":
        # Clean presentation window: lock the score axis to 0-100 so the near-3y
        # wave reads as a crisp macro oscillator (matches the reference chart).
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


def backtest_panel(scores: pd.Series, params: dict):
    """Interactive Bubble-DCA vs Buy&Hold backtest. Delegates to the crash-proof
    engine in backtest.py (run_backtest returns (metrics_df, chart_fig) or None).
    Metrics are HONEST for a DCA (money-weighted return / total invested /
    final value / max drawdown) — never a naive end/start equity ratio.
    """
    st.subheader("📊 Strategy Backtest — Bubble-Risk DCA vs Buy & Hold (2000-Present)")
    st.caption("Benchmark: fixed ${:,}/mo into SPY (buy & hold). Strategy: scales "
               "the same contrib. by the Bubble Index band and de-risks to cash when "
               "the index is extreme. Tune the sliders — table & curve recompute live."
               .format(int(params["base_monthly"])))

    spy = load_spy()
    res = bt.run_backtest(scores, spy, params)
    if res is None:
        st.warning("Backtest needs SPY price history + a scored series (cache). "
                   "Run the scoring first / ensure network connectivity.")
        return

    metrics_df, chart_fig = res
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.plotly_chart(chart_fig, use_container_width=True)


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

    # overlay chart: live history + horizontal reference lines at episode peaks
    fig = go.Figure()
    s = scores.dropna()
    if not s.empty:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Bubble Index",
                                 line={"color": "#c1121f", "width": 2},
                                 fill="tozeroy", fillcolor="rgba(193,18,31,0.06)"))
        for label, d in bm.items():
            fig.add_hline(y=d["score"], line_dash="dot", line_color="#94a3b8",
                          annotation_text=f"{label} ≈ {d['score']:.0f}",
                          annotation_position="right")
        for label, d in opp.items():
            fig.add_hline(y=d["score"], line_dash="dot", line_color="#1a9850",
                          annotation_text=f"{label} ≈ {d['score']:.0f}",
                          annotation_position="right")
        # V2 risk bands as background
        fig.add_hrect(y0=75, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0)
        fig.add_hrect(y0=60, y1=75, fillcolor="rgba(245,158,11,0.07)", line_width=0)
        fig.add_hrect(y0=0, y1=40, fillcolor="rgba(26,152,80,0.08)", line_width=0)
    fig.update_layout(height=380, hovermode="x unified",
                      yaxis_title="Bubble Index", yaxis_range=[0, 100],
                      margin={"t": 20, "b": 30, "l": 55, "r": 55},
                      legend=dict(orientation="h", y=1.08, x=0),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📈 US Equity Bubble Risk Index — V2")
    st.caption("Professional 5-module macro framework · Valuation 30% · Sentiment 20% "
               "· Leverage 20% · Structure 15% · Macro 15% · every indicator is a "
               "trailing-historical PERCENTILE (0-100) · weighted blend → historical "
               "affine calibration (dot-com peak ≈ 97, GFC trough ≈ 12) → K-line "
               "stability filter (slow-EMA trend + small bounded oscillation + "
               "stress-aware daily clamp) · cache-first, crash-proof loading · "
               "free/open data (FRED, yfinance/Stooq)")

    # ---- Sidebar: refresh controls ---------------------------------------
    refresh = st.sidebar.checkbox("Force refresh live data", value=False)
    if st.sidebar.button("Re-run scoring"):
        st.cache_data.clear()
        refresh = True

    log_scale = st.sidebar.checkbox("Log price scale", value=True)

    # ---- Sidebar: valuation acceleration toggle -------------------------
    tail_boost = st.sidebar.checkbox(
        "Valuation acceleration curve", value=pipe.TAIL_BOOST_ON,
        help="When ON, the Valuation module uses the non-linear froth-acceleration "
             "curve (percentile 80-95 escalates convexly). When OFF, plain percentile.")

    # ---- Sidebar: interactive backtest sliders ---------------------------
    with st.sidebar.expander("🎛️ Backtest Parameters", expanded=False):
        base_monthly = st.number_input("Base Monthly DCA ($)", min_value=0,
                                       max_value=10000, value=1000, step=100)
        low_mult = st.slider("Low-Risk Multiplier (Score < 40)", 1.0, 3.0, 2.0, 0.1)
        high_mult = st.slider("High-Risk Multiplier (60 ≤ Score < thr)",
                              0.0, 1.0, 0.5, 0.05)
        derisk_threshold = st.slider("De-Risk Threshold Score", 75, 95, 90, 1)
        derisk_cash = st.slider("De-Risk Cash Allocation", 0.0, 0.5, 0.20, 0.05)
        cash_yield = st.number_input("Cash Yield (Annualized %)", min_value=0.0,
                                     max_value=10.0, value=4.0, step=0.5)
    params = dict(base_monthly=base_monthly, low_mult=low_mult,
                  high_mult=high_mult, derisk_threshold=derisk_threshold,
                  derisk_cash=derisk_cash, cash_yield=cash_yield)

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
        st.warning("⚠️ Live data unavailable — showing a **deterministic synthetic** "
                   "series for layout/demo only. Set `FRED_API_KEY` and uncheck the "
                   "cache to fetch real data.")
    elif src == "cache":
        st.info("ℹ️ Showing cached data (set refresh to pull live).")

    # ---- Top: gauge (left) + status card (right) -------------------------
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.markdown('<div class="gauge-wrap">', unsafe_allow_html=True)
        st.plotly_chart(gauge_fig(state["score"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        status_card(state, meta, src_label, tail_boost)

    # ---- Actionable guidance (zone -> posture, anchored to past events) ---
    try:
        _daily_for_guidance = load_daily_scores(refresh=False,
                                                tail_boost=tail_boost)
    except Exception:
        _daily_for_guidance = pd.Series(dtype=float)
    guidance_panel(state, _daily_for_guidance)

    # ---- Module radar + module cards -------------------------------------
    st.subheader("Five Risk Modules (current)")
    rc1, rc2 = st.columns([1, 1.25])
    with rc1:
        st.plotly_chart(radar_fig(state.get("modules", {})), use_container_width=True)
    with rc2:
        module_cards(state.get("modules", {}))
        drivers_panel(state)

    # ---- Historical comparison ------------------------------------------
    historical_comparison(scores, state)

    # ---- History (dual-axis, linked zoom) --------------------------------
    st.subheader("Historical Trend")
    view = st.radio(
        "Historical view",
        options=["all", "3y"],
        format_func=lambda v: ("All (2000 - Present)"
                               if v == "all" else "3-Year Trend (近3年历史走势)"),
        horizontal=True,
        help="All = full 2000-present macro cycle; 3-Year = zoomed 2023-07-31 → "
             "today window with a fixed 0-100 score axis.")
    daily = _daily_for_guidance
    try:
        spx, ndx = load_prices()
    except Exception:
        spx, ndx = None, None
    st.plotly_chart(history_fig(daily if not daily.empty else scores, spx, ndx,
                                log_scale=log_scale, view=view),
                    use_container_width=True)

    # ---- Strategy backtest (interactive) --------------------------------
    backtest_panel(scores, params)


if __name__ == "__main__":
    main()
