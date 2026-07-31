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

BANDS = [  # (lo, hi, color, label) — clean, strictly non-overlapping palette
    (0, 40, "#10b981", "Low / Cooling"),
    (40, 60, "#3b82f6", "Normal"),
    (60, 80, "#f59e0b", "Watch"),
    (80, 90, "#ef4444", "Elevated"),
    (90, 100, "#991b1b", "Bubble Warning"),
]


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


def band_color(score: float) -> str:
    if pd.isna(score):
        return "gray"
    for lo, hi, c, _ in BANDS:
        if lo <= score < hi:
            return c
    return "darkred"


def status_action(score: float):
    """Return (badge_text, badge_color, note) for the Status Card."""
    if pd.isna(score):
        return ("UNKNOWN", "#64748b", "No data")
    if score < 40:
        return ("LOW · 2.0× DCA", "#1a9850", "Cooling — lean in")
    if score < 60:
        return ("NORMAL · 1.5× DCA", "#2166ac", "Steady accumulation")
    if score < 80:
        return ("WATCH · 1.0× DCA", "#f4a01c", "Monitor closely")
    if score < 90:
        return ("ELEVATED · 0.5× DCA", "#e4572e", "Reduced exposure")
    return ("BUBBLE WARNING · 0× DCA", "#c1121f", "De-risk to cash")


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

    # --- 5-step colour bands (new, strictly non-overlapping palette) --------
    STEPS = [
        (0, 20, "#059669", "Extreme Fear / Overweight"),
        (20, 40, "#10b981", "Underweight / Add"),
        (40, 60, "#64748b", "Neutral"),
        (60, 80, "#f59e0b", "Overheat / Halve"),
        (80, 100, "#dc2626", "Bubble / De-risk"),
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
    feats = state.get("features", {})
    elevated = [f["label"].split(" (")[0]
                for f in feats.values()
                if f.get("score") is not None and f["score"] >= 80]
    elev_html = (f'<div class="status-elev">⚠ Elevated (&ge;80): '
                 f'{", ".join(elevated)}</div>') if elevated else ""
    boost_state = ("ON · non-linear escalation" if tail_boost
                   else "OFF · raw weighted score")
    asof = (pd.Timestamp(state["as_of"]).date() if state.get("as_of") else "n/a")
    st.markdown(f"""
    <div class="status-card">
      <div class="status-head">Current Strategy Action</div>
      <div class="badge" style="background:{color}">{action}</div>
      <div class="status-score">Score <b>{score:.1f}</b> / 100 &nbsp;·&nbsp; as of {asof}</div>
      <div class="status-note">{note}</div>
      <div class="status-cov">Data source: {src_label} &nbsp;·&nbsp; Live coverage: {avail}/8</div>
      <div class="status-cov">Tail amplification: <b>{boost_state}</b></div>
      {elev_html}
    </div>
    """, unsafe_allow_html=True)


def feature_cards(features: dict):
    cells = ""
    for key, f in features.items():
        sc = f["score"]
        if sc is None:
            border, color, txt, cls = "#d1d5db", "gray", "Pending", " missing"
        else:
            color = band_color(sc)
            border = color
            txt = f"{sc:.0f}"
            cls = " alert" if sc >= 80 else ""
        cells += (
            f'<div class="feat-card{cls}" style="border-color:{border}">'
            f'<div class="feat-lbl">{f["label"]}</div>'
            f'<div class="feat-val" style="color:{color}">{txt}</div>'
            f'<div class="feat-w">w {f["weight"]:.2f} · '
            f'{"live" if sc is not None else "pending"}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="feat-grid">{cells}</div>', unsafe_allow_html=True)


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
                        "Bubble Risk Score (60-day EMA-smoothed)"),
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

    # Row 2: smoothed score with risk-zone background shading
    if not sc.empty:
        fig.add_trace(go.Scatter(x=sc.index, y=sc.values, name="Bubble Score",
                                 line={"color": "#c1121f", "width": 2},
                                 fill="tozeroy", fillcolor="rgba(193,18,31,0.06)"),
                      row=2, col=1)
        fig.add_hrect(y0=0, y1=40, fillcolor="rgba(26,152,80,0.08)", line_width=0,
                      row=2, col=1)
        fig.add_hrect(y0=40, y1=80, fillcolor="rgba(244,160,28,0.07)", line_width=0,
                      row=2, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(193,18,31,0.12)", line_width=0,
                      row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="#c1121f",
                      annotation_text="High-risk > 80",
                      annotation_position="top left", row=2, col=1)

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
    """
    st.subheader("📊 Strategy Historical Backtest (2000 - Present)")
    st.caption("Benchmark: fixed ${:,} / mo into SPY (buy & hold). "
               "Strategy: Bubble Risk-Adjusted DCA with de-risk. "
               "Tune the sliders — the table & curve recompute live."
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


def main():
    st.title("📈 US Equity Bubble Risk — Dalio-style Monitor")
    st.caption("Dual-speed Z-score + Sigmoid macro model · 7 factors (67% slow "
               "macro anchors + 33% fast sentiment/momentum, F8/F2 emphasised) · "
               "soft F1+F8+F2 bubble-confirmation interaction · dual-pass "
               "smoothing (20d SMA + 60d EMA) · free/open data (FRED incl. "
               "EMVMACROBUS, yfinance/Stooq)")

    # ---- Sidebar: refresh controls ---------------------------------------
    refresh = st.sidebar.checkbox("Force refresh live data", value=False)
    if st.sidebar.button("Re-run scoring"):
        st.cache_data.clear()
        refresh = True

    log_scale = st.sidebar.checkbox("Log price scale", value=True)

    # ---- Sidebar: tail-risk amplification toggle -------------------------
    tail_boost = st.sidebar.checkbox(
        "Tail-risk amplification (F1/F4/F8 >85)", value=pipe.TAIL_BOOST_ON,
        help="When ON, extreme valuation/credit/tech readings are non-linearly "
             "weighted up so the composite punches through the 85-90 warning line. "
             "When OFF, the plain weighted-percentile score is shown.")

    # ---- Sidebar: interactive backtest sliders ---------------------------
    with st.sidebar.expander("🎛️ Backtest Parameters", expanded=False):
        base_monthly = st.number_input("Base Monthly DCA ($)", min_value=0,
                                       max_value=10000, value=1000, step=100)
        low_mult = st.slider("Low-Risk Multiplier (Score < 40)", 1.0, 3.0, 2.0, 0.1)
        high_mult = st.slider("High-Risk Multiplier (80 ≤ Score < thr)",
                              0.0, 1.0, 0.5, 0.05)
        derisk_threshold = st.slider("De-Risk Threshold Score", 80, 95, 90, 1)
        derisk_cash = st.slider("De-Risk Cash Allocation", 0.0, 0.5, 0.20, 0.05)
        cash_yield = st.number_input("Cash Yield (Annualized %)", min_value=0.0,
                                     max_value=10.0, value=4.0, step=0.5)
    params = dict(base_monthly=base_monthly, low_mult=low_mult,
                  high_mult=high_mult, derisk_threshold=derisk_threshold,
                  derisk_cash=derisk_cash, cash_yield=cash_yield)

    # ---- Single network pass ---------------------------------------------
    with st.spinner("正在并发拉取最新宏观数据中，预计耗时 3 秒..."):
        scores, meta = load_scores(refresh=refresh, tail_boost=tail_boost)
        state = pipe.get_latest_state(refresh=False, tail_boost=tail_boost)

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

    # ---- Feature cards ---------------------------------------------------
    st.subheader("Eight Risk Features (current percentile)")
    feature_cards(state["features"])

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
    daily = load_daily_scores(refresh=False, tail_boost=tail_boost)
    spx, ndx = load_prices()
    st.plotly_chart(history_fig(daily if not daily.empty else scores, spx, ndx,
                                log_scale=log_scale, view=view),
                    use_container_width=True)

    # ---- Strategy backtest (interactive) --------------------------------
    backtest_panel(scores, params)


if __name__ == "__main__":
    main()
