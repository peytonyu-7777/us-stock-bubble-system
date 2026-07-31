"""
app.py — Streamlit dashboard for the Dalio-style US Equity Bubble Risk Score.

Sections
   1. Top   : gauge (current score) with the 5 risk bands and 2000/2021 refs.
   2. Middle: 8 feature cards (current sub-score + red/green status).
   3. Bottom: S&P 500 / Nasdaq main chart + Bubble Risk Score history with
              the >80 high-risk zone shaded.

Deploy: `streamlit run app.py`  (Render / HuggingFace Spaces ready).
Live data needs a FRED_API_KEY env var; without it the app still runs on cache
or, as a last resort, a clearly-flagged synthetic series.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import pipeline as pipe

st.set_page_config(page_title="US Equity Bubble Risk", page_icon="📈",
                   layout="wide", initial_sidebar_state="auto")

# --- Responsive layout (mobile + desktop) ---------------------------------
# Streamlit columns don't auto-wrap, so we (a) force a fluid main container,
# (b) stack horizontal column blocks below ~720px, and (c) render the 8
# feature cards as an auto-fit CSS grid that reflows from 8 -> 1 columns.
st.markdown("""
<style>
.main .block-container{max-width:1200px;padding-top:1.2rem;padding-bottom:2rem}
@media (max-width:720px){
  .stHorizontalBlock{flex-direction:column !important}
  .stHorizontalBlock>div{width:100% !important;min-width:0 !important}
}
.feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px;margin-top:6px}
.feat-card{border:1px solid #e5e7eb;border-radius:10px;padding:12px;
  background:#fff;text-align:center}
.feat-card.missing{opacity:.55;background:#f9fafb;border-style:dashed}
.feat-lbl{font-size:12px;color:#555}
.feat-val{font-size:30px;font-weight:700}
.feat-w{font-size:11px;color:#888}
</style>
""", unsafe_allow_html=True)

REF_2000 = 83.2   # historical reference points (approx, per Dalio-style framing)
REF_2021 = 92.1

BANDS = [  # (lo, hi, color, label)
    (0, 40, "forestgreen", "Low / Cooling"),
    (40, 60, "gold", "Normal"),
    (60, 80, "darkorange", "Watch"),
    (80, 90, "crimson", "Elevated"),
    (90, 100, "darkred", "Bubble Warning"),
]


@st.cache_data(ttl=3600)
def load_scores(refresh: bool):
    return pipe.get_monthly_scores(refresh=refresh)


@st.cache_data(ttl=3600)
def load_prices():
    spx = pipe.get_price_series("^GSPC", start=pipe.LIVE_START)
    ndx = pipe.get_price_series("^IXIC", start=pipe.LIVE_START)
    return spx, ndx


def band_color(score: float) -> str:
    if pd.isna(score):
        return "gray"
    for lo, hi, c, _ in BANDS:
        if lo <= score < hi:
            return c
    return "darkred"


def gauge_fig(score: float) -> go.Figure:
    steps = [{"range": [lo, hi], "color": c} for lo, hi, c, _ in BANDS]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score if not pd.isna(score) else 0,
        number={"font": {"size": 56}, "suffix": " / 100"},
        delta={"reference": REF_2021, "increasing": {"color": "crimson"},
               "decreasing": {"color": "forestgreen"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": band_color(score)},
            "steps": steps,
            "threshold": {"line": {"color": "black", "width": 3}, "value": score},
        },
        title={"text": "US Equity Bubble Risk Score", "font": {"size": 20}},
    ))
    fig.add_annotation(x=0.5, y=-0.05,
                       text=f"Refs — 2000: {REF_2000}  |  2021: {REF_2021}",
                       showarrow=False, font={"size": 12, "color": "gray"})
    fig.update_layout(height=380, margin={"t": 50, "b": 30, "l": 30, "r": 30})
    return fig


def feature_cards(features: dict):
    cells = ""
    for key, f in features.items():
        sc = f["score"]
        if sc is None:
            color, txt, cls, avail = "gray", "Pending", " missing", "pending"
        else:
            color = band_color(sc)
            txt = f"{sc:.0f}"
            cls, avail = "", "live"
        cells += (
            f'<div class="feat-card{cls}">'
            f'<div class="feat-lbl">{f["label"]}</div>'
            f'<div class="feat-val" style="color:{color}">{txt}</div>'
            f'<div class="feat-w">w {f["weight"]:.2f} · {avail}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="feat-grid">{cells}</div>', unsafe_allow_html=True)


def history_fig(scores: pd.Series, spx, ndx) -> go.Figure:
    START = pd.Timestamp("2000-01-01")
    # Align EVERY series to the common window [2000-01-01, today] so the price
    # chart and the score chart never start at different years (old bug: prices
    # from 2001, score from 1995). shared_xaxes=True then links zoom/pan
    # between the two subplots.
    sc = scores.dropna()
    sc = sc[sc.index >= START]
    if spx is not None:
        spx = spx[spx.index >= START]
    if ndx is not None:
        ndx = ndx[ndx.index >= START]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("S&P 500 / Nasdaq Composite (log)", "Bubble Risk Score"),
    )
    if spx is not None and not spx.empty:
        fig.add_trace(go.Scatter(x=spx.index, y=spx.values, name="S&P 500",
                                 line={"color": "steelblue"}),
                      row=1, col=1)
    if ndx is not None and not ndx.empty:
        # `yaxis` is a TOP-LEVEL trace property, not a property of `line`.
        # Nesting it inside `line={...}` raises the scatter.Line 'yaxis' error.
        # S&P and Nasdaq share the row-1 log axis, so `yaxis="y"` (default).
        fig.add_trace(go.Scatter(x=ndx.index, y=ndx.values, name="Nasdaq",
                                 yaxis="y", line={"color": "purple"}),
                      row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    if not sc.empty:
        fig.add_trace(go.Scatter(x=sc.index, y=sc.values, name="Bubble Score",
                                 line={"color": "crimson", "width": 2},
                                 fill="tozeroy", fillcolor="rgba(220,20,60,0.08)"),
                      row=2, col=1)
        # shade the >80 high-risk zone on the score subplot
        fig.add_hrect(y0=80, y1=100, line_width=0, fillcolor="rgba(220,20,60,0.12)",
                      row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="crimson",
                      annotation_text="High-risk > 80", row=2, col=1)
        # mark individual >80 points
        hi = sc[sc > 80]
        if not hi.empty:
            fig.add_trace(go.Scatter(x=hi.index, y=hi.values, mode="markers",
                                     name=">80", marker={"color": "darkred", "size": 5},
                                     showlegend=False),
                          row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified", showlegend=True,
                      margin={"t": 40, "b": 30, "l": 50, "r": 30})
    return fig


@st.cache_data(ttl=3600)
def load_spy():
    return pipe.get_price_series("SPY", start=pipe.LIVE_START)


def _max_drawdown(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    cummax = series.cummax()
    dd = series / cummax - 1.0
    return float(dd.min())


def _sharpe(returns):
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    sd = arr.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(arr.mean() / sd * np.sqrt(12))


def run_backtest(scores: pd.Series) -> Optional[dict]:
    """Monthly-DCA backtest, 2000 -> present.

    Benchmark : every month invest a flat $1,000 into SPY (buy & hold DCA).
    Strategy  : invest per the Bubble Score band (<40:2.0x, 40-60:1.5x,
                60-80:1.0x, 80-90:0.5x, >=90:0x) and, at score >= 90, move
                20% of equities into cash (BIL proxy, modelled at 0% here).

    Returns a dict with the metric table + the two portfolio-value Series,
    or None if SPY / scores are unavailable.
    """
    spy = load_spy()
    if spy is None or scores is None:
        return None
    s = scores.dropna()
    idx = spy.index.intersection(s.index)
    idx = idx[idx >= pd.Timestamp("2000-01-01")]
    if len(idx) < 24:
        return None
    spy = spy.loc[idx]
    s = s.loc[idx]
    n = len(idx)
    years = n / 12.0

    shares_b = 0.0          # benchmark equity shares
    shares_s = 0.0          # strategy equity shares
    cash_s = 0.0            # strategy cash bucket
    val_b, val_s = [], []
    ret_b, ret_s = [], []   # monthly time-weighted returns
    inv_b = 0.0
    inv_s = 0.0
    prev_p = None
    prev_total_s = 0.0

    for dt in idx:
        p = float(spy.loc[dt])
        sc = s.loc[dt]

        # ---------- Benchmark (always $1,000) ----------
        cb = 1000.0
        shares_b += cb / p
        inv_b += cb
        if prev_p is not None and prev_p > 0:
            ret_b.append(p / prev_p - 1.0)
        val_b.append(shares_b * p)

        # ---------- Strategy contribution by band ----------
        if pd.isna(sc):
            cs = 1000.0
        elif sc < 40:   cs = 2000.0
        elif sc < 60:   cs = 1500.0
        elif sc < 80:   cs = 1000.0
        elif sc < 90:   cs = 500.0
        else:           cs = 0.0
        shares_s += cs / p
        inv_s += cs

        # market move on the existing strategy book this month
        if prev_p is not None and prev_p > 0 and prev_total_s > 0:
            eq_pre = shares_s * p
            total_pre = eq_pre + cash_s
            ret_s.append((total_pre - prev_total_s) / prev_total_s)

        # de-risk 20% of equities -> cash when score >= 90
        if (not pd.isna(sc)) and sc >= 90:
            eq_now = shares_s * p
            move = 0.20 * eq_now
            if move > 0:
                shares_s -= move / p
                cash_s += move

        port_val = shares_s * p + cash_s
        val_s.append(port_val)
        prev_p = p
        prev_total_s = port_val

    bench = pd.Series(val_b, index=idx)
    strat = pd.Series(val_s, index=idx)

    twr_b = float(np.prod([1.0 + r for r in ret_b])) if ret_b else 1.0
    twr_s = float(np.prod([1.0 + r for r in ret_s])) if ret_s else 1.0
    cagr_b = twr_b ** (1.0 / years) - 1.0 if years > 0 else 0.0
    cagr_s = twr_s ** (1.0 / years) - 1.0 if years > 0 else 0.0
    tot_b = bench.iloc[-1] / inv_b - 1.0 if inv_b > 0 else 0.0
    tot_s = strat.iloc[-1] / inv_s - 1.0 if inv_s > 0 else 0.0
    mdd_b = _max_drawdown(bench)
    mdd_s = _max_drawdown(strat)
    sharpe_b = _sharpe(ret_b)
    sharpe_s = _sharpe(ret_s)

    return {
        "bench": bench, "strat": strat,
        "metrics": {
            "Total Return": (tot_b, tot_s),
            "CAGR": (cagr_b, cagr_s),
            "Max Drawdown": (mdd_b, mdd_s),
            "Sharpe": (sharpe_b, sharpe_s),
        },
    }


def backtest_panel(scores: pd.Series):
    st.subheader("📊 Strategy Historical Backtest (2000 - Present)")
    st.caption("Monthly $1,000 DCA into SPY · Benchmark vs Bubble Risk-Adjusted DCA "
               "(2.0× / 1.5× / 1.0× / 0.5× / 0×, with 20% de-risk to cash at ≥90).")
    res = run_backtest(scores)
    if res is None:
        st.warning("Backtest needs SPY price history + a scored series (cache). "
                   "Run the scoring first / ensure network connectivity.")
        return
    m = res["metrics"]
    rows = []
    for name, (b, s) in m.items():
        if name == "Sharpe":
            rows.append({"Metric": name,
                         "Benchmark (Buy&Hold DCA)": f"{b:.2f}",
                         "Bubble-DCA Strategy": f"{s:.2f}"})
        else:
            rows.append({"Metric": name,
                         "Benchmark (Buy&Hold DCA)": f"{b*100:.1f}%",
                         "Bubble-DCA Strategy": f"{s*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # asset-curve comparison
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["bench"].index, y=res["bench"].values,
                             name="Benchmark (Buy & Hold DCA)",
                             line={"color": "steelblue"}))
    fig.add_trace(go.Scatter(x=res["strat"].index, y=res["strat"].values,
                             name="Bubble Risk-Adjusted DCA",
                             line={"color": "crimson"}))
    fig.update_layout(height=430, hovermode="x unified",
                      yaxis_title="Portfolio Value (USD)",
                      margin={"t": 30, "b": 30, "l": 70, "r": 30},
                      legend=dict(orientation="h", y=1.06, x=0))
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📈 US Equity Bubble Risk — Dalio-style Monitor")
    st.caption("8-feature percentile model · rolling 20-year window · "
               "free/open data (FRED incl. EMVMACROBUS, yfinance/Stooq)")

    refresh = st.sidebar.checkbox("Force refresh live data", value=False)
    if st.sidebar.button("Re-run scoring"):
        st.cache_data.clear()
        refresh = True

    # Single network pass: load_scores fetches (concurrent, incremental cache)
    # and writes the unified cache; get_latest_state then reads that cache so
    # the dashboard never triggers a second fetch.
    with st.spinner("正在并发拉取最新宏观数据中，预计耗时 3 秒..."):
        scores, meta = load_scores(refresh=refresh)
        state = pipe.get_latest_state(refresh=False)

    src = meta.get("source", "unknown")
    src_label = {"live": "Real-time", "cache": "Cached", "synthetic": "Synthetic"}
    if src == "synthetic":
        st.warning("⚠️ Live data unavailable — showing a **deterministic synthetic** "
                   "series for layout/demo only. Set `FRED_API_KEY` and uncheck the "
                   "cache to fetch real data.")
    elif src == "cache":
        st.info("ℹ️ Showing cached data (set refresh to pull live).")

    # ---- Top gauge -------------------------------------------------------
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.plotly_chart(gauge_fig(state["score"]), use_container_width=True)
    with c2:
        st.markdown(f"### Status: **{state['status']}**")
        st.markdown(f"**Score:** `{state['score']:.1f} / 100`  "
                    f"(as of {pd.Timestamp(state['as_of']).date() if state['as_of'] else 'n/a'})")
        st.markdown("**DCA rule (monthly):**")
        st.markdown("- 0–40 : 2.0×  - 40–60 : 1.5×  - 60–80 : 1.0×  "
                    "- 80–90 : 0.5×  - 90–100 : 0× (de-risk 20% → cash)")
        st.markdown(f"**Data source:** `{src_label.get(src, src)}` · "
                    f"Real-time coverage: `{meta.get('available_count','?')}/8 live`")

    # ---- Feature cards ---------------------------------------------------
    st.subheader("Eight Risk Features (current percentile)")
    feature_cards(state["features"])

    # ---- History ---------------------------------------------------------
    st.subheader("Historical Trend")
    spx, ndx = load_prices()
    st.plotly_chart(history_fig(scores, spx, ndx), use_container_width=True)

    # ---- Strategy backtest (full panel) ----------------------------------
    backtest_panel(scores)


if __name__ == "__main__":
    main()
