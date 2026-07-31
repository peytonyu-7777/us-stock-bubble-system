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
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("S&P 500 / Nasdaq Composite (log)", "Bubble Risk Score"),
    )
    if spx is not None:
        fig.add_trace(go.Scatter(x=spx.index, y=spx.values, name="S&P 500",
                                 line={"color": "steelblue"}),
                      row=1, col=1)
    if ndx is not None:
        fig.add_trace(go.Scatter(x=ndx.index, y=ndx.values, name="Nasdaq",
                                 line={"color": "purple", "yaxis": "y"}),
                      row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    sc = scores.dropna()
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

    with st.expander("Backtest snapshot (2000 → today)"):
        st.markdown("Run `python backtest.py` for the full Buy&Hold vs "
                    "Bubble-DCA comparison (CAGR, Max DD, Sharpe, Calmar).")
        try:
            import backtest as bt
            res = bt.main(refresh=False)
            mb, ms = res["benchmark"], res["strategy"]
            st.table(pd.DataFrame({
                "Metric": ["Cumulative", "CAGR", "Max DD", "Sharpe", "Calmar"],
                "Benchmark": [f"{mb['cum_return']*100:.1f}%", f"{mb['cagr']*100:.1f}%",
                              f"{mb['max_drawdown']*100:.1f}%", f"{mb['sharpe']:.2f}",
                              f"{mb['calmar']:.2f}"],
                "Bubble-DCA": [f"{ms['cum_return']*100:.1f}%", f"{ms['cagr']*100:.1f}%",
                               f"{ms['max_drawdown']*100:.1f}%", f"{ms['sharpe']:.2f}",
                               f"{ms['calmar']:.2f}"],
            }))
        except Exception as exc:
            st.error(f"Backtest could not run inline: {exc}")


if __name__ == "__main__":
    main()
