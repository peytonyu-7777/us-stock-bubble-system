"""
backtest.py — Historical backtest of the Dalio-style Bubble Risk Score.

Compares, from 2000-01 to today, two DCA strategies on SPY:

  1. BENCHMARK  : fixed contribution every rebalance period, buy & hold
                  (no timing). Default $1,000/mo -> ~$230.77/wk so the
                  ANNUAL cash flow is identical across frequencies.
  2. BUBBLE-DCA : contribution scaled by the Bubble Risk Score band
                  (low_mult / 1.5x / 1.0x / high_mult / 0.0x), and when the
                  score >= derisk_threshold the portfolio is rebalanced toward
                  a `derisk_cash` cash sleeve (idempotent target, so it
                  re-deploys when risk fades).

Public engine
-------------
  run_backtest(scores, spy_df, params) -> (metrics_df, chart_fig) | None
      * scores   : pd.Series of the Bubble Risk Score (any frequency).
      * spy_df   : pd.Series of SPY prices at the SAME frequency.
      * params   : dict with base_monthly, low_mult, high_mult,
                   derisk_threshold, derisk_cash, cash_yield.
      * Returns None when inputs are missing / too short (guard clauses) so the
        caller can render a friendly card instead of crashing on an empty
        series or an unpack of `ret_b, ret_s = []`.

Run:
    python backtest.py            # weekly, uses cached/live score history
    python backtest.py --refresh  # force re-fetch of the score history
    python backtest.py --freq M   # monthly rebalancing instead
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pipeline as pipe

LIVE_START = pipe.LIVE_START
MONTHLY_BUY = 1000.0
WEEKLY_BUY = MONTHLY_BUY * 12.0 / 52.0   # ~ $230.77/wk -> same annual flow

# Default backtest parameters. These reproduce the original fixed-schedule
# behaviour (low_mult / 1.5x / 1.0x / high_mult bands, 20% de-risk at >=90,
# cash modelled off the real SHY short-bond return since cash_yield defaults 0).
DEFAULT_PARAMS = {
    "base_monthly": 1000.0,    # base contribution per rebalance period
    "low_mult": 3.0,           # deep-value deploy cap: base + up to (low_mult-1)x from reserve
    "mid_mult": 1.5,           # value deploy cap: base + up to (mid_mult-1)x from reserve
    "high_mult": 0.5,          # taper fraction at band_high <= score < de-risk threshold
    "band_low": 40.0,          # deep-value threshold (crash readings)
    "band_mid": 50.0,          # value threshold (below-median risk)
    "band_high": 80.0,         # neutral cap (above -> taper, stockpile reserve)
    "derisk_threshold": 95.0,  # score at/above which deploy -> 0 + de-risk fires
    "derisk_cash": 0.15,       # target cash-sleeve fraction on de-risk
    "cash_yield": 0.0,         # annualized cash return (%); 0 => use SHY return
    "recycle": False,          # True => reserve-recycle DCA (same outflow as DCA);
                               # False => multiplier mode (default — more dollars at
                               # lows, fewer at highs; visibly diverges from DCA)
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _fred_cash_returns(idx: pd.DatetimeIndex, ppy: int) -> Optional[pd.Series]:
    """Per-period cash return from FRED DGS3MO (3-month T-bill), read from the
    pipeline parquet cache (no network). Last-resort for when SHY cannot be
    fetched at all (Yahoo 429 + Stooq anti-bot wall).
    """
    try:
        if not os.path.exists(pipe.CACHE_PATH):
            return None
        d = pd.read_parquet(pipe.CACHE_PATH).get("dgs3mo")
        if d is None or not d.notna().any():
            return None
        r = (d.dropna() / 100.0) / float(ppy)      # yield% -> per-period return
        return r.reindex(idx, method="ffill").fillna(0.0)
    except Exception:
        return None


def _load_prices(freq: str = "W") -> Tuple[pd.Series, pd.Series]:
    """Return (SPY, SHY) rebalanced to `freq` ('W' = W-FRI, 'M' = month-end)."""
    spy = pipe.get_price_series("SPY", start="1999-06-01")
    shy = pipe.get_price_series("SHY", start="1999-06-01")
    if spy is None:
        raise RuntimeError("Could not fetch SPY price history (yfinance/Stooq/FRED).")
    if shy is None:
        shy = pd.Series(0.0, index=spy.index)   # no short-bond proxy -> 0% cash

    rule = "W-FRI" if freq.upper().startswith("W") else "ME"
    spy = spy.resample(rule).last().dropna()
    shy = shy.resample(rule).last().ffill().dropna()
    return spy, shy


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _irr(cashflows: list, ppy: int) -> float:
    """Money-weighted return (annualized) via bisection on the per-period rate.

    ``cashflows`` is the net cashflow per period (negative = contribution out,
    positive = withdrawal) with the FINAL portfolio value appended as the last
    positive inflow. This is the HONEST return for a DCA strategy — it weights
    each dollar by when it was invested, unlike a naive end/start equity ratio
    which conflates ongoing contributions with investment growth.
    """
    if len(cashflows) < 2:
        return np.nan
    lo, hi = -0.95, 2.0          # per-period search bounds
    for _ in range(80):
        r = 0.5 * (lo + hi)
        npv = sum(cf / (1.0 + r) ** t for t, cf in enumerate(cashflows))
        if npv > 0:
            lo = r
        else:
            hi = r
    r = 0.5 * (lo + hi)
    return float((1.0 + r) ** ppy - 1.0)


def metrics(equity: pd.Series, contrib: pd.Series,
            rf_annual: float = 0.0, ppy: int = 12) -> dict:
    """Honest DCA metrics.

    * total_invested : sum of all contributions (the real cash the investor put in)
    * final_value    : ending portfolio value
    * mwr            : money-weighted (IRR) annualized return
    * max_drawdown   : worst peak-to-trough decline of the account value
    * sharpe         : on contribution-STRIPPED (time-weighted) returns, so it is
                       not inflated by the DCA cashflows
    """
    eq = equity.dropna()
    cf = contrib.reindex(eq.index).fillna(0.0)
    if len(eq) < 2:
        return {}
    total_invested = float(cf.sum())
    final_value = float(eq.iloc[-1])

    # Max drawdown of the account value (peak-to-trough).
    roll_max = eq.cummax()
    mdd = float((eq / roll_max - 1.0).min())

    # Time-weighted (contribution-STRIPPED) returns: the value BEFORE the
    # end-of-period contribution is (eq - cf); its period-over-period growth is
    # the pure market return, undistorted by how much cash was added. Using this
    # (instead of a naive end/start equity ratio) is exactly what makes the
    # backtest honest for a DCA strategy — it never conflates new contributions
    # with investment growth.
    mkt = (eq - cf).replace(0, np.nan)
    twr_ret = (mkt / mkt.shift(1) - 1.0).dropna()
    vol = twr_ret.std()
    sharpe = float(((twr_ret.mean() - rf_annual / ppy) / vol * np.sqrt(ppy))
                   if vol and vol > 0 else 0.0)

    # Money-weighted return (IRR): per-period OUTflows (contributions) plus the
    # final portfolio value as the terminal INflow. This is the real investor
    # return because it weights every dollar by *when* it entered the market.
    cfs = ([-c for c in cf.values] + [final_value])
    mwr = _irr(cfs, ppy)

    return {
        "total_invested": total_invested,
        "final_value": final_value,
        "mwr": mwr,
        "max_drawdown": mdd,
        "sharpe": sharpe,
    }


def _trough_during(eq_bench: pd.Series, eq_strat: pd.Series,
                   start: str, end: str) -> dict:
    """Compare benchmark vs strategy drawdown trough inside a window."""
    b = eq_bench.loc[start:end]
    s = eq_strat.loc[start:end]
    if b.empty or s.empty:
        return {}
    b_dd = (b / b.cummax() - 1.0).min()
    s_dd = (s / s.cummax() - 1.0).min()
    return {
        "bench_mdd": float(b_dd),
        "strat_mdd": float(s_dd),
        "avoided_pp": float((s_dd - b_dd) * 100.0),  # positive = strategy shallower
    }


# ---------------------------------------------------------------------------
# Unified simulator (returns equity + de-risk trigger dates)
# ---------------------------------------------------------------------------
def _simulate(price: pd.Series, shy_ret: pd.Series, scores: pd.Series,
              dates: list, base_contrib: float, ppy: int,
              timing: bool = True, params: dict = None) -> Tuple[pd.Series, list]:
    """
    Walk `dates`, buying SPY with `base_contrib` each period.

    timing=True  -> deploy according to the Bubble Risk Score band. Two modes:
                    - recycle=True (default): the investor ALWAYS puts the same
                      base amount in (identical cash outflow to plain DCA); the
                      index only decides how much of the cash on hand to DEPLOY.
                      High risk -> taper & stockpile a cash reserve; low risk ->
                      deploy base + draw the reserve down; extreme -> de-risk
                      toward a cash sleeve (idempotent target: re-deploys when
                      risk fades). Total invested == DCA, so the IRR comparison
                      isolates the index's TIMING skill.
                    - recycle=False (legacy): the contribution itself is scaled
                      by the band multiplier.
    timing=False -> pure buy & hold benchmark (fixed contribution, no de-risk).

    Returns (equity_series, contribution_series, derisk_dates).
    """
    p = params or DEFAULT_PARAMS
    low_mult = float(p["low_mult"])
    mid_mult = float(p.get("mid_mult", 1.5))
    high_mult = float(p["high_mult"])
    b_low = float(p.get("band_low", 40.0))      # deep-value band: score < b_low -> low_mult
    b_mid = float(p.get("band_mid", 50.0))      # value band:      score < b_mid -> mid_mult
    b_high = float(p.get("band_high", 80.0))    # neutral band:    score < b_high -> 1.0
    thr = float(p["derisk_threshold"])
    cash_frac = float(p["derisk_cash"])
    cash_yield = float(p["cash_yield"])
    recycle = bool(p.get("recycle", False))
    fixed_growth = (1.0 + cash_yield / 100.0 / ppy) if cash_yield > 0 else None

    shares = 0.0
    cash = 0.0
    vals = []
    contribs = []
    derisk_dates = []
    cash_history = []           # per-period end-of-period cash reserve
    prev_p = None
    prev_total = 0.0
    for i, d in enumerate(dates):
        if i > 0:
            if fixed_growth is not None:
                cash *= fixed_growth
            else:
                cash *= (1.0 + float(shy_ret.get(d, 0.0)))

        price_d = float(price[d])
        sc = scores.get(d, np.nan) if timing else np.nan

        if timing:
            if pd.isna(sc):
                mult, derisk = 1.0, False
            elif sc < b_low:
                mult, derisk = low_mult, False
            elif sc < b_mid:
                mult, derisk = mid_mult, False
            elif sc < b_high:
                mult, derisk = 1.0, False
            elif sc < thr:
                mult, derisk = high_mult, False
            else:
                mult, derisk = 0.0, True
            if recycle:
                # RESERVE-RECYCLE mode: the investor always contributes the
                # same base amount (identical cash outflow to plain DCA), the
                # index only decides how much of the cash on hand to DEPLOY.
                #   high risk  -> deploy less, stockpile the rest (earns SHY)
                #   low risk   -> deploy the base AND draw the stockpile down
                # so total invested == DCA invested and the IRR comparison is
                # a pure measure of the index's timing skill.
                pocket = base_contrib
                if derisk:
                    deploy = 0.0
                elif mult < 1.0:
                    deploy = pocket * mult
                elif mult > 1.0:
                    deploy = pocket + min((mult - 1.0) * pocket, cash)
                else:
                    deploy = pocket
                shares += deploy / price_d
                cash += pocket - deploy
                if derisk:
                    total = shares * price_d + cash
                    desired_cash = cash_frac * total
                    if desired_cash > cash:
                        move = desired_cash - cash
                        shares -= move / price_d
                        cash += move
                    elif desired_cash < cash:
                        move = cash - desired_cash
                        cash -= move
                        shares += move / price_d
                    derisk_dates.append(d)
                contribs.append(pocket)
            else:
                shares += base_contrib * mult / price_d
                if derisk:
                    total = shares * price_d + cash
                    desired_cash = cash_frac * total
                    if desired_cash > cash:
                        move = desired_cash - cash
                        shares -= move / price_d
                        cash += move
                    elif desired_cash < cash:
                        move = cash - desired_cash
                        cash -= move
                        shares += move / price_d
                    derisk_dates.append(d)
                contribs.append(base_contrib * mult)
        else:
            shares += base_contrib / price_d
            contribs.append(base_contrib)

        prev_total = shares * price_d + cash
        vals.append(prev_total)
        cash_history.append(cash)
        prev_p = price_d

    return (pd.Series(vals, index=dates),
            pd.Series(contribs, index=dates),
            derisk_dates,
            pd.Series(cash_history, index=dates))


# ---------------------------------------------------------------------------
# Crash-proof engine — returns (metrics_df, chart_fig) or None
# ---------------------------------------------------------------------------
def run_backtest(scores: Optional[pd.Series], spy_df: Optional[pd.Series],
                 params: dict) -> Optional[Tuple[pd.DataFrame, go.Figure]]:
    """
    Bubble Risk-Adjusted DCA vs Buy & Hold, 2000 -> present.

    Parameters
    ----------
    scores  : Bubble Risk Score series (monthly recommended) — a pd.Series.
    spy_df  : SPY price series at the SAME frequency — a pd.Series.
    params  : dict (base_monthly, low_mult, high_mult, derisk_threshold,
              derisk_cash, cash_yield). Missing keys fall back to DEFAULT_PARAMS.

    Returns
    -------
    (metrics_df, chart_fig) on success, or None when data is insufficient
    (guard clauses — never raises, never returns a partial / empty tuple that
    would blow up an `a, b = run_backtest(...)` unpacking).
    """
    # ---- Guard clauses (strict, no exception path) -----------------------
    if scores is None or spy_df is None:
        return None
    scores = pd.Series(scores).dropna()
    if len(scores) < 30:
        return None

    prm = dict(DEFAULT_PARAMS)
    prm.update({k: v for k, v in (params or {}).items() if k in DEFAULT_PARAMS})

    # ---- Inner-join alignment + NaN drop across prices & scores ----------
    aligned = pd.concat(
        [pd.Series(spy_df).rename("spy"),
         scores.rename("score")], axis=1, join="inner"
    ).dropna()
    aligned = aligned[aligned.index >= pd.Timestamp(LIVE_START)]
    dates = list(aligned.index.sort_values())
    if len(dates) < 30:
        return None

    spy = aligned["spy"]
    sc = aligned["score"]
    ppy = 12                      # scores are monthly -> 12 periods / year
    base = float(prm["base_monthly"])

    # SHY cash return (or fixed money-market if cash_yield > 0)
    shy = pipe.get_price_series("SHY", start="1999-06-01")
    shy_ret = None
    if shy is not None and prm["cash_yield"] <= 0:
        shy = shy.reindex(spy.index).ffill()
        shy_ret = shy.pct_change().fillna(0.0)
    elif prm["cash_yield"] <= 0:
        # SHY unavailable (Yahoo 429 + Stooq wall) -> FRED 3M T-bill from cache
        shy_ret = _fred_cash_returns(spy.index, ppy)
    if shy_ret is None:
        shy_ret = pd.Series(0.0, index=spy.index)

    bench, bench_cf, _, bench_cash = _simulate(spy, shy_ret, sc, dates, base, ppy,
                                                timing=False, params=prm)
    strat, strat_cf, derisk_dates, strat_cash = _simulate(spy, shy_ret, sc, dates, base, ppy,
                                                          timing=True, params=prm)

    mb = metrics(bench, bench_cf, ppy=ppy)
    ms = metrics(strat, strat_cf, ppy=ppy)

    # ---- Metrics table (formatted, ready to display) ---------------------
    # HONEST framing: Total Invested / Final Value / Total Gain show the real
    # dollars; "Money-Weighted Return (IRR)" replaces the old misleading naive
    # "Total Return" (end/start equity ratio) which conflated contributions
    # with investment growth for a DCA strategy.
    def _money(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.0f}"
    def _pct(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}%"
    def _num(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"

    gain_b = (mb.get("final_value") or 0) - (mb.get("total_invested") or 0)
    gain_s = (ms.get("final_value") or 0) - (ms.get("total_invested") or 0)
    rows = [
        {"Metric": "Total Invested",
         "Benchmark (Buy & Hold DCA)": _money(mb.get("total_invested")),
         "Bubble-DCA Strategy": _money(ms.get("total_invested"))},
        {"Metric": "Final Value",
         "Benchmark (Buy & Hold DCA)": _money(mb.get("final_value")),
         "Bubble-DCA Strategy": _money(ms.get("final_value"))},
        {"Metric": "Total Gain ($)",
         "Benchmark (Buy & Hold DCA)": _money(gain_b),
         "Bubble-DCA Strategy": _money(gain_s)},
        {"Metric": "Money-Weighted Return (IRR)",
         "Benchmark (Buy & Hold DCA)": _pct(mb.get("mwr")),
         "Bubble-DCA Strategy": _pct(ms.get("mwr"))},
        {"Metric": "Max Drawdown",
         "Benchmark (Buy & Hold DCA)": _pct(mb.get("max_drawdown")),
         "Bubble-DCA Strategy": _pct(ms.get("max_drawdown"))},
        {"Metric": "Sharpe Ratio",
         "Benchmark (Buy & Hold DCA)": _num(mb.get("sharpe")),
         "Bubble-DCA Strategy": _num(ms.get("sharpe"))},
    ]
    metrics_df = pd.DataFrame(rows)

    # ---- Chart: equity curves (row 1, dual axis) + strategy cash reserve (row 2)
    # Row 2 lets you SEE the recycle-mode stockpile/drawdown cycle; in multiplier
    # mode the cash balance is essentially zero (all dollars go to SPY each month).
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
        row_heights=[0.68, 0.32],
        subplot_titles=("Portfolio Value (Strategy vs Benchmark)",
                        "Strategy Cash Reserve ($)"),
        specs=[[{"secondary_y": True}], [{}]],
    )
    # NOTE: secondary_y is an add_trace() argument, NOT a go.Scatter property —
    # passing it inside go.Scatter(...) raises "Invalid property" in Plotly's
    # _process_kwargs (this is what crashed the backtest panel on Render).
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values,
                             name="Benchmark (Buy & Hold DCA)",
                             line={"color": "#1f4e79", "width": 2}),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=strat.index, y=strat.values,
                             name="Bubble Risk-Adjusted DCA",
                             line={"color": "#c1121f", "width": 2.5}),
                  row=1, col=1, secondary_y=False)

    # Cash reserve (row 2): filled area, dashed zero line for reference.
    if strat_cash is not None and strat_cash.abs().sum() > 0:
        fig.add_trace(go.Scatter(x=strat_cash.index, y=strat_cash.values,
                                 name="Strategy cash reserve",
                                 line={"color": "#7c3aed", "width": 1.5,
                                       "dash": "dot"},
                                 fill="tozeroy",
                                 fillcolor="rgba(124,58,237,0.10)",
                                 hovertemplate="Reserve: $%{y:,.0f}<extra></extra>"),
                      row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="#e2e8f0",
                      row=2, col=1)
        fig.update_yaxes(title_text="Reserve ($)", row=2, col=1,
                         tickformat="$,.0f")

    # V2 risk-band horizontal shading + labels INSIDE the plot (right edge,
    # anchored to the score axis) — outside labels got clipped/overlapped.
    band_tints = ["#e8f0fe", "#e6f4ea", "#fef6e0", "#fde7d3", "#fbe2e2"]
    for i, (lo, hi, _color, label) in enumerate(pipe.RISK_BANDS):
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1,
                      yref="y2", y0=lo, y1=hi,
                      fillcolor=band_tints[i], opacity=0.30, line_width=0,
                      layer="below", row=1, col=1)
        # label slightly inset (paper-x 0.985), tiny font, no border — won't
        # collide with the score line or the secondary-y axis.
        fig.add_annotation(xref="paper", x=0.985, yref="y2", y=(lo + hi) / 2,
                           text=label, showarrow=False, xanchor="right",
                           yanchor="middle", xshift=-4,
                           font={"size": 8, "color": "#6b7280"},
                           bgcolor="rgba(255,255,255,0.0)",
                           row=1, col=1)

    # Bubble Risk Score context line (right axis) — explains the de-risk calls.
    fig.add_trace(go.Scatter(x=sc.index, y=sc.values,
                             name="Bubble Risk Score",
                             line={"color": "#888888", "width": 1, "dash": "dot"},
                             opacity=0.65),
                  row=1, col=1, secondary_y=True)

    if derisk_dates:
        dm = pd.Series([strat.get(d, np.nan) for d in derisk_dates],
                       index=derisk_dates).dropna()
        if not dm.empty:
            fig.add_trace(go.Scatter(x=dm.index, y=dm.values, mode="markers",
                                     name="De-risk triggered",
                                     marker={"color": "#e4572e", "size": 7,
                                             "symbol": "triangle-down"},
                                     hovertemplate="De-risk @ %{x|%Y-%m}<extra></extra>"),
                          row=1, col=1, secondary_y=False)

    fig.update_layout(height=620, hovermode="x unified",
                      margin={"t": 55, "b": 50, "l": 80, "r": 80},
                      legend=dict(orientation="h", y=1.10, x=1,
                                  xanchor="right", yanchor="bottom",
                                  bgcolor="rgba(255,255,255,0.6)",
                                  bordercolor="#e2e8f0", borderwidth=0),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(size=11))
    # Shorter, non-overlapping axis titles + side margins so labels breathe.
    # Explicit autorange so the chart auto-stretches to the selected time
    # window (the user asked for auto-rescale after changing the range).
    fig.update_yaxes(title_text="USD", row=1, col=1, secondary_y=False,
                     title_font=dict(size=11), autorange=True)
    fig.update_yaxes(title_text="Score", row=1, col=1, secondary_y=True,
                     title_font=dict(size=11), range=[0, 100])
    fig.update_yaxes(autorange=True, row=2, col=1)
    fig.update_xaxes(title_text="", row=1, col=1, autorange=True)
    fig.update_xaxes(title_text="Date", row=2, col=1, autorange=True,
                     title_font=dict(size=11))

    return metrics_df, fig


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main(refresh: bool = False, freq: str = "W", params: dict = None) -> dict:
    freq = freq.upper()
    is_weekly = freq.startswith("W")
    ppy = 52 if is_weekly else 12
    base = WEEKLY_BUY if is_weekly else MONTHLY_BUY
    period_label = "weekly" if is_weekly else "monthly"
    prm = dict(DEFAULT_PARAMS)
    if params:
        prm.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})

    scores, meta = pipe.get_monthly_scores(refresh=refresh)
    spy, shy = _load_prices(freq)

    scores_ff = scores.reindex(spy.index, method="ffill")
    res = run_backtest(scores_ff, spy, prm)
    if res is None:
        print("BACKTEST: insufficient overlapping price/score data; skipping report.")
        return {}

    metrics_df, _ = res
    print("=" * 72)
    print(f"BUBBLE-RISK BACKTEST   freq={period_label}   source={meta.get('source')}")
    print(metrics_df.to_string(index=False))
    print("=" * 72)

    # ---- Per-side metric dicts (honest) + drawdown comparison ------------
    # Reuse the same simulations so the CLI / report and the dashboard agree.
    strat_eq, strat_cf, _, _ = _simulate(
        spy, shy.pct_change().fillna(0.0), scores_ff, list(spy.index),
        base, ppy, timing=True, params=prm)
    # benchmark (no timing) for the metrics + trough comparison
    bench_eq, bench_cf, _, _ = _simulate(
        spy, shy.pct_change().fillna(0.0), scores_ff, list(spy.index),
        base, ppy, timing=False, params=prm)
    mb = metrics(bench_eq, bench_cf, ppy=ppy)
    ms = metrics(strat_eq, strat_cf, ppy=ppy)

    windows = {
        "2000 Dot-com": ("2000-03-01", "2002-12-31"),
        "2008 GFC": ("2007-10-01", "2009-06-30"),
        "2021 COVID-tech": ("2021-01-01", "2022-12-31"),
    }
    tops = {}
    print("\nDrawdown comparison during classic tops (strategy vs benchmark):")
    for name, (s, e) in windows.items():
        t = _trough_during(bench_eq, strat_eq, s, e)
        if t:
            tops[name] = t
            print(f"  {name:<18} bench MDD {t['bench_mdd']*100:6.1f}%  |  "
                  f"strategy MDD {t['strat_mdd']*100:6.1f}%  |  "
                  f"avoided {t['avoided_pp']:5.1f} pp")

    return {
        "metrics_df": metrics_df,
        "freq": period_label,
        "benchmark": mb,
        "strategy": ms,
        "tops": tops,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="force re-fetch score history")
    ap.add_argument("--freq", choices=["W", "M"], default="W",
                    help="rebalancing frequency: W=weekly (default), M=monthly")
    ap.add_argument("--base", type=float, default=DEFAULT_PARAMS["base_monthly"],
                    help="base contribution per period (USD)")
    ap.add_argument("--low-mult", type=float, default=DEFAULT_PARAMS["low_mult"],
                    help="contribution multiplier when score < 40")
    ap.add_argument("--high-mult", type=float, default=DEFAULT_PARAMS["high_mult"],
                    help="contribution multiplier when 80<=score<threshold")
    ap.add_argument("--derisk-thr", type=float, default=DEFAULT_PARAMS["derisk_threshold"],
                    help="score at/above which contribution -> 0x and de-risk fires")
    ap.add_argument("--derisk-cash", type=float, default=DEFAULT_PARAMS["derisk_cash"],
                    help="fraction of portfolio moved to cash on de-risk (0-1)")
    ap.add_argument("--cash-yield", type=float, default=DEFAULT_PARAMS["cash_yield"],
                    help="annualized cash yield (%%); 0 = use real SHY return")
    args = ap.parse_args()
    cli_params = {
        "base_monthly": args.base,
        "low_mult": args.low_mult,
        "high_mult": args.high_mult,
        "derisk_threshold": args.derisk_thr,
        "derisk_cash": args.derisk_cash,
        "cash_yield": args.cash_yield,
    }
    try:
        main(refresh=args.refresh, freq=args.freq, params=cli_params)
    except Exception as exc:
        print(f"BACKTEST ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
