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
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pipeline as pipe

LIVE_START = pipe.LIVE_START
MONTHLY_BUY = 1000.0
WEEKLY_BUY = MONTHLY_BUY * 12.0 / 52.0   # ~ $230.77/wk -> same annual flow

# Default backtest parameters. These reproduce the original fixed-schedule
# behaviour (low_mult / 1.5x / 1.0x / high_mult bands, 20% de-risk at >=90,
# cash modelled off the real SHY short-bond return since cash_yield defaults 0).
DEFAULT_PARAMS = {
    "base_monthly": 1000.0,    # base contribution per rebalance period
    "low_mult": 2.0,           # multiplier when score < 40
    "high_mult": 0.5,          # multiplier when 80 <= score < de-risk threshold
    "derisk_threshold": 90.0,  # score at/above which contribution -> 0x + de-risk
    "derisk_cash": 0.20,       # fraction of portfolio moved to cash on de-risk
    "cash_yield": 0.0,         # annualized cash return (%); 0 => use SHY return
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_prices(freq: str = "W") -> Tuple[pd.Series, pd.Series]:
    """Return (SPY, SHY) rebalanced to `freq` ('W' = W-FRI, 'M' = month-end)."""
    spy = pipe.get_price_series("SPY", start="1999-06-01")
    shy = pipe.get_price_series("SHY", start="1999-06-01")
    if spy is None:
        raise RuntimeError("Could not fetch SPY price history (yfinance/Stooq).")
    if shy is None:
        shy = pd.Series(0.0, index=spy.index)   # no short-bond proxy -> 0% cash

    rule = "W-FRI" if freq.upper().startswith("W") else "ME"
    spy = spy.resample(rule).last().dropna()
    shy = shy.resample(rule).last().ffill().dropna()
    return spy, shy


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(equity: pd.Series, rf_annual: float = 0.0, ppy: int = 52) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().dropna()
    cum = eq.iloc[-1] / eq.iloc[0] - 1.0
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan

    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    mdd = float(dd.min())

    vol = rets.std()
    sharpe = ((rets.mean() - rf_annual / ppy) / vol * np.sqrt(ppy)) if vol > 0 else 0.0
    calmar = (cagr / abs(mdd)) if mdd < 0 else np.nan

    return {
        "cum_return": float(cum),
        "cagr": float(cagr),
        "max_drawdown": mdd,
        "sharpe": float(sharpe),
        "calmar": float(calmar) if calmar == calmar else np.nan,
        "end_value": float(eq.iloc[-1]),
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

    timing=True  -> scale contribution by the Bubble Risk Score band and
                    rebalance toward a cash sleeve when score >= de-risk
                    threshold (idempotent target: re-deploys to equity when
                    risk fades).
    timing=False -> pure buy & hold benchmark (fixed contribution, no de-risk).

    Returns (equity_series, derisk_dates).
    """
    p = params or DEFAULT_PARAMS
    low_mult = float(p["low_mult"])
    high_mult = float(p["high_mult"])
    thr = float(p["derisk_threshold"])
    cash_frac = float(p["derisk_cash"])
    cash_yield = float(p["cash_yield"])
    fixed_growth = (1.0 + cash_yield / 100.0 / ppy) if cash_yield > 0 else None

    shares = 0.0
    cash = 0.0
    vals = []
    derisk_dates = []
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
            elif sc < 40:
                mult, derisk = low_mult, False
            elif sc < 60:
                mult, derisk = 1.5, False
            elif sc < 80:
                mult, derisk = 1.0, False
            elif sc < thr:
                mult, derisk = high_mult, False
            else:
                mult, derisk = 0.0, True
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
        else:
            shares += base_contrib / price_d

        prev_total = shares * price_d + cash
        vals.append(prev_total)
        prev_p = price_d

    return pd.Series(vals, index=dates), derisk_dates


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
    if shy is not None and prm["cash_yield"] <= 0:
        shy = shy.reindex(spy.index).ffill()
        shy_ret = shy.pct_change().fillna(0.0)
    else:
        shy_ret = pd.Series(0.0, index=spy.index)

    bench, _ = _simulate(spy, shy_ret, sc, dates, base, ppy, timing=False, params=prm)
    strat, derisk_dates = _simulate(spy, shy_ret, sc, dates, base, ppy,
                                     timing=True, params=prm)

    mb = metrics(bench, ppy=ppy)
    ms = metrics(strat, ppy=ppy)

    # ---- Metrics table (formatted, ready to display) ---------------------
    rows = []
    for name in ("cum_return", "cagr", "max_drawdown", "sharpe"):
        b, s = mb.get(name, np.nan), ms.get(name, np.nan)
        if name == "sharpe":
            rows.append({"Metric": "Sharpe Ratio",
                         "Benchmark (Buy & Hold DCA)": f"{b:.2f}",
                         "Bubble-DCA Strategy": f"{s:.2f}"})
        else:
            label = {"cum_return": "Total Return",
                     "cagr": "CAGR",
                     "max_drawdown": "Max Drawdown"}[name]
            rows.append({"Metric": label,
                         "Benchmark (Buy & Hold DCA)": f"{b*100:.1f}%",
                         "Bubble-DCA Strategy": f"{s*100:.1f}%"})
    metrics_df = pd.DataFrame(rows)

    # ---- Chart: equity curves + de-risk markers --------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values,
                             name="Benchmark (Buy & Hold DCA)",
                             line={"color": "#1f4e79", "width": 2}))
    fig.add_trace(go.Scatter(x=strat.index, y=strat.values,
                             name="Bubble Risk-Adjusted DCA",
                             line={"color": "#c1121f", "width": 2}))
    if derisk_dates:
        dm = pd.Series([strat.get(d, np.nan) for d in derisk_dates],
                       index=derisk_dates).dropna()
        if not dm.empty:
            fig.add_trace(go.Scatter(x=dm.index, y=dm.values, mode="markers",
                                     name="De-risk triggered",
                                     marker={"color": "#e4572e", "size": 7,
                                             "symbol": "triangle-down"},
                                     hovertemplate="De-risk @ %{x|%Y-%m}<extra></extra>"))
    fig.update_layout(height=440, hovermode="x unified",
                      yaxis_title="Portfolio Value (USD)",
                      margin={"t": 30, "b": 30, "l": 75, "r": 30},
                      legend=dict(orientation="h", y=1.06, x=0),
                      plot_bgcolor="white", paper_bgcolor="white")

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

    # ---- Drawdown comparison during classic tops -------------------------
    bench, derisk = _simulate(
        spy, shy.pct_change().fillna(0.0), scores_ff, list(spy.index),
        base, ppy, timing=True, params=prm)
    # re-run benchmark for the trough comparison
    bench_eq, _ = _simulate(
        spy, shy.pct_change().fillna(0.0), scores_ff, list(spy.index),
        base, ppy, timing=False, params=prm)
    windows = {
        "2000 Dot-com": ("2000-03-01", "2002-12-31"),
        "2008 GFC": ("2007-10-01", "2009-06-30"),
        "2021 COVID-tech": ("2021-01-01", "2022-12-31"),
    }
    print("\nDrawdown comparison during classic tops (strategy vs benchmark):")
    for name, (s, e) in windows.items():
        t = _trough_during(bench_eq, bench, s, e)
        if t:
            print(f"  {name:<18} bench MDD {t['bench_mdd']*100:6.1f}%  |  "
                  f"strategy MDD {t['strat_mdd']*100:6.1f}%  |  "
                  f"avoided {t['avoided_pp']:5.1f} pp")
    return {"metrics_df": metrics_df}


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
