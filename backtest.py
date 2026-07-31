"""
backtest.py — Historical backtest of the Dalio-style Bubble Risk Score.

Compares, from 2000-01 to today, two DCA strategies on SPY:

  1. BENCHMARK  : fixed contribution every rebalance period, buy & hold
                  (no timing). Default $1,000/mo -> ~$230.77/wk so the
                  ANNUAL cash flow is identical across frequencies.
  2. BUBBLE-DCA : contribution scaled by the Bubble Risk Score band
                  (2.0x / 1.5x / 1.0x / 0.5x / 0.0x), and when the score > 90
                  the portfolio is rebalanced toward a 20% cash (SHY / T-bills)
                  sleeve (idempotent target, so it re-deploys when risk fades).

Default frequency is WEEKLY (`--freq W`). Monthly is supported via `--freq M`.

Reports cumulative return, CAGR, max drawdown, Sharpe and Calmar, and prints
the strategy vs benchmark trough during the three classic blow-off tops
(2000 dot-com, 2008 GFC, 2021 COVID-tech).

Run:
    python backtest.py            # weekly, uses cached/live score history
    python backtest.py --refresh  # force re-fetch of the score history
    python backtest.py --freq M   # monthly rebalancing instead
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd

import pipeline as pipe

LIVE_START = pipe.LIVE_START
MONTHLY_BUY = 1000.0
WEEKLY_BUY = MONTHLY_BUY * 12.0 / 52.0   # ~ $230.77/wk -> same annual flow

# Default backtest parameters. These reproduce the original fixed-schedule
# behaviour (2.0x / 1.5x / 1.0x / 0.5x / 0x bands, 20% de-risk at >=90, cash
# modelled off the real SHY short-bond return since cash_yield defaults to 0).
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
# Unified simulator (weekly or monthly)
# ---------------------------------------------------------------------------
def _simulate(price: pd.Series, shy_ret: pd.Series, scores: pd.Series,
              dates: list, base_contrib: float, ppy: int,
              timing: bool = True, params: dict = None) -> pd.Series:
    """
    Walk `dates`, buying SPY with `base_contrib` each period.

    timing=True  -> scale contribution by the Bubble Risk Score band and
                    rebalance toward a cash sleeve when score >= de-risk
                    threshold (idempotent target: re-deploys to equity when
                    risk fades).
    timing=False -> pure buy & hold benchmark (fixed contribution, no de-risk).

    `params` (see DEFAULT_PARAMS) controls the multipliers, de-risk threshold,
    cash allocation and cash yield. `ppy` is periods-per-year (52 weekly /
    12 monthly) used to convert the annualized cash yield to a per-period one.
    """
    p = params or DEFAULT_PARAMS
    low_mult = float(p["low_mult"])
    high_mult = float(p["high_mult"])
    thr = float(p["derisk_threshold"])
    cash_frac = float(p["derisk_cash"])
    cash_yield = float(p["cash_yield"])
    # When cash_yield > 0 we use a fixed money-market return; otherwise we keep
    # the real SHY short-bond return (backward-compatible default).
    fixed_growth = (1.0 + cash_yield / 100.0 / ppy) if cash_yield > 0 else None

    shares = 0.0
    cash = 0.0
    vals = []
    for i, d in enumerate(dates):
        # cash sleeve earns its return over the prior period
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
                if desired_cash > cash:            # sell equity -> raise cash
                    move = desired_cash - cash
                    shares -= move / price_d
                    cash += move
                elif desired_cash < cash:          # buy equity -> redeploy cash
                    move = cash - desired_cash
                    cash -= move
                    shares += move / price_d
        else:
            shares += base_contrib / price_d

        vals.append(shares * price_d + cash)
    return pd.Series(vals, index=dates)


# ---------------------------------------------------------------------------
# Main
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

    spy, shy = _load_prices(freq)
    scores, meta = pipe.get_monthly_scores(refresh=refresh)

    # Forward-fill the MONTHLY score onto the rebalance calendar so every
    # week (or month) is scored by the most recent month-end reading.
    idx = spy.index
    scores_ff = scores.reindex(idx, method="ffill")

    common = idx[idx >= pd.Timestamp(LIVE_START)].sort_values()
    dates = list(common)
    if not dates:
        raise RuntimeError("No overlapping periods between prices and scores.")

    shy_ret = shy.pct_change().fillna(0.0)

    bench = _simulate(spy, shy_ret, scores_ff, dates, base, ppy,
                      timing=False, params=prm)
    strat = _simulate(spy, shy_ret, scores_ff, dates, base, ppy,
                      timing=True, params=prm)

    mb = metrics(bench, ppy=ppy)
    ms = metrics(strat, ppy=ppy)

    windows = {
        "2000 Dot-com": ("2000-03-01", "2002-12-31"),
        "2008 GFC": ("2007-10-01", "2009-06-30"),
        "2021 COVID-tech": ("2021-01-01", "2022-12-31"),
    }
    tops = {name: _trough_during(bench, strat, s, e) for name, (s, e) in windows.items()}

    # ---- Report ----------------------------------------------------------
    print("=" * 72)
    print(f"BUBBLE-RISK BACKTEST   freq={period_label}   source={meta.get('source')}")
    print(f"  rebalances={len(dates)}   {dates[0].date()} -> {dates[-1].date()}")
    print(f"  per-period contribution = ${base:,.2f} "
          f"(annual ~ ${base*ppy:,.0f})")
    print(f"  params: low_mult={prm['low_mult']}  high_mult={prm['high_mult']}  "
          f"de-risk>= {prm['derisk_threshold']} ({prm['derisk_cash']*100:.0f}% cash)"
          f"  cash_yield={prm['cash_yield']:.1f}%")
    print("=" * 72)
    print(f"{'Metric':<18}{'Benchmark':>16}{'Bubble-DCA':>16}{'Δ':>12}")
    print("-" * 72)
    rows = [
        ("Cumulative Ret", mb['cum_return'], ms['cum_return'], "%"),
        ("CAGR", mb['cagr'], ms['cagr'], "%"),
        ("Max Drawdown", mb['max_drawdown'], ms['max_drawdown'], "%"),
        ("Sharpe", mb['sharpe'], ms['sharpe'], "x"),
        ("Calmar", mb['calmar'], ms['calmar'], "x"),
        ("Ending Value", mb['end_value'], ms['end_value'], "$"),
    ]
    for name, b, s, kind in rows:
        if kind == "%":
            bs, ss = f"{b*100:>14.1f}%", f"{s*100:>14.1f}%"
            delta = f"{(s-b)*100:>11.1f}pp"
        elif kind == "$":
            bs, ss = f"${b:>13,.0f}", f"${s:>13,.0f}"
            delta = f"{s-b:>12,.0f}"
        else:
            bs, ss = f"{b:>15.2f}", f"{s:>15.2f}"
            delta = f"{s-b:>12.2f}"
        print(f"{name:<18}{bs:>16}{ss:>16}{delta:>12}")
    print("-" * 72)
    print("\nDrawdown comparison during classic tops (strategy vs benchmark):")
    for name, t in tops.items():
        if t:
            print(f"  {name:<18} bench MDD {t['bench_mdd']*100:6.1f}%  |  "
                  f"strategy MDD {t['strat_mdd']*100:6.1f}%  |  "
                  f"avoided {t['avoided_pp']:5.1f} pp")
    print("=" * 72)

    return {"benchmark": mb, "strategy": ms, "tops": tops,
            "bench_equity": bench, "strat_equity": strat,
            "scores": scores.reindex(dates),
            "freq": period_label}


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
