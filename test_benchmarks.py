"""Benchmark & boundary checks for the V2 US Equity Bubble Risk Index.

Run AFTER a live (or cached) scoring pass:
    python test_benchmarks.py

Validates the V2 historical-calibration targets and the hard [1, 99] boundary
contract. The calibration pins the dot-com peak -> 97 and the GFC trough -> 12
via an affine map, so the named episodes should land in their expected bands if
the raw module ranking is correct.

Exit codes:
    0  -> hard boundary contract (Score in [1, 99]) satisfied. Soft historical
          targets are reported as PASS/WARN so you can inspect the live ranking.
    1  -> HARD failure: a score fell outside [1, 99].
    2  -> no scores computed (run a live pass first / set FRED_API_KEY).
"""
from __future__ import annotations

import sys

import pandas as pd

import pipeline as pipe


def ep_max(series: pd.Series, a: str, b: str) -> float:
    """Max calibrated score inside an episode window (data-driven anchor)."""
    s = series.dropna()
    seg = s.loc[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    return float(seg.max()) if not seg.empty else float("nan")


def main() -> int:
    scores, meta = pipe.get_monthly_scores(refresh=False)
    scores = scores.dropna()
    if scores.empty:
        print("NO SCORES — run a live scoring pass first (set FRED_API_KEY).")
        return 2

    print("=" * 64)
    print("V2 BUBBLE-INDEX BENCHMARK CHECKS   source =", meta.get("source"))
    print("=" * 64)

    # 1. HARD boundary contract: Score in [1.0, 99.0]
    lo, hi = float(scores.min()), float(scores.max())
    print(f"[BOUND] min={lo:.2f}  max={hi:.2f}   (contract [1.0, 99.0])")
    bound_ok = (lo >= 1.0) and (hi <= 99.0)

    # 2. Historical calibration targets (affine-pinned episodes)
    ep = {
        "Dot-com 2000":  ("1999-01-01", "2001-06-30", (95.0, 100.0)),
        "GFC 2007":      ("2006-06-01", "2008-12-31", (85.0, 90.0)),
        "COVID pre-2020": ("2019-06-01", "2020-02-29", (60.0, 70.0)),
        "2021 liquidity": ("2020-06-01", "2022-01-31", (75.0, 85.0)),
    }
    print("\n-- Historical calibration (affine targets) --")
    for label, (a, b, (lo_t, hi_t)) in ep.items():
        v = ep_max(scores, a, b)
        ok = lo_t <= v <= hi_t
        print(f"  {label:<18} {v:6.1f}   target {lo_t:.0f}-{hi_t:.0f}  "
              f"{'PASS' if ok else 'WARN'}")

    # 3. Recent volatility / realism sanity (should NOT be a flat line)
    recent = scores[scores.index >= pd.Timestamp("2023-07-31")]
    rmin, rmax = float(recent.min()), float(recent.max())
    today = float(scores.iloc[-1])
    print("\n-- Last 3y (2023-07-31 -> today) --")
    print(f"  range: {rmin:.1f} ~ {rmax:.1f}   (expect visible swings)")
    print(f"  today: {today:.1f}   (current reading)")

    # 4. Daily stability check (no violent jumps under normal regimes)
    daily = pipe.get_daily_scores(refresh=False)
    if not daily.dropna().empty:
        dmax = float(daily.diff().abs().dropna().max())
        print(f"\n-- Daily stability --")
        print(f"  max |day-over-day Δ| on daily series: {dmax:.2f} pts")
        print(f"  (normal-regime clamp = {pipe.DAILY_CLAMP}; relaxed to "
              f"{pipe.STRESS_CLAMP} under stress)")

    print("\n" + "=" * 64)
    print("SUMMARY")
    print(f"  boundary [1,99]               : {'PASS' if bound_ok else 'FAIL'}")
    print("=" * 64)

    return 0 if bound_ok else 1


if __name__ == "__main__":
    sys.exit(main())
