"""Benchmark & boundary checks for the US Equity Bubble Risk Score.

Run AFTER a live (or cached) scoring pass:
    python test_benchmarks.py

It prints the key historical anchor readings and the boundary statistics.

Exit codes:
    0  -> hard boundary contract (Score in [1, 99]) satisfied. Soft targets
          (history anchors, 3y range, today band) are reported as PASS/WARN
          so you can tune Z_GAIN / BUBBLE_CONFIRM_BOOST after a live run.
    1  -> HARD failure: a score fell outside [1, 99].
    2  -> no scores computed (run a live pass first / set FRED_API_KEY).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import pipeline as pipe


def nearest(series: pd.Series, date: str) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    pos = s.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]
    return float(s.iloc[pos])


def main() -> int:
    scores, meta = pipe.get_monthly_scores(refresh=False)
    scores = scores.dropna()
    if scores.empty:
        print("NO SCORES — run a live scoring pass first (set FRED_API_KEY).")
        return 2

    print("=" * 64)
    print("BUBBLE-RISK BENCHMARK CHECKS   source =", meta.get("source"))
    print("=" * 64)

    # 1. HARD boundary contract: Score in [1.0, 99.0], no negatives, none > 100
    lo, hi = float(scores.min()), float(scores.max())
    neg = float((scores < 0).sum())
    over = float((scores > 100).sum())
    print(f"[BOUND] min={lo:.2f}  max={hi:.2f}  "
          f"negatives={neg:.0f}  >100={over:.0f}   (contract [1.0, 99.0])")
    bound_ok = (lo >= 1.0) and (hi <= 99.0) and (neg == 0) and (over == 0)

    # 2. Historical TOP anchors (target >= 88)
    tops = {
        "2000-03 (dot-com top)": "2000-03-31",
        "2021-11 (growth bubble)": "2021-11-30",
    }
    print("\n-- Tops (target >= 88) --")
    top_ok = True
    for label, d in tops.items():
        v = nearest(scores, d)
        ok = v >= 88.0
        top_ok = top_ok and ok
        print(f"  {label:<26} {v:6.1f}  {'PASS' if ok else 'WARN (<88)'}")

    # 3. Crisis BOTTOM anchors (target <= 20)
    bottoms = {
        "2008-11 (GFC)": "2008-11-30",
        "2020-03 (COVID)": "2020-03-31",
    }
    print("\n-- Bottoms (target <= 20) --")
    bot_ok = True
    for label, d in bottoms.items():
        v = nearest(scores, d)
        ok = v <= 20.0
        bot_ok = bot_ok and ok
        print(f"  {label:<26} {v:6.1f}  {'PASS' if ok else 'WARN (>20)'}")

    # 4. Recent 3y range + today band
    recent = scores[scores.index >= pd.Timestamp("2023-07-31")]
    rmin, rmax = float(recent.min()), float(recent.max())
    today = float(scores.iloc[-1])
    print("\n-- Last 3y (2023-07-31 -> today) --")
    print(f"  range: {rmin:.1f} ~ {rmax:.1f}   (target 38 ~ 82)")
    rng_ok = (rmin >= 35.0) and (rmax <= 85.0)
    print(f"  today: {today:.1f}   (target 73 ~ 75)")
    today_ok = 73.0 <= today <= 75.0

    print("\n" + "=" * 64)
    print("SUMMARY")
    print(f"  boundary [1,99] no-neg no->100 : {'PASS' if bound_ok else 'FAIL'}")
    print(f"  tops >= 88                    : {'PASS' if top_ok else 'WARN'}")
    print(f"  bottoms <= 20                 : {'PASS' if bot_ok else 'WARN'}")
    print(f"  3y range 38~82                : {'PASS' if rng_ok else 'WARN'}")
    print(f"  today 73~75                  : {'PASS' if today_ok else 'WARN'}")
    print("=" * 64)

    return 0 if bound_ok else 1


if __name__ == "__main__":
    sys.exit(main())
