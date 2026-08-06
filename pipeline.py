"""
pipeline.py — Dalio-style US Equity Bubble Risk scoring pipeline.

Fetches feature series from free / open APIs (FRED via direct CSV with a
pandas-datareader fallback, yfinance prices with a bot-evading session header
and a Stooq keyless fallback) and computes a 0-100 Bubble Risk Score (V2):

  RISK ACCUMULATION INDICATOR — not a crash forecast. It answers
  "how much speculative risk has built up?" by blending five modules.

  1. Every indicator is converted to a TRAILING-HISTORICAL PERCENTILE (never a
     raw daily level) so no single indicator can whip the index around.
  2. Percentiles feed 5 modules — A.Valuation (30%), B.Sentiment (20%),
     C.Leverage (20%), D.Structure (15%), E.Macro (15%) — each aggregated from
     sub-indicators with coverage gating (a module with <50% coverage is
     neutralised instead of guessed).
  3. Valuation uses an ACCELERATION CURVE (flat <50th pct, linear to 80th,
     convex 80-95th, ramp >95th) so "expensive" and "true bubble" separate.
  4. The module blend is HISTORICALLY CALIBRATED (affine pin of the dot-com
     peak -> ~97 and the GFC trough -> ~12; linear in between) — data-driven,
     no hard "today" pin.
  5. A STABILITY LAYER (EMA span 20 ≈ "70% current + 30% 20d avg") plus a
     daily-change CLAMP (<=1.5 pts normal, <=8 pts under a stress flag:
     VIX>40 OR 21d SPX drop<-15% OR BAA10Y MoM jump>0.5) kills daily whipsaw
     and preserves the multi-year cycle feel.

------------------------------------------------------------------------------
PERFORMANCE DESIGN (production refactor)
------------------------------------------------------------------------------
* CONCURRENT FETCH: all raw series are pulled in parallel with a
  `ThreadPoolExecutor(max_workers=8)`. Every single network call carries a hard
  per-request timeout (`FETCH_TIMEOUT = 5s`) and the whole batch is bounded by a
  total wall-clock deadline (`FETCH_DEADLINE`). A request that times out or
  errors is set to None and never blocks the other 7 — the system returns in
  ~5-20 seconds with whatever subset is live.
* INCREMENTAL CACHE: the first successful run pulls 1990->today and persists the
  raw monthly series + derived features to `bubble_cache.parquet`. On a later
  refresh we only re-fetch the last ~30 days (INCREMENTAL_DAYS), append and
  de-duplicate into the cached history, then recompute the vectorized rolling
  percentiles. Network volume drops from ~26 years to ~30 days (≈100x).
* VECTORIZED SCORING: the rolling percentile and the weighted composite are both
  pandas/numpy C-level operations, so recomputation over the full history is
  sub-second. The expensive part was never the math — it was the network.

------------------------------------------------------------------------------
ZERO-CRASH NORMALIZATION
------------------------------------------------------------------------------
* Every feature's fetch + transform is isolated; one failure can never raise out
  of the pipeline. A failed feature is recorded as None and simply skipped — it is
  NEVER filled with 0 (that would inject a false "lowest-risk" reading and can
  collapse the composite). Missing stays NaN and is excluded by the availability
  mask.
* Dynamic weight renormalization: only VALID (non-null) features enter the
  composite, and their weights are re-normalized to sum to 1.0 over the survivors,
  so the score always lands on 0-100 even if only a subset is available.
* COVERAGE GATE (MIN_VALID_WEIGHT): when a data gap / timeout drops several
  factors at once, the renormalized blend would otherwise be dominated by a few
  survivors and could swing to 0 or 100 (the "curve plunges to 0" bug). Dates
  below the gate are emitted as NaN — the curve shows a short gap, never a spike.
* Synthetic fallback is used ONLY when every feature fails (W_valid == 0).

------------------------------------------------------------------------------
FEATURE MAP  (weight in composite — dual-speed architecture)
------------------------------------------------------------------------------
SLOW MACRO ANCHORS (67%)  — lock the long-cycle extremes
F1  Valuation      (0.22)  CAPE (Shiller PE) + Buffett Indicator (Wilshire/GDP)  [High = Risk]
F4  Leverage       (0.22)  Credit Spread (BAA10Y, INVERTED: low spread = High Risk)
F6  Business Sent. (0.13)  FRED EMVMACROBUS (INVERTED: low index = High Risk)     [AAII fallback]
F8  Tech Froth     (0.20)  QQQ / SPY ratio, 3-year (~156-week) rolling percentile [High = Risk]

FAST SENTIMENT / MOMENTUM (33%)  — capture the market's current "temperature"
F2  Momentum       (0.13)  S&P 500 6m ann. return, 20-day SMA pre-smoothed        [High = Risk]
F3  Market Vol     (0.05)  VIX, 20-day SMA pre-smoothed, INVERTED                 [Low = Risk]
F5  Liquidity      (0.05)  Fed balance-sheet YoY (WALCL)  [+ M2 YoY secondary]   [High = Risk]

BUBBLE-CONFIRMATION INTERACTION: when F1 + F8 + F2 are all in their top-30%
historical percentiles, a small extra Z boost is applied. This pushes true
bubble regimes (2000, 2021) above 90 while keeping 2007/2018 in the 60-75
range as observed in the reference chart.

F7 (Policy / real rate) is merged into F5 and dropped (weight 0) to reduce
micro jitter.  Weights sum to 1.00.

Smoothing: (1) 20-day SMA pre-smoothing on VIX / momentum before their
percentiles; (2) 60-day EMA on the composite.  Non-linear S-stretch (toggle
TAIL_BOOST_ON) escalates |Z|>1 readings for forward-looking tail warnings.
A soft bubble-confirmation interaction adds Z boost when F1/F8/F2 are jointly
in their top 30% historical percentile.
"""

from __future__ import annotations

import os
import json
import math
import threading
import warnings
from concurrent.futures import (ThreadPoolExecutor, as_completed,
                                TimeoutError as FuturesTimeoutError)
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from io import StringIO

try:
    from dotenv import load_dotenv
    load_dotenv()   # pull FRED_API_KEY (and friends) from a local .env file
except Exception:
    pass

# Standard-normal CDF. scipy is the canonical, exact implementation (spec
# requires scipy.stats.norm.cdf). If scipy is somehow unavailable we fall back
# to the erf-based vectorized CDF so the pipeline still runs.
try:
    from scipy.stats import norm as _scipy_norm
    def _standard_normal_cdf(z):
        return _scipy_norm.cdf(np.asarray(z, dtype=float))
except Exception:  # pragma: no cover - scipy is expected on Render / local
    def _standard_normal_cdf(z):
        return _norm_cdf_arr(np.asarray(z, dtype=float))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # optional; api endpoint needs it
CACHE_PATH = os.getenv("BUBBLE_CACHE", "bubble_cache.parquet")
META_PATH = os.getenv("BUBBLE_META", "bubble_cache_meta.json")
HF_DAILY_PATH = os.getenv("HF_DAILY_CACHE", "hf_daily.parquet")

WINDOW_MONTHS = 240       # 20 years for the trailing robust-Z window
# (F5_tech formerly used a 36m recency window — removed: it made the index
# whipsaw; every feature now measures against the same 20y reference.)

# Concurrency / timeout controls (the heart of the perf refactor)
# 8s per request: FRED/Stooq from a cold Render container (fresh DNS+TLS,
# shared egress IP, possible rate-limit backoff) can take 4-6s each — 5s was
# tight enough to contribute to the all-failed -> synthetic fallback.
FETCH_TIMEOUT = 8         # hard per-request timeout (seconds)
# Total wall-clock deadline for the whole batch. 19 series / 8 workers = 3
# waves; worst case 3 x 8s = 24s, bounded here at 20s (stragglers are
# cancelled and treated as missing, never raise). This bound only applies to
# the refresh path — the default page load is cache-first and instant.
FETCH_DEADLINE = 20       # total wall-clock deadline for the whole batch
INCREMENTAL_DAYS = 30     # on refresh: only re-fetch the last ~30 days
CACHE_MAX_AGE_HOURS = 6.0 # stale after this -> auto incremental refresh on load

# ===========================================================================
# V2 BUBBLE INDEX — 5-MODULE ARCHITECTURE
# ===========================================================================
# Design philosophy (professional macro-fund / quant framing):
#   * Every underlying indicator is first converted to a trailing-historical
#     PERCENTILE (0-100). The score NEVER uses a single day's raw level — this
#     is what kills the "over-sensitive to one indicator / daily whipsaw"
#     problem of naive composite indices.
#   * The 8 granular percentile factors are aggregated into 5 risk MODULES.
#   * The 5 modules are weighted and blended into a raw composite.
#   * The raw composite is HISTORICALLY CALIBRATED (affine) so the dot-com
#     peak maps to ~97 and the GFC trough to ~12 — the index scale is pinned to
#     real bubbles, not to arbitrary constants.
#   * A STABILITY layer (steady EMA + a hard daily-change clamp) guarantees the
#     published daily score cannot jump more than ~1.5 pts unless a genuine
#     stress regime (VIX>40 / credit blow-out / >15% monthly drop) hits.
# The index deliberately measures RISK ACCUMULATION (price vs fundamentals,
# euphoria, leverage), NOT a crash prediction.

# --- Granular factor weights (used for the per-factor detail / coverage) ----
WEIGHTS = {
    "F1_valuation": 0.20,   # CAPE / Buffett (valuation anchor)
    "F1b_cape_z":   0.00,   # CAPE vs its own 10y average (valuation sub)
    "F2_leverage":  0.20,   # FINRA margin debt / market cap
    "F3_credit":    0.15,   # BAA10Y spread (inverted)
    "F3b_realrate": 0.00,   # real fed funds (FEDFUNDS - CPI, inverted)
    "F3c_yield":    0.00,   # 10Y-3M treasury spread (inverted)
    "F4_business":  0.15,   # EMVMACROBUS / AAII bullish (complacency)
    "F5_tech":      0.10,   # QQQ/SPY 3y relative (structure)
    "F6_momentum":  0.10,   # S&P 500 6m ann. (10d-SMA pre-smoothed)
    "F7_volatility": 0.05,  # VIX (inverted, 10d-SMA pre-smoothed)
    "F8_liquidity": 0.05,   # Fed WALCL YoY (+ M2 YoY secondary)
}

# --- The 5 risk modules and how each granular factor maps onto them ---------
# Each module is the (weighted) mean of its sub-indicator percentiles; the
# valuation module additionally runs its inputs through the acceleration curve.
MODULE_WEIGHTS = {
    "valuation": 0.30,   # A. Valuation  (CAPE, Buffett, CAPE-vs-10y)
    "sentiment": 0.20,   # B. Sentiment  (VIX complacency, EMV/AAII bullish)
    "leverage":  0.20,   # C. Leverage   (FINRA margin debt / market cap)
    "structure": 0.15,   # D. Structure  (Nasdaq/S&P divergence, mega-cap)
    "macro":     0.15,   # E. Macro      (yield curve, credit, real rate)
}
MODULE_SUBINDICATORS = {
    "valuation": ["F1_valuation", "F1b_cape_z"],
    "sentiment": ["F7_volatility", "F4_business"],
    "leverage":  ["F2_leverage"],
    "structure": ["F5_tech"],
    "macro":     ["F3_credit", "F3b_realrate", "F3c_yield"],
}

# --- Valuation acceleration curve (principle A) ----------------------------
# Maps a valuation PERCENTILE p (0-100) to a risk score:
#   p < 50        -> 0      (below-median valuation = NO bubble risk)
#   50 <= p <= 80 -> linear (0 -> 50)      moderate, proportional
#   80 <  p <= 95 -> accel  (50 -> 90)     convex — froth accelerates
#   p  > 95       -> max    (90 -> 100)    extreme valuation = max risk
VAL_FLAT, VAL_LIN_HI, VAL_ACC_HI, VAL_MAX = 50.0, 80.0, 95.0, 100.0
VAL_LIN_OUT, VAL_ACC_OUT = 50.0, 90.0
VAL_ACC_POWER = 1.8          # convexity of the 80-95 segment

# --- Historical affine calibration -----------------------------------------
# We pin the local EXTREME of two unambiguous episodes (data-driven, not a
# hard-coded date): the MAX raw composite inside the dot-com window -> 97, and
# the MIN raw composite inside the GFC window -> 12. Everything else is linearly
# interpolated, so the full macro wave is preserved and today falls wherever the
# data puts it. This simultaneously lands 2007/2021/COVID-pre in their expected
# zones IF the raw ranking is correct (verified by test_benchmarks.py).
HIST_PEAK_WINDOW = ("1999-01-01", "2001-06-30")    # dot-com local max -> 97
HIST_TROUGH_WINDOW = ("2007-10-01", "2009-12-31")  # GFC local min   -> 12
HIST_PEAK_TARGET = 97.0
HIST_TROUGH_TARGET = 12.0

# --- V3 fixed-gain calibration ---------------------------------------------
# The old data-anchored affine (historical_calibrate) re-derived its scale
# from the dot-com window MAX on every run. When feature availability
# depressed that anchor (e.g. CAPE missing locally), the whole scale
# compressed and dozens of months clipped at 97-99 — the "always at max"
# pathology. V3 replaces it with a DETERMINISTIC linear map on the module
# blend Z-score: score = 50 + CALIB_Z_GAIN * blend_z, clipped to [1, 99].
# It depends on NO data anchors, so it is immune to feature availability.
# Gain chosen so the expected anchors land sensibly:
#   blend +1.66σ -> ~97 (dot-com 2000, full CAPE data)
#   blend +0.9σ  -> ~75 (2021 liquidity mania)
#   blend +0.5σ  -> ~64 (COVID-pre 2020)
#   blend  0.0   ->  50 (median risk)
#   blend -1.5σ  ->  ~8 (GFC trough 2008-09)
CALIB_Z_GAIN = 28.0

# --- Stability layer (principle 1) -----------------------------------------
# --- K-line style two-timescale stability filter ---------------------------
# A slow EMA carries the mid/long-term macro trend; a damped fast component
# adds a SMALL, bounded short-term oscillation on top — like a stock K-line:
# a clear medium-term trend with minor daily wiggle. Never a violent sawtooth,
# never an over-smoothed flat line.
TREND_SPAN = 75              # slow EMA on the daily series (≈ one quarter)
OSC_SPAN = 8                 # fast EMA -> short-term component
OSC_GAIN = 0.0               # DISABLED: the regime overlay (get_daily_scores)
                             # already provides short-term variation. The OSC
                             # term on top created a day-to-day sawtooth
                             # instead of a recognisable regime trend. Set
                             # back to 0.45 to restore the old K-line wiggle
                             # (small visible OSC around the slow trend).
OSC_MAX = 3.5                # hard cap on the short-term oscillation (points)
# 0.6 pts/day: a 3-pt monthly step is absorbed over ~a week (calm stair-step
# instead of a jump); was 1.2 which let the line drift ~6 pts/week.
DAILY_CLAMP = 0.6            # max |Δ| per day in normal regimes
STRESS_CLAMP = 6.0           # relaxed clamp under genuine stress
STRESS_VIX = 40.0            # VIX > 40 -> stress
STRESS_CREDIT_JUMP = 0.5     # BAA10Y MoM widen > 50 bps -> stress
STRESS_DROP = -0.15          # trailing-21d S&P 500 drop < -15% -> stress

# --- Risk-level bands (display + gauge) ------------------------------------
RISK_BANDS = [
    (0, 40, "#10b981", "Cheap / Fear"),
    (40, 60, "#3b82f6", "Normal"),
    (60, 75, "#f59e0b", "Expensive"),
    (75, 90, "#ef4444", "Bubble Risk"),
    (90, 100, "#991b1b", "Extreme Bubble"),
]

# Minimum valid module-coverage required to emit a score for a date; otherwise
# the date is NaN (gap, not spike). With all 5 modules live -> 1.0.
MIN_VALID_WEIGHT = 0.70

# Legacy toggle kept for API compatibility (valuation acceleration curve was
# retired in V3 — see compute_modules; this flag is now a no-op)
TAIL_BOOST_ON = True

# Cache format marker written into bubble_cache_meta.json. "z" = feat_* cols
# are robust Z-scores (V3). Anything else/missing = legacy V2 percentiles ->
# the cache is ignored and rebuilt live (prevents silently misreading old
# percentile caches as Z after a redeploy).
FEAT_FORMAT = "z"

FEATURE_LABELS = {
    "F1_valuation": "Valuation (CAPE / Buffett)",
    "F1b_cape_z": "Valuation (CAPE vs 10y avg)",
    "F2_leverage": "Leverage (FINRA Margin Debt)",
    "F3_credit": "Credit Spread (BAA10Y inv.)",
    "F3b_realrate": "Real Rate (FedFunds-CPI inv.)",
    "F3c_yield": "Yield Curve (10Y-3M inv.)",
    "F4_business": "Sentiment (EMV/AAII bullish)",
    "F5_tech": "Structure (QQQ/SPY, 3y)",
    "F6_momentum": "Momentum (6m ann.)",
    "F7_volatility": "Volatility (VIX inv.)",
    "F8_liquidity": "Liquidity (Fed BS / M2)",
}

HISTORY_START = "1990-01-01"   # long history so the 20y window is "full" by 2010
LIVE_START = "2000-01-01"      # backtest / reporting start

# Raw series to fetch. key -> (kind, source_id)
#   kind "fred"  -> FRED series id (fetched via direct CSV, keyless-capable)
#   kind "price" -> ticker fetched via yfinance/Stooq (monthly close)
RAW_SPECS = {
    "cape":     ("fred", "CAPE"),
    "wilshire": ("fred", "WILL5000INDFC"),
    "gdp":      ("fred", "GDP"),
    "vixcls":   ("fred", "VIXCLS"),
    "baa10y":   ("fred", "BAA10Y"),
    "m2":       ("fred", "M2SL"),
    "walcl":    ("fred", "WALCL"),
    "dgs10":    ("fred", "DGS10"),     # 10Y Treasury yield (yield-curve module)
    "dgs3mo":   ("fred", "DGS3MO"),    # 3M Treasury yield (yield-curve module)
    "emv":      ("fred", "EMVMACROBUS"),
    "mgdte":    ("fred", "MGDTE"),      # FINRA margin debt (broker-dealers), monthly
    "cpi":      ("fred", "CPIAUCSL"),
    "ffr":      ("fred", "FEDFUNDS"),
    "sp500div": ("fred", "SP500DIV"),   # tertiary F1 valuation fallback
    "spx":      ("price", "^GSPC"),
    "spy":      ("price", "SPY"),
    "qqq":      ("price", "QQQ"),
    "ixic":     ("price", "^IXIC"),
    "vix":      ("price", "^VIX"),
}
# Ticker -> raw column, for the dashboard price chart
RAW_TICKER_MAP = {"^GSPC": "spx", "^IXIC": "ixic", "SPY": "spy", "QQQ": "qqq"}


# ---------------------------------------------------------------------------
# Low-level, timeout-guarded helpers
# ---------------------------------------------------------------------------
def _run_with_timeout(fn: Callable[[], object], timeout: float, default=None):
    """Run ``fn`` in a daemon thread and return its result, or ``default`` if it
    does not finish within ``timeout`` seconds.

    This lets us put a hard wall-clock bound on third-party fetchers (e.g.
    pandas_datareader) that do not honour our own ``requests`` timeout.
    """
    box: dict = {}

    def _t():
        try:
            box["v"] = fn()
        except Exception:
            box["v"] = default

    th = threading.Thread(target=_t, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return default
    return box.get("v", default)


def _http_get(url: str, timeout: int = FETCH_TIMEOUT) -> Optional[str]:
    """Keyless HTTP GET with a browser UA (Stooq / FRED block empty UAs)."""
    try:
        hdr = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0 Safari/537.36")}
        r = requests.get(url, headers=hdr, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        print(f"[http] {url[:78]}... failed: {exc}")
        return None


def _fred_csv(series_id: str, start: str = HISTORY_START,
              timeout: int = FETCH_TIMEOUT,
              monthly: bool = True) -> Optional[pd.Series]:
    """Fetch a FRED series via the direct CSV endpoint.

    Uses the authenticated `api.stlouisfed.org` endpoint when FRED_API_KEY is
    set (reliable + server-side date filtering) and the keyless
    `fredgraph.csv` endpoint otherwise. Both honour `timeout` natively.
    Falls back to pandas_datareader (bounded by _run_with_timeout) only if the
    CSV endpoints fail. ``monthly=False`` keeps the native (daily) frequency —
    used by the high-frequency VIX/SPX pre-smoothing layer.
    """
    if FRED_API_KEY:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={FRED_API_KEY}"
               f"&file_type=csv&observation_start={start}")
    else:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={series_id}&cosd={start}")
    txt = _http_get(url, timeout=timeout)
    if txt:
        s = _parse_fred_csv(txt, series_id, monthly=monthly)
        if s is not None and not s.empty:
            return s

    # ---- pandas_datareader fallback (bounded so it can't hang the batch) ---
    s = _run_with_timeout(
        lambda: _fred_pdr(series_id, start), timeout=FETCH_TIMEOUT)
    if s is not None and not monthly:
        # _fred_pdr resamples to ME; re-expand is lossy, so only monthly mode
        # uses the pdr fallback meaningfully. Daily mode: return what we have.
        return None
    return s


def _parse_fred_csv(txt: str, series_id: str,
                    monthly: bool = True) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(StringIO(txt))
        if df.empty:
            return None
        cols = [str(c) for c in df.columns]
        date_col = next((c for c in cols if "date" in c.lower()), cols[0])
        val_col = next((c for c in cols
                        if ("value" in c.lower()) or (series_id.lower() in c.lower())),
                       cols[-1])
        s = pd.to_numeric(df[val_col], errors="coerce")
        s.index = pd.to_datetime(df[date_col], errors="coerce")
        s = s.dropna().sort_index()
        if monthly:
            s = s.resample("ME").last()      # monthly month-end
        return s if not s.empty else None
    except Exception as exc:
        print(f"[fred] {series_id} parse failed: {exc}")
        return None


def _fred_pdr(series_id: str, start: str) -> Optional[pd.Series]:
    """pandas_datareader fallback for a FRED series (used only if direct CSV fails)."""
    try:
        import pandas_datareader.data as web  # pandas_datareader is in requirements.txt
        df = web.get_data_fred(series_id, start=start)
        if df is None or df.empty:
            return None
        col = df.columns[0]
        s = pd.to_numeric(df[col], errors="coerce").dropna().sort_index()
        s = s.resample("ME").last()
        return s if not s.empty else None
    except Exception as exc:
        print(f"[fred-pdr] {series_id} failed: {exc}")
        return None


_STOOQ_MAP = {"^GSPC": "SPX.US", "SPY": "SPY.US", "QQQ": "QQQ.US",
              "^IXIC": "IXIC.US", "^VIX": "VIX.US"}

# FRED price fallbacks — the anti-fragile price layer (user request):
# Yahoo frequently 429s datacenter IPs and Stooq sits behind a JS proof-of-work
# wall (confirmed 2026-08), so FRED is the ONLY price source that is both
# keyless-friendly and API-stable.
#   ^IXIC -> NASDAQCOM (daily, 1971+; full dot-com history — FRED-PRIMARY)
#   ^VIX  -> VIXCLS    (daily, 1990+; FRED-PRIMARY, same series as vixcls spec)
#   ^GSPC -> SP500     (daily, ~2013+; recent-tail fallback only, yf/Stooq
#                        carry the deep history)
#   SPY   -> SP500     (LAST RESORT for the backtest: price index, no divs;
#                        benchmark & strategy use the same series -> fair)
FRED_PRICE_MAP = {"^GSPC": "SP500", "^IXIC": "NASDAQCOM", "^VIX": "VIXCLS",
                  "SPY": "SP500"}
FRED_PRICE_PRIMARY = {"^IXIC", "^VIX"}


def _yf_session() -> requests.Session:
    """A browser-like session so Yahoo does not 403/429 us as a bot."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36"),
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,*/*;q=0.8"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return s


def _stooq_daily(symbol: str, start: str = HISTORY_START) -> Optional[pd.Series]:
    """Keyless, datacenter-friendly DAILY price source (stooq.com CSV)."""
    txt = _http_get(f"https://stooq.com/q/d/l/?s={symbol}&i=d",
                    timeout=FETCH_TIMEOUT)
    if not txt:
        return None
    try:
        df = pd.read_csv(StringIO(txt))
        if df.empty or "Close" not in df.columns or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        s = df["Close"].dropna()
        s = s[s.index >= pd.Timestamp(start)]
        return s if not s.empty else None
    except Exception as exc:
        print(f"[stooq] {symbol} failed: {exc}")
        return None


def _fetch_price(ticker: str, start: str = HISTORY_START,
                 timeout: int = FETCH_TIMEOUT) -> Optional[pd.Series]:
    """Monthly-close price with a three-layer anti-fragile chain.

    Order: [FRED (for FRED_PRICE_PRIMARY tickers)] -> yfinance -> Stooq ->
    FRED (fallback). Yahoo frequently 429s cloud IPs; Stooq sits behind a JS
    proof-of-work wall; FRED (SP500 / NASDAQCOM / VIXCLS) is the stable,
    keyless-capable backbone that keeps prices REAL when both fail.
    """
    fred_id = FRED_PRICE_MAP.get(ticker)

    # Layer 0: FRED-primary tickers (Nasdaq Composite, VIX) — most reliable.
    if fred_id and ticker in FRED_PRICE_PRIMARY:
        s = _fred_csv(fred_id, start=start, timeout=timeout)
        if s is not None and not s.empty:
            return s

    # Layer 1: yfinance (deep history, but datacenter-hostile).
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, auto_adjust=True, actions=False,
                         progress=False, threads=False, timeout=timeout,
                         session=_yf_session())
        if df is not None and not df.empty:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna()
            if not s.empty:
                return s.resample("ME").last()
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[yfinance] {ticker} failed: {exc}")

    # Layer 2: Stooq (often behind a JS anti-bot wall — usually fails headless).
    st = _STOOQ_MAP.get(ticker)
    if st:
        s = _stooq_daily(st, start=start)
        if s is not None and not s.empty:
            return s

    # Layer 3: FRED fallback (SP500 for SPY/^GSPC is price-only, no dividends —
    # acceptable for charts and for the benchmark-vs-strategy comparison,
    # which uses the SAME series on both sides).
    if fred_id:
        return _fred_csv(fred_id, start=start, timeout=timeout)
    return None


def _fetch_daily_prices(ticker: str, start: str = HISTORY_START,
                        timeout: int = FETCH_TIMEOUT) -> Optional[pd.Series]:
    """DAILY-close price for the high-frequency pre-smoothing layer (VIX, SPX).

    Unlike _fetch_price (which resamples to month-end), this returns the raw
    daily close so we can compute a 20-trading-day SMA BEFORE the percentile.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, auto_adjust=True, actions=False,
                         progress=False, threads=False, timeout=timeout,
                         session=_yf_session())
        if df is not None and not df.empty:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna()
            if not s.empty:
                return s
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[yfinance-daily] {ticker} failed: {exc}")
    st = _STOOQ_MAP.get(ticker)
    if st:
        s = _stooq_daily(st, start=start)
        if s is not None and not s.empty:
            return s
    # FRED daily fallback (^VIX->VIXCLS, ^GSPC->SP500): the anti-fragile layer
    # for the 10d-SMA pre-smoothing inputs when Yahoo 429s and Stooq is walled.
    fred_id = FRED_PRICE_MAP.get(ticker)
    if fred_id:
        return _fred_csv(fred_id, start=start, timeout=timeout, monthly=False)
    return None


def _load_hf_cache() -> Optional[pd.DataFrame]:
    if not os.path.exists(HF_DAILY_PATH):
        return None
    try:
        return pd.read_parquet(HF_DAILY_PATH)
    except Exception:
        return None


def _save_hf_cache(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(HF_DAILY_PATH)
    except Exception:
        pass


def _get_hf_daily() -> dict:
    """Return a dict of daily Series (keys: 'vix', 'spx', 'ndx') used for the
    first smoothing layer AND for the daily price lines on the dashboard
    chart, with an incremental parquet cache so refreshes only pull the last
    ~30 days. Non-fatal: missing keys simply fall back to the monthly path
    inside compute_features_from_raw.

    Network is bounded by FETCH_TIMEOUT + the global deadline; any failure
    returns whatever subset is available (possibly empty).
    """
    cached = _load_hf_cache()
    df = cached if cached is not None else pd.DataFrame()
    out: dict = {}
    for tag, ticker in (("vix", "^VIX"), ("spx", "^GSPC"), ("ndx", "^IXIC")):
        start = HISTORY_START
        if tag in df.columns and df[tag].notna().any():
            last = df[tag].dropna().index.max()
            start = (last - pd.Timedelta(days=INCREMENTAL_DAYS)).strftime("%Y-%m-%d")
        s = _fetch_daily_prices(ticker, start=start)
        if s is not None and not s.empty:
            s = s[s.index >= pd.Timestamp(HISTORY_START)]
            if tag in df.columns:
                df[tag] = s.combine_first(df[tag]).sort_index()
            else:
                df[tag] = s
            df[tag] = df[tag].ffill(limit=6)
            out[tag] = df[tag].dropna()
    if not df.empty:
        _save_hf_cache(df)
    return out


def get_daily_price(ticker: str) -> Optional[pd.Series]:
    """DAILY price series for the dashboard chart (S&P 500 / Nasdaq), served
    from the incremental daily cache (hf_daily.parquet). Falls back to the
    monthly raw cache when the daily series is unavailable. Never raises.
    """
    key = {"^GSPC": "spx", "SPX": "spx", "^IXIC": "ndx", "NDX": "ndx"}.get(
        ticker, ticker)
    try:
        hf = _load_hf_cache()
        if hf is not None and key in hf.columns:
            s = hf[key].dropna()
            if not s.empty:
                return s
    except Exception:
        pass
    # fall back to the monthly raw cache column
    raw_key = {"spx": "spx", "ndx": "ixic"}.get(key)
    if raw_key:
        raw = _load_raw_cache()
        if raw is not None and raw_key in raw.columns:
            s = raw[raw_key].dropna()
            if not s.empty:
                return s
    return None


def _fetch_one(spec: Tuple[str, str], start: str,
               timeout: int = FETCH_TIMEOUT) -> Optional[pd.Series]:
    """Single-series fetch wrapper. Never raises — returns None on any failure."""
    kind, sid = spec
    try:
        if kind == "fred":
            return _fred_csv(sid, start=start, timeout=timeout)
        return _fetch_price(sid, start=start, timeout=timeout)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[fetch] {sid} error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Keyless fallback source for F6 (only used when FRED EMVMACROBUS is missing)
# ---------------------------------------------------------------------------
def _aaii_sentiment(start: str = "1987-01-01") -> Optional[pd.Series]:
    """AAII Investor Sentiment — % bullish. High bullish = complacency = risk.

    REAL, keyless, weekly (resampled to monthly). Used only as the F6 fallback
    when EMVMACROBUS is unavailable.
    """
    txt = _http_get("https://www.aaii.com/sentimentsurvey/sentiment_history.csv")
    if not txt:
        return None
    try:
        df = pd.read_csv(StringIO(txt))
        if "Bullish" not in df.columns or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        s = df["Bullish"].dropna().resample("ME").mean()
        s = s[s.index >= pd.Timestamp(start)]
        return s if not s.empty else None
    except Exception as exc:
        print(f"[aaii] failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Rolling percentile (no look-ahead)
# ---------------------------------------------------------------------------
def rolling_pct(series, window: int = WINDOW_MONTHS,
                min_periods: Optional[int] = None) -> pd.Series:
    """
    Percentile rank (0-100) of each value within its TRAILING `window`
    observations. Vectorized via pandas rolling rank (C-level). Purely
    backwards-looking (no look-ahead bias).
    """
    s = pd.Series(series, dtype="float64")
    if min_periods is None:
        min_periods = max(24, window // 4)
    rp = s.rolling(window, min_periods=min_periods).rank(pct=True, method="average")
    return rp * 100.0


# ---------------------------------------------------------------------------
# Rolling ROBUST Z-score (the V3 anti-saturation transform)
# ---------------------------------------------------------------------------
def rolling_robust_z(series, window: int = WINDOW_MONTHS,
                     min_periods: int = 60) -> pd.Series:
    """Robust trailing Z-score: (x − median) / (1.4826·MAD) over a TRAILING
    `window`, with a rolling-std fallback when the MAD collapses (flat series).

    WHY THIS REPLACES THE TRAILING PERCENTILE (V2 saturation bug):
    a trailing-window PERCENTILE pins every record-breaking reading at ~100
    for as long as it remains the record — so CAPE/momentum/liquidity all sat
    at pct≈100 for YEARS (2007, 2020, 2021, 2025-26), the module blend maxed
    out, and the calibrated index clipped at 99 "经常处于最大值". The Z keeps
    measuring HOW FAR beyond the window each observation is (2000 vs 2007 vs
    2021 vs today stay differentiated), and the affine calibration maps that
    spacing linearly onto the 0-100 scale — the top of the scale only binds
    for genuinely beyond-dot-com extremes.

    Output is clipped to ±4σ for outlier control; NaN-safe, no look-ahead.
    """
    s = pd.Series(series, dtype="float64")
    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    sd = s.rolling(window, min_periods=min_periods).std()
    scale = 1.4826 * mad
    scale = scale.where(scale > 1e-12, sd)   # MAD=0 (flat run) -> std fallback
    scale = scale.replace(0, np.nan)
    return ((s - med) / scale).clip(-4.0, 4.0)


def z_display(z):
    """Display mapping for a Z feature/module: 100·Φ(z) -> 0-100, 50 = neutral.

    Used ONLY for presentation (feature cards, module cards). The composite
    itself works on the raw Z scale, so the monotone squash cannot compress
    the index's top-end differentiation.
    """
    if pd.isna(z):
        return np.nan
    return 100.0 * _norm_cdf(float(z))


# ---------------------------------------------------------------------------
# Standard-normal helpers (no scipy dependency)
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (scalar)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_cdf_arr(x: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal CDF for a numpy array."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal quantile (inverse CDF) — Acklam's rational approx.

    Accurate to ~1e-9 across (0,1); used to turn a factor percentile into a
    comparable standard-normal Z so every factor sits on the same scale before
    the weighted blend.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    pp = min(max(p, 1e-12), 1 - 1e-12)
    if pp < plow:
        q = math.sqrt(-2.0 * math.log(pp))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if pp > phigh:
        q = math.sqrt(-2.0 * math.log(1 - pp))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = pp - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def _pct_to_z(pct: pd.Series) -> pd.Series:
    """Map a 0-100 PERCENTILE rank to a standard-normal Z (Gaussian quantile).

    pct = 50 -> 0, pct = 84.1 -> +1, pct = 97.7 -> +2. The exact 0/100 extremes
    are guarded inside ``_norm_ppf`` (clamped to 1e-12 / 1-1e-12) so the quantile
    never returns +/-inf. NaN in -> NaN out. No ``fillna(0)`` anywhere: a missing
    percentile stays NaN and is excluded from the blend by the availability mask.
    """
    p = pct.astype(float) / 100.0
    return p.apply(lambda v: _norm_ppf(v) if pd.notna(v) else np.nan)


# ---------------------------------------------------------------------------
# V2 Composite scoring — 5-module, percentile-based, historically calibrated
# ---------------------------------------------------------------------------
def valuation_curve(p: float) -> float:
    """Map a valuation PERCENTILE (0-100) to a risk score via the acceleration
    curve (principle A):

        p < 50        -> 0      (below-median valuation = no bubble risk)
        50 <= p <= 80 -> linear (0 -> 50)
        80 <  p <= 95 -> accel  (50 -> 90, convex)
        p  > 95       -> max    (90 -> 100)

    A single-day print can NEVER move this (input is itself a trailing
    percentile), so valuation risk is inherently smooth.
    """
    if pd.isna(p):
        return np.nan
    p = float(p)
    if p < VAL_FLAT:
        return 0.0
    if p <= VAL_LIN_HI:
        return (p - VAL_FLAT) / (VAL_LIN_HI - VAL_FLAT) * VAL_LIN_OUT
    if p <= VAL_ACC_HI:
        frac = (p - VAL_LIN_HI) / (VAL_ACC_HI - VAL_LIN_HI)
        return VAL_LIN_OUT + (VAL_ACC_OUT - VAL_LIN_OUT) * (frac ** VAL_ACC_POWER)
    # p > 95 -> ramp 90 -> 100
    frac = min((p - VAL_ACC_HI) / (VAL_MAX - VAL_ACC_HI), 1.0)
    return VAL_ACC_OUT + (VAL_MAX - VAL_ACC_OUT) * frac


def compute_modules(feat_z: pd.DataFrame,
                    tail_boost: Optional[bool] = None) -> pd.DataFrame:
    """Aggregate the granular Z features into the 5 V2 risk modules (Z scale).

    Each module is the mean of its available sub-indicator Z-scores. A module
    with NO available sub-indicator is filled with neutral 0 (= median risk),
    so a single missing series can never swing the blend; coverage is recorded
    for the global gate.

    V3 NOTE: the old valuation acceleration curve (applied to saturated
    trailing percentiles) was a major contributor to the index pinning at 99 —
    on the Z scale the blend already differentiates extremes, so the curve is
    gone. `tail_boost` is kept in the signature for API compatibility and is
    now a no-op.
    """
    out = pd.DataFrame(index=feat_z.index)
    coverage = {}
    for mod, cols in MODULE_SUBINDICATORS.items():
        present = [c for c in cols if c in feat_z.columns
                   and feat_z[c].notna().any()]
        coverage[mod] = (len(present) / len(cols)) if cols else 0.0
        if not present:
            out[mod] = 0.0        # neutral Z (= median), was 50.0 on pct scale
            continue
        if mod == "sentiment":
            # SENTIMENT = MIN, not mean (V3 fix). The sub-indicators are BOTH
            # complacency gauges (VIX-inverted, EMV/AAII-inverted): a spike in
            # VIX (fear) while EMV stays calm is still a fear event — averaging
            # let the calm series CANCEL the fear (e.g. 2026-03: F7=-1.41,
            # F4=+0.92 -> mean -0.24, hiding a real VIX spike). Taking the
            # minimum means EITHER fear gauge flashing de-risks the module;
            # both must be complacent for the module to read frothy.
            out[mod] = feat_z[present].min(axis=1)
        else:
            out[mod] = feat_z[present].mean(axis=1)
    out.attrs["coverage"] = coverage
    return out


def historical_calibrate(raw: pd.Series) -> pd.Series:
    """LEGACY (V2) — superseded by the V3 fixed-gain map in compute_composite.

    Affine-calibrate the raw composite to the historical bubble scale.

    Pin the MAX raw reading inside ``HIST_PEAK_WINDOW`` (the dot-com episode)
    to ``HIST_PEAK_TARGET`` (97) and the MIN raw reading inside
    ``HIST_TROUGH_WINDOW`` (the GFC episode) to ``HIST_TROUGH_TARGET`` (12).
    Linear interpolation everywhere else preserves the relative macro wave and
    guarantees the index lands in a realistic [~12, ~97] band with today
    falling wherever the data puts it (no hard-coded "today" pin).

    Kept for reference only: re-deriving the scale from data anchors made the
    whole index fragile to feature availability — a depressed dot-com anchor
    compressed the scale and pinned dozens of months at 97-99.
    """
    if raw is None or raw.dropna().empty:
        return raw
    r = raw.astype(float)
    pk = r.loc[(r.index >= pd.Timestamp(HIST_PEAK_WINDOW[0]))
               & (r.index <= pd.Timestamp(HIST_PEAK_WINDOW[1]))].dropna()
    tr = r.loc[(r.index >= pd.Timestamp(HIST_TROUGH_WINDOW[0]))
               & (r.index <= pd.Timestamp(HIST_TROUGH_WINDOW[1]))].dropna()
    x_hi = pk.max() if not pk.empty else r.max()
    x_lo = tr.min() if not tr.empty else r.min()
    if not np.isfinite(x_hi) or not np.isfinite(x_lo) or x_hi <= x_lo:
        # degenerate: fall back to a neutral 50-centred linear stretch
        x_lo, x_hi = r.min(), r.max()
        if x_hi <= x_lo:
            return pd.Series(50.0, index=r.index)
    score = HIST_TROUGH_TARGET + (r - x_lo) / (x_hi - x_lo) * (
        HIST_PEAK_TARGET - HIST_TROUGH_TARGET)
    return score.clip(1.0, 99.0)


def compute_composite(feat_pct: pd.DataFrame, weights: dict = None,
                      tail_boost: Optional[bool] = None) -> pd.Series:
    """V3 Bubble Risk Score (0-100) from the granular robust-Z features.

    Pipeline (no look-ahead, fully vectorized):
      1. Aggregate the 8 granular Z features into 5 risk MODULES (Z scale).
      2. Weighted blend the modules (MODULE_WEIGHTS); a module below the
         coverage gate is neutralised (Z=0 = median) so it can't distort.
      3. Fixed-gain calibration ON THE Z SCALE -> realistic bubble scale:
         score = 50 + CALIB_Z_GAIN * blend_z, clipped to [1, 99].
         Linear-in-Z preserves top-end spacing, so the index differentiates
         2000 / 2007 / 2021 / today instead of clipping at 99. Being
         data-anchor-free, it cannot be distorted by feature availability
         (the failure mode that saturated the old affine at 99).
      4. Coverage gate: dates with < MIN_VALID_WEIGHT module coverage -> NaN
         (gap, not spike).
    """
    modules = compute_modules(feat_pct, tail_boost=tail_boost)
    cov = modules.attrs.get("coverage", {})
    w = pd.Series(MODULE_WEIGHTS)
    # neutralise modules that are essentially missing
    avail = pd.Series({m: (cov.get(m, 0.0) >= 0.5) for m in w.index})
    w_eff = w * avail
    if w_eff.sum() == 0:
        w_eff = w
    else:
        w_eff = w_eff / w_eff.sum()
    blended = (modules * w_eff).sum(axis=1)

    # coverage gate on the module level
    total_cov = sum(MODULE_WEIGHTS[m] * cov.get(m, 0.0) for m in MODULE_WEIGHTS)
    # NOTE: pandas 3.0 no longer accepts a scalar bool as the cond of
    # Series.where ("Array conditional must be same shape as self") — gate
    # with an explicit branch instead (works on pandas 2.x AND 3.x).
    if total_cov < MIN_VALID_WEIGHT:
        blended = pd.Series(np.nan, index=blended.index)

    # V3 fixed-gain calibration: deterministic linear map on the blend Z.
    # Robust to feature availability (unlike the old data-anchored affine,
    # which compressed the scale and pinned 44 months at 97-99 when the
    # dot-com anchor was depressed by missing features).
    score = (50.0 + CALIB_Z_GAIN * blended).clip(1.0, 99.0)
    return score


def contribution_factor(score: float) -> float:
    """Monthly DEPLOYMENT multiple from the Bubble Risk Score (V3 bands).

    Kept in sync with backtest.DEFAULT_PARAMS for reference/display only
    (the backtest engine in backtest.py is the authoritative implementation).
    V3 semantics: the monthly outflow is constant; this factor describes how
    much of the cash on hand gets DEPLOYED — >1 draws the reserve, <1
    stockpiles it.
    """
    if pd.isna(score):
        return 1.0
    if score < 40:
        return 3.0
    if score < 50:
        return 1.5
    if score < 80:
        return 1.0
    if score < 95:
        return 0.5
    return 0.0


def status_of(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score < 40:
        return "Low / Cooling (2.0x DCA)"
    if score < 60:
        return "Normal (1.5x DCA)"
    if score < 80:
        return "Watch (1.0x DCA)"
    if score < 90:
        return "Elevated (0.5x DCA)"
    return "Bubble Warning (0x DCA)"


# ---------------------------------------------------------------------------
# Concurrent raw fetch + incremental cache
# ---------------------------------------------------------------------------
def _load_raw_cache() -> Optional[pd.DataFrame]:
    """Load the raw monthly series stored in the unified cache (everything that
    is not a `feat_*` column or `score`)."""
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        df = pd.read_parquet(CACHE_PATH)
        raw_cols = [c for c in df.columns
                    if not (c.startswith("feat_") or c == "score")]
        if not raw_cols:
            return None
        return df[raw_cols].copy()
    except Exception:
        return None


def fetch_all_raw(incremental: bool) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch every raw series CONCURRENTLY (ThreadPoolExecutor, 5s per-request
    timeout, hard total deadline). Returns (raw_df, fetch_meta).

    `incremental=True` re-fetches only the last ~30 days and merges them into
    the on-disk cache (append + de-dup), so refreshes cost ~30 days of network
    instead of ~26 years.
    """
    fmeta: dict = {}
    cached = _load_raw_cache()
    today = pd.Timestamp.today()

    if incremental and cached is not None and len(cached) > 0:
        last = cached.index.max()
        start = max(pd.Timestamp(last),
                    today - pd.Timedelta(days=INCREMENTAL_DAYS))
        start = start.strftime("%Y-%m-%d")
    else:
        start = HISTORY_START

    results: dict = {}
    # ZERO-CRASH + NON-BLOCKING design:
    #  * Do NOT use `with ThreadPoolExecutor(...)` — its __exit__ runs
    #    shutdown(wait=True) and would BLOCK on the slowest hung request,
    #    defeating FETCH_DEADLINE.
    #  * `as_completed(timeout=...)` raises concurrent.futures.TimeoutError on
    #    deadline. On Python <= 3.10 (Render's image) that class is DISTINCT
    #    from the builtin TimeoutError (they were only unified in 3.11), so
    #    `except TimeoutError` silently let it escape and crash Streamlit.
    #    Catch FuturesTimeoutError explicitly (a tuple also covers 3.11+).
    ex = ThreadPoolExecutor(max_workers=8)
    future_to_key = {ex.submit(_fetch_one, spec, start): key
                     for key, spec in RAW_SPECS.items()}
    try:
        for fut in as_completed(list(future_to_key.keys()),
                                timeout=FETCH_DEADLINE):
            key = future_to_key[fut]
            try:
                results[key] = fut.result()
            except Exception:
                results[key] = None
    except (FuturesTimeoutError, TimeoutError):
        # Deadline hit with stragglers: keep whatever finished, treat the rest
        # as missing. NEVER raise to the caller (this is what crashed Render).
        fmeta["timeout"] = True
    finally:
        for fut, key in future_to_key.items():
            if not fut.done():
                fut.cancel()
                results.setdefault(key, None)
        # cancel_futures=True (py3.9+) drops queued-but-unstarted work; any
        # already-running thread still honours its own per-request
        # FETCH_TIMEOUT, so the process is never held hostage by the network.
        ex.shutdown(wait=False, cancel_futures=True)

    # Merge into the cached history (fresh tail wins on overlap).
    full_idx = pd.date_range(HISTORY_START, today, freq="ME")
    if cached is not None:
        raw = cached.reindex(full_idx).copy()
    else:
        raw = pd.DataFrame(index=full_idx)

    for key, s in results.items():
        if s is None:
            continue
        s = s[s.index >= pd.Timestamp(HISTORY_START)]
        s = s[~s.index.duplicated(keep="last")]
        if key in raw.columns and raw[key].notna().any():
            raw[key] = s.combine_first(raw[key]).reindex(full_idx)
        else:
            raw[key] = s.reindex(full_idx)

    raw = raw.ffill(limit=6)
    for key in RAW_SPECS:
        fmeta[key] = "Y" if (key in raw.columns and raw[key].notna().any()) else "N"
    fmeta["_live"] = sum(1 for v in fmeta.values() if v == "Y")
    return raw, fmeta


# ---------------------------------------------------------------------------
# Feature construction from the (cached) raw frame
# ---------------------------------------------------------------------------
def compute_features_from_raw(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Turn the raw monthly frame into the 8 risk features as ROBUST Z-SCORES
    (V3: replaces the V2 trailing percentiles that saturated at 100 for years
    and pinned the composite at 99 — see rolling_robust_z).

    Factor matrix (weights live in WEIGHTS):
      F1 Valuation    CAPE / Buffett            — slow macro anchor
      F2 Leverage     FINRA Margin Debt Ratio   — NEW leverage-warning anchor
      F3 Credit       BAA10Y spread (inverted)  — credit expansion
      F4 Business     EMVMACROBUS (inverted)    — complacency
      F5 Tech Froth   QQQ/SPY 3y relative       — structural deviation
      F6 Momentum     S&P 500 6m ann. (10d-SMA) — trend
      F7 Volatility   VIX (inverted, 10d-SMA)   — short-term complacency
      F8 Liquidity    Fed WALCL YoY (+ M2 YoY)  — central-bank liquidity
    """
    meta: dict = {}
    idx = raw.index

    # First smoothing layer for the high-frequency indicators (VIX, S&P): pull
    # the DAILY series once and reuse it for F2 / F3 below.
    hf = _get_hf_daily()

    def g(key):
        return raw[key] if key in raw.columns else None

    cape, wilshire, gdp = g("cape"), g("wilshire"), g("gdp")
    vix, vixcls = g("vix"), g("vixcls")
    baa10y = g("baa10y")
    m2, walcl = g("m2"), g("walcl")
    emv = g("emv")
    cpi, ffr = g("cpi"), g("ffr")
    spx, spy, qqq = g("spx"), g("spy"), g("qqq")
    mgdte = g("mgdte")
    dgs10, dgs3mo = g("dgs10"), g("dgs3mo")

    feat = pd.DataFrame(index=idx)

    # ---- F1 Valuation (zero-fail) ---------------------------------------
    # Primary = CAPE and/or Buffett (Wilshire/GDP), blended with nanmean.
    # A ZERO-FAIL fallback then fills any remaining gaps with the S&P 500
    # premium above its ~200-week (~46-month) moving average (sourced live
    # via yfinance/Stooq). It is used ONLY where CAPE/Buffett are NaN, so it
    # never double-counts momentum — but it guarantees F1 is never blank.
    parts = []
    if cape is not None and cape.notna().any():
        parts.append(rolling_robust_z(cape))
    if (wilshire is not None and gdp is not None
            and wilshire.notna().any() and gdp.notna().any()):
        # GDP is QUARTERLY; FRED returns it only at Mar/Jun/Sep/Dec. A naive
        # monthly division (wilshire / gdp) leaves 8 of 12 months as NaN, and
        # np.mean([cape, buffett]) then becomes NaN at the latest date ->
        # F1 shows as Pending. Fix: promote GDP to a DAILY skeleton (.last()),
        # then forward-fill across days (.ffill()) so the quarter figure is
        # carried to every day, and collapse back to month-end. Every month
        # now carries the current quarter's GDP — cross-frequency NaN gone.
        # (Note: resample("D").ffill() alone fills WITHIN each day group and
        #  would NOT carry the value forward — the .last() step is required.)
        gdp_daily = gdp.resample("D").last().ffill()
        gdp_m = gdp_daily.resample("ME").last()
        buffett = (wilshire / gdp_m) * 1000.0   # scale-invariant ratio
        parts.append(rolling_robust_z(buffett.dropna()))

    primary = np.nanmean(parts, axis=0) if parts else None

    # Zero-fail fallback proxy: S&P 500 distance above its ~200-week MA.
    # A fat premium = expensive market = high risk (positive Z).
    ma_pct = None
    if spx is not None and spx.notna().sum() >= 60:
        ma_long = spx.rolling(46).mean()        # 46 months ~ 200 weeks
        prem = (spx / ma_long - 1.0) * 100.0
        ma_pct = rolling_robust_z(prem.dropna())

    if primary is not None:
        # fill only the NaN gaps in the primary with the MA proxy
        feat["F1_valuation"] = (primary.combine_first(ma_pct)
                                if ma_pct is not None else primary)
        meta["F1_valuation"] = (
            f"CAPE={'Y' if cape is not None and cape.notna().any() else 'N'} "
            f"Buffett={'Y' if wilshire is not None and gdp is not None and wilshire.notna().any() and gdp.notna().any() else 'N'} "
            f"MAfallback={'Y' if ma_pct is not None else 'N'}")
    elif ma_pct is not None:
        feat["F1_valuation"] = ma_pct
        meta["F1_valuation"] = "SPX_MA200w (primary fallback)"
    else:
        # Last-resort tertiary proxy: inverse S&P 500 dividend yield.
        sp500div = g("sp500div")
        if sp500div is not None and sp500div.notna().sum() >= 12:
            feat["F1_valuation"] = -rolling_robust_z(sp500div)
            meta["F1_valuation"] = "SP500DIV inverse (last resort)"
        else:
            feat["F1_valuation"] = np.nan
            meta["F1_valuation"] = "N"

    # ---- F1b CAPE vs its own 10y average (valuation sub-indicator) --------
    # "Forward PE vs 10y average" proxy: how stretched today's Shiller CAPE is
    # relative to its own trailing decade. A z-score of the log CAPE vs its
    # 10y mean, mapped to a percentile. Captures "expensive vs recent history".
    cape_z = None
    if cape is not None and cape.notna().sum() >= 120:
        log_cape = np.log(cape.replace(0, np.nan))
        mu = log_cape.rolling(120, min_periods=60).mean()
        sd = log_cape.rolling(120, min_periods=60).std()
        # already a Z vs the trailing 10y — used directly (no re-percentiling)
        cape_z = ((log_cape - mu) / sd.replace(0, np.nan)).clip(-4.0, 4.0)
        meta["F1b_cape_z"] = "CAPE z vs 10y"
    if cape_z is None:
        feat["F1b_cape_z"] = np.nan
        meta["F1b_cape_z"] = "N"
    else:
        feat["F1b_cape_z"] = cape_z

    # ---- F2 Leverage (FINRA Margin Debt Ratio) [NEW] ---------------------
    # Risk direction: a fast-growing margin-debt balance AND a high
    # debt-to-market ratio both flag leveraged speculation (bubble fuel). The
    # two historical percentiles are blended into one leverage score.
    # PRIMARY = FRED MGDTE (FINRA margin debt, monthly). FALLBACK (100% uptime
    # guarantee) = an interaction proxy of 12m S&P momentum + loose credit
    # (inverted BAA10Y): rising prices on easy credit historically ride on
    # expanding margin use, so the proxy tracks the same leverage regime.
    mg_primary = None
    if mgdte is not None and mgdte.notna().any():
        mg = mgdte.interpolate().ffill().bfill()
        mg_yoy = mg.pct_change(12) * 100.0
        if spx is not None and spx.notna().any():
            mg_ratio = (mg / spx).replace([np.inf, -np.inf], np.nan)
            mg_ratio = mg_ratio.interpolate().ffill().bfill()
            pct_ratio = rolling_robust_z(mg_ratio)
        else:
            pct_ratio = None
        pct_yoy = rolling_robust_z(mg_yoy)
        comps = [c for c in (pct_yoy, pct_ratio) if c is not None]
        mg_primary = np.nanmean(comps, axis=0) if comps else None
        meta["F2_leverage"] = "FINRA MGDTE (YoY + debt/SPX)"
    else:
        meta["F2_leverage"] = "proxy"

    # Fallback proxy (always computable from spx + baa10y)
    mg_fallback = None
    if (spx is not None and spx.notna().any()
            and baa10y is not None and baa10y.notna().any()):
        spx_ret12 = spx.pct_change(12) * 100.0
        credit_ease = -rolling_robust_z(baa10y)     # loose credit = high ease
        r1 = rolling_robust_z(spx_ret12)
        mg_fallback = r1 * 0.6 + credit_ease * 0.4
        meta["F2_leverage"] += " + SPX12m/BAA proxy"

    if mg_primary is not None:
        feat["F2_leverage"] = mg_primary
        if mg_fallback is not None:
            feat["F2_leverage"] = feat["F2_leverage"].combine_first(mg_fallback)
    elif mg_fallback is not None:
        feat["F2_leverage"] = mg_fallback
    else:
        feat["F2_leverage"] = np.nan
        meta["F2_leverage"] = "N"

    # ---- F6 Momentum (6m annualized, 10-day SMA pre-smoothed) -----------
    # FIRST-pass denoise: take the 10-trading-day SMA of DAILY S&P 500, then
    # resample to month-end and compute the 6-month annualized return. This
    # strips intra-month whipsaw before it ever reaches the percentile.
    f6_src = None
    if "spx" in hf and hf["spx"].notna().any():
        spx_sma = hf["spx"].rolling(10).mean().dropna()
        spx_m = spx_sma.resample("ME").last().dropna()
        if len(spx_m) >= 6:
            mom6 = (spx_m / spx_m.shift(6)) ** (12.0 / 6.0) - 1.0
            f6_src = rolling_robust_z(mom6)
            meta["F6_momentum"] = "SPX 10d-SMA -> 6m ann."
    if f6_src is None and spx is not None and spx.notna().any():
        mom6 = (spx / spx.shift(6)) ** (12.0 / 6.0) - 1.0
        f6_src = rolling_robust_z(mom6)
        meta["F6_momentum"] = "SPX monthly (SMA fallback)"
    feat["F6_momentum"] = f6_src if f6_src is not None else np.nan
    if meta.get("F6_momentum") is None:
        meta["F6_momentum"] = "N"

    # ---- F7 Volatility (VIX inverted, 10-day SMA pre-smoothed) -----------
    # Denoise VIX with a 10-trading-day SMA before inverting into a risk
    # percentile, so a single vol spike doesn't paint a false "all-clear".
    f7_src = None
    if "vix" in hf and hf["vix"].notna().any():
        vix_sma = hf["vix"].rolling(10).mean().dropna()
        vix_m = vix_sma.resample("ME").last().dropna()
        if not vix_m.empty:
            f7_src = -rolling_robust_z(vix_m)
            meta["F7_volatility"] = "VIX 10d-SMA (inv)"
    if f7_src is None:
        v = vix if (vix is not None and vix.notna().any()) else vixcls
        if v is not None and v.notna().any():
            f7_src = -rolling_robust_z(v)
            meta["F7_volatility"] = "VIX/VIXCLS monthly (inv)"
    feat["F7_volatility"] = f7_src if f7_src is not None else np.nan
    if meta.get("F7_volatility") is None:
        meta["F7_volatility"] = "N"

    # ---- F3 Credit (credit spread INVERTED) -----------------------------
    # A compressed spread (blind risk-chasing, ultra-loose credit) is a bubble
    # signal; a wide spread marks panic (2008, 2020-03) — the opposite of froth.
    if baa10y is not None and baa10y.notna().any():
        feat["F3_credit"] = -rolling_robust_z(baa10y)
        meta["F3_credit"] = "Credit(inv)=Y"
    else:
        feat["F3_credit"] = np.nan
        meta["F3_credit"] = "N"

    # ---- F3b Real Rate (FedFunds - CPI yoy, INVERTED) --------------------
    # Low / negative real policy rate = loose financial conditions = risk.
    real_rate = None
    if ffr is not None and cpi is not None and ffr.notna().any() and cpi.notna().any():
        cpi_yoy = cpi.pct_change(12) * 100.0
        rr = (ffr - cpi_yoy).replace([np.inf, -np.inf], np.nan)
        real_rate = rolling_robust_z(rr)   # high real rate = late-cycle tightening risk
        meta["F3b_realrate"] = "FedFunds-CPI (z)"
    if real_rate is None:
        feat["F3b_realrate"] = np.nan
        meta["F3b_realrate"] = "N"
    else:
        feat["F3b_realrate"] = real_rate          # already inverted (high = tight)

    # ---- F3c Yield Curve (10Y-3M spread, INVERTED) -----------------------
    # An inverted / flat curve is a classic late-cycle risk signal.
    # SIGN FIX: the old percentile version used +pct(spread) — which reads HIGH
    # when the curve is STEEP, the exact opposite of the documented intent
    # (this is part of why 2000/2007 showed no macro-module risk). Now:
    # Z of the NEGATED spread -> inverted curve = high risk.
    yc = None
    if dgs10 is not None and dgs3mo is not None and dgs10.notna().any() and dgs3mo.notna().any():
        spread = (dgs10 - dgs3mo).replace([np.inf, -np.inf], np.nan)
        yc = -rolling_robust_z(spread)
        meta["F3c_yield"] = "10Y-3M (inv, z)"
    if yc is None:
        feat["F3c_yield"] = np.nan
        meta["F3c_yield"] = "N"
    else:
        feat["F3c_yield"] = yc

    # ---- F8 Liquidity (M2 YoY + Fed BS YoY) ------------------------------
    parts = []
    if m2 is not None and m2.notna().any():
        parts.append(rolling_robust_z(m2.pct_change(12) * 100.0))
    if walcl is not None and walcl.notna().any():
        parts.append(rolling_robust_z(walcl.pct_change(12) * 100.0))
    feat["F8_liquidity"] = np.nanmean(parts, axis=0) if parts else np.nan
    meta["F8_liquidity"] = (f"M2={'Y' if m2 is not None and m2.notna().any() else 'N'} "
                            f"FedBS={'Y' if walcl is not None and walcl.notna().any() else 'N'}")

    # ---- F4 Business sentiment (FRED EMVMACROBUS, INVERTED) --------------
    # Low index = complacency = bubble-prone -> invert. FRED series, so with a
    # FRED_API_KEY it is as stable as every other macro feature. Only if it is
    # entirely missing do we fall back to the keyless AAII bullish survey.
    if emv is not None and emv.notna().sum() >= 12:
        feat["F4_business"] = -rolling_robust_z(emv)
        meta["F4_business"] = "EMVMACROBUS (FRED, inv)"
    else:
        aaii = _aaii_sentiment()
        if aaii is not None and aaii.notna().sum() >= 6:
            # High bullish (complacency) = risk -> positive Z.
            feat["F4_business"] = rolling_robust_z(aaii)
            meta["F4_business"] = "AAII bullish (EMV fallback)"
        else:
            feat["F4_business"] = np.nan
            meta["F4_business"] = "N"

    # ---- F5 Tech froth (3-year window) ----------------------------------
    # ONE consistent ratio — never splice different-scale pairs (the earlier
    # qqq/spy + ixic/spx combine_first created a level discontinuity that the
    # Z-score read as a fake ±4σ crash/spike, corrupting the dot-com anchor).
    # PRIMARY: NASDAQ Composite / S&P 500 — BOTH FRED-backed (NASDAQCOM 1971+,
    # SP500), full dot-com coverage, survives a total Yahoo+Stooq outage.
    # FALLBACK: QQQ/SPY (NDX froth; only from 1999).
    ixic = g("ixic")
    ratio, ratio_src = None, None
    if (ixic is not None and spx is not None
            and ixic.notna().any() and spx.notna().any()):
        ratio = (ixic / spx).dropna()
        ratio_src = "IXIC/SPX (FRED-backed)"
    elif (qqq is not None and spy is not None
            and qqq.notna().any() and spy.notna().any()):
        ratio = (qqq / spy).dropna()
        ratio_src = "QQQ/SPY"
    if ratio is not None and ratio.notna().any():
        # Measure divergence against the LONG 20y window (like every other
        # feature), NOT the old 3y recency window: a 36-month trailing
        # median/MAD made F5 a fast momentum gauge whose z whipsawed
        # (1.2 -> 3.4 -> 1.2 within weeks) and jerked the whole composite
        # around. A 6-month EMA pre-smooth kills one-month ratio spikes;
        # with the 20y window the dot-com extreme stays the reference max,
        # so later readings are calm and comparable across eras.
        ratio_sm = ratio.ewm(span=6, min_periods=3).mean()
        feat["F5_tech"] = rolling_robust_z(
            ratio_sm, window=WINDOW_MONTHS, min_periods=60)
        meta["F5_tech"] = ratio_src
    else:
        feat["F5_tech"] = np.nan
        meta["F5_tech"] = "N"

    return feat, meta


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------
def _save_cache(raw: pd.DataFrame, feat: pd.DataFrame, score: pd.Series,
                meta: dict) -> None:
    try:
        out = raw.copy()
        for c in feat.columns:
            out[f"feat_{c}"] = feat[c]
        out["score"] = score
        out.to_parquet(CACHE_PATH)
    except Exception:
        pass
    try:
        # Cache-format marker: feat_* columns are ROBUST Z-SCORES (V3). Readers
        # must reject caches without this marker (they hold V2 percentiles and
        # would be silently misread as Z, producing garbage scores).
        meta = dict(meta)
        meta["feat_format"] = FEAT_FORMAT
        meta["written_at"] = pd.Timestamp.now().isoformat()
        with open(META_PATH, "w") as f:
            json.dump(meta, f, default=str)
    except Exception:
        pass


def cache_info() -> dict:
    """When was the on-disk score cache last written, and how old is it now?
    Used by the dashboard to decide an auto-refresh and to surface the data
    date to the user. Never raises.
    """
    info = {"source": None, "written_at": None, "age_hours": None,
            "score_as_of": None}
    try:
        if os.path.exists(META_PATH):
            meta = json.load(open(META_PATH))
            info["source"] = meta.get("source")
            wa = meta.get("written_at")
            if wa:
                t = pd.Timestamp(wa)
                info["written_at"] = t
                info["age_hours"] = (pd.Timestamp.now() - t).total_seconds() / 3600.0
    except Exception:
        pass
    try:
        if os.path.exists(CACHE_PATH):
            s = pd.read_parquet(CACHE_PATH).get("score")
            if s is not None:
                s = s.dropna()
                if not s.empty:
                    info["score_as_of"] = s.index[-1]
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Synthetic fallback (deterministic) — only used when every API fails
# ---------------------------------------------------------------------------
def _synthetic_scores(start: str = LIVE_START) -> pd.Series:
    idx = pd.date_range(start, pd.Timestamp.today(), freq="ME")
    t = np.arange(len(idx))
    base = 46.0 + 7.0 * np.sin(t / 45.0)

    def peak(date_str, height, width):
        center = (pd.Timestamp(date_str) - idx[0]).days / 30.44
        return height * np.exp(-(((t - center) / width) ** 2))

    bumps = (peak("2000-03-01", 18, 5)
             + peak("2007-10-01", 11, 7)
             + peak("2021-12-01", 26, 6)
             + peak("2025-01-01", 6, 9))
    rng = np.random.RandomState(7)
    s = base + bumps + rng.normal(0, 2.0, len(idx))
    # Bounded by construction (max ~46+26+18+noise ≈ 92); the live pipeline
    # still applies a hard clip(1, 99) as a belt-and-braces boundary guarantee.
    return pd.Series(s, index=idx)


# ---------------------------------------------------------------------------
# Connectivity probe (for the on-page diagnostics panel)
# ---------------------------------------------------------------------------
def probe_connectivity(timeout: int = 6) -> dict:
    """Probe each upstream endpoint and report per-target status + latency.

    Used by the dashboard's diagnostics expander so a failing deploy (e.g.
    blocked egress, rate-limited shared IP, missing FRED_API_KEY) is visible
    from the page itself instead of needing Render shell/log access.
    NEVER raises.
    """
    import time
    out: dict = {"fred_api_key_set": bool(FRED_API_KEY), "targets": []}

    def _probe(name: str, url: str, expect: Optional[str] = None) -> None:
        t0 = time.time()
        txt = _http_get(url, timeout=timeout)
        dt = int((time.time() - t0) * 1000)
        if txt is None:
            out["targets"].append({"target": name, "ok": False, "ms": dt,
                                   "detail": "request failed (timeout/blocked)"})
            return
        ok = (expect in txt) if expect else bool(txt.strip())
        detail = f"{len(txt)} bytes" if ok else \
            f"unexpected payload ({txt[:60]!r})"
        out["targets"].append({"target": name, "ok": ok, "ms": dt,
                               "detail": detail})

    _probe("FRED keyless CSV (fredgraph.csv)",
           "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
           expect="VIXCLS")
    if FRED_API_KEY:
        _probe("FRED API (with key)",
               f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id=VIXCLS&api_key={FRED_API_KEY}&file_type=csv",
               expect="VIXCLS")
    _probe("Stooq daily CSV (SPX.US)",
           "https://stooq.com/q/d/l/?s=SPX.US&i=d", expect="Close")
    _probe("Yahoo Finance chart API (^GSPC)",
           "https://query1.finance.yahoo.com/v8/finance/chart/"
           "%5EGSPC?range=5d&interval=1d", expect="chart")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _cached_scores(tail_boost: Optional[bool] = None
                   ) -> Optional[Tuple[pd.Series, dict]]:
    """(monthly_score_series, meta) rebuilt from the on-disk feature cache, or
    None if the cache is missing/unusable. Pure local I/O — instant, no network.

    The composite is always recomputed from the stored features with the CURRENT
    tail_boost setting, so toggling the switch takes effect immediately on
    cached data (no refetch needed).
    """
    if not (os.path.exists(CACHE_PATH) and os.path.exists(META_PATH)):
        return None
    try:
        # Reject legacy percentile caches (see FEAT_FORMAT).
        _m = json.load(open(META_PATH))
        if _m.get("feat_format") != FEAT_FORMAT:
            return None
        cached = pd.read_parquet(CACHE_PATH)
        fcols = [c for c in cached.columns if c.startswith("feat_")]
        if not fcols:
            return None
        feat = cached[fcols].copy()
        feat.columns = [c[5:] for c in feat.columns]
        score = compute_composite(feat, tail_boost=tail_boost).dropna()
        if score.empty:
            return None
        meta = json.load(open(META_PATH))
        meta["source"] = "cache"
        latest = score.index[-1]
        live_cols = [c for c in WEIGHTS
                     if WEIGHTS[c] > 0 and c in feat.columns
                     and not pd.isna(feat[c].get(latest, np.nan))]
        meta["available_count"] = len(live_cols)
        return score, meta
    except Exception:
        return None


def _synthetic_result() -> Tuple[pd.Series, dict]:
    return _synthetic_scores(), {
        "source": "synthetic",
        "available_count": 0,
        "features": {},
        "note": "All live APIs unavailable; showing deterministic synthetic series. Check network / FRED_API_KEY.",
    }


def get_monthly_scores(refresh: bool = False,
                        tail_boost: Optional[bool] = None) -> Tuple[pd.Series, dict]:
    """
    Returns (monthly_score_series, meta). NEVER raises — the dashboard must
    always render, even with zero network.

    Resolution order:
      1. cache-first fast path (instant, no network) when refresh=False;
      2. live INCREMENTAL fetch (last ~30 days) merged into the cache;
      3. on ANY fetch/compute failure -> the on-disk cache (even if stale);
      4. only if nothing real exists -> the flagged deterministic synthetic.

    `meta['source']` is one of 'live', 'cache', 'synthetic'.
    """
    # 1) Instant path: serve the baked/on-disk cache without touching network —
    #    BUT only when it is FRESH enough. If the cache is older than
    #    CACHE_MAX_AGE_HOURS (or has no timestamp — e.g. a legacy build-time
    #    cache), fall through to the live incremental refresh so the charts
    #    and backtest always pull the latest trading day. The refresh is
    #    crash-proof and falls back to this same cache on any failure.
    if not refresh:
        hit = _cached_scores(tail_boost)
        if hit is not None:
            info = cache_info()
            age = info.get("age_hours")
            if age is not None and age < CACHE_MAX_AGE_HOURS:
                return hit
            # stale cache: proceed to the live incremental refresh below

    # 2) Live path — wrapped so a network/compute failure can NEVER crash the
    #    app (this whole block is what used to propagate the TimeoutError).
    try:
        incremental = os.path.exists(CACHE_PATH)
        raw, fmeta = fetch_all_raw(incremental=incremental)
        feat, fmeta_feat = compute_features_from_raw(raw)
        score = compute_composite(feat, tail_boost=tail_boost).dropna()
    except Exception as exc:
        print(f"[score] live fetch/compute failed ({exc!r}); "
              f"falling back to cache/synthetic")
        hit = _cached_scores(tail_boost)
        return hit if hit is not None else _synthetic_result()

    # ---- Live-feature accounting (features valid at the latest date) ------
    latest = score.index[-1] if not score.empty else None
    live_cols = [c for c in WEIGHTS
                 if WEIGHTS[c] > 0 and c in feat.columns
                 and latest is not None and not pd.isna(feat[c].get(latest, np.nan))]
    available_count = len(live_cols)
    missing = [c for c in WEIGHTS if WEIGHTS[c] > 0 and c not in live_cols]
    print(f"[score] {available_count} features live at "
          f"{latest.date() if latest is not None else 'n/a'}"
          f"  (missing: {missing or '-'})")

    # All-fail extreme case: prefer REAL cached history over synthetic.
    if latest is None or available_count == 0:
        hit = _cached_scores(tail_boost)
        if hit is not None:
            return hit
        return _synthetic_result()

    meta_features = {**fmeta, **fmeta_feat}
    _save_cache(raw, feat, score, {
        "source": "live", "available_count": available_count,
        "features": meta_features,
    })
    return score, {"source": "live", "available_count": available_count,
                   "features": meta_features}


def _stress_flag(vix: pd.Series, spx: pd.Series, baa: pd.Series,
                 idx: pd.DatetimeIndex) -> pd.Series:
    """Per-day stress flag (0 = calm, 1 = genuine stress) for the clamp.

    Stress = VIX > STRESS_VIX  OR  trailing-21d S&P drop < STRESS_DROP  OR
    BAA10Y widened > STRESS_CREDIT_JUMP MoM. The daily clamp is relaxed only
    when this flag fires, so normal regimes are hard-limited to DAILY_CLAMP.
    """
    flag = pd.Series(0.0, index=idx)
    if vix is not None and len(vix):
        v = vix.reindex(idx, method="ffill")
        flag = np.maximum(flag, (v > STRESS_VIX).astype(float).fillna(0.0))
    if spx is not None and len(spx):
        s = spx.reindex(idx, method="ffill")
        drop = s.pct_change(21)
        flag = np.maximum(flag, (drop < STRESS_DROP).astype(float).fillna(0.0))
    if baa is not None and len(baa):
        b = baa.reindex(idx, method="ffill")
        jump = b.diff(21)            # ~1 month MoM change (in level, bps-ish)
        flag = np.maximum(flag, (jump > STRESS_CREDIT_JUMP).astype(float).fillna(0.0))
    return flag.clip(0.0, 1.0)


def stability_filter(score: pd.Series, vix: pd.Series = None,
                     spx: pd.Series = None, baa: pd.Series = None) -> pd.Series:
    """K-line style two-timescale stability layer.

    Decompose the (monthly -> daily up-sampled) raw score into:

      trend = slow EMA (TREND_SPAN)        -> mid/long-term macro wave
      fast  = fast EMA (OSC_SPAN)          -> short-term information
      osc   = clip((fast - trend) * OSC_GAIN, ±OSC_MAX)
              -> a SMALL, bounded short-term oscillation (the "K-line wiggle")
      x     = trend + osc

    then run a stress-aware daily clamp on x: |Δ| <= DAILY_CLAMP on normal
    days, relaxed to STRESS_CLAMP when a genuine stress regime fires (VIX
    spike / 21-day crash drop / credit blowout), so real risk events still
    break out quickly instead of being smoothed away.

    Result: the index keeps its multi-year cycle amplitude (not over-smoothed),
    shows only minor day-to-day movement (no violent whipsaw), yet reacts
    decisively at true crash onsets and capitulation bottoms. Every step is
    causal — no look-ahead.
    """
    if score is None or len(score) == 0:
        return score
    idx = score.index
    trend = score.ewm(span=TREND_SPAN, adjust=False).mean()
    fast = score.ewm(span=OSC_SPAN, adjust=False).mean()
    osc = ((fast - trend) * OSC_GAIN).clip(-OSC_MAX, OSC_MAX)
    vals = (trend + osc).to_numpy(dtype=float)
    flag = _stress_flag(vix, spx, baa, idx).to_numpy(dtype=float)

    out = np.empty_like(vals)
    prev = np.nan
    for i in range(len(vals)):
        x = vals[i]
        if np.isnan(x):
            out[i] = prev
            continue
        if np.isnan(prev):
            out[i] = x
            prev = x
            continue
        lim = DAILY_CLAMP + (STRESS_CLAMP - DAILY_CLAMP) * flag[i]
        delta = x - prev
        if abs(delta) > lim:
            x = prev + np.sign(delta) * lim
        out[i] = x
        prev = x
    return pd.Series(out, index=idx)


def get_daily_scores(refresh: bool = False,
                      tail_boost: Optional[bool] = None) -> pd.Series:
    """Daily, stability-filtered Bubble Risk Score for charting.

    The composite is computed MONTHLY (the percentile normalization needs a
    long trailing window). We up-sample it onto a daily calendar — forward
    filling the most recent month-end reading to every day — then run it through
    the K-line style stability layer: a slow-EMA trend + a small bounded
    oscillation + a stress-aware daily clamp (<= DAILY_CLAMP pts unless
    genuinely stressed). The result is a calm macro wave with a gentle
    short-term wiggle that cannot whipsaw.

    Returns an empty Series if no monthly scores are available.
    """
    monthly, _ = get_monthly_scores(refresh=refresh, tail_boost=tail_boost)
    monthly = monthly.dropna()
    if monthly.empty:
        return monthly

    # load the daily vol-regime signals for the stress-aware clamp + regime
    hf = _get_hf_daily()
    vix_d = hf.get("vix") if hf else None
    spx_d = hf.get("spx") if hf else None

    # Cap the daily index at the LATEST market-data date (last daily price
    # close), NOT at the calendar 'today' — otherwise the red index line pokes
    # one or more calendar days past the price lines with a flat value (no new
    # information). Fall back to today only if no daily prices are available.
    end = pd.Timestamp.today().normalize()
    for _s in (vix_d, spx_d):
        if _s is not None and not _s.empty:
            cand = _s.index[-1].normalize()
            if cand < end:
                end = cand
    daily_idx = pd.date_range(monthly.index.min(), end, freq="D")
    daily = monthly.reindex(daily_idx, method="ffill")

    # monthly BAA10Y for the credit-jump stress test
    baa = None
    if os.path.exists(CACHE_PATH):
        try:
            baa = pd.read_parquet(CACHE_PATH)["baa10y"]
        except Exception:
            baa = None
    daily_anchor = stability_filter(daily, vix=vix_d, spx=spx_d, baa=baa)

    # --- Daily market-regime overlay --------------------------------------
    # The pure monthly anchor is too smooth to track day-to-day market moves
    # (CAPE / MGDTE / EMV are monthly or lag — they can't see today's tape).
    # Add a DAILY regime composite from SPX momentum + VIX + SPX-vs-200d-MA
    # extension, robust-z'd, mapped to 0-100, and blend it into the anchor.
    # The headline score (gauge) stays on the validated monthly macro; this
    # overlay only affects the displayed daily series so the chart can react.
    # Regime is itself smoothed (15d EMA before Φ) so it doesn't add raw
    # day-to-day noise; a daily-change clamp (BLEND_DAILY_CLAMP) below
    # guarantees the displayed line can't sawtooth regardless of inputs.
    regime = compute_daily_regime(hf) if hf is not None else None
    if regime is not None and not regime.empty:
        common = daily_anchor.index.intersection(regime.index)
        if not common.empty:
            blended = (DAILY_ANCHOR_WEIGHT * daily_anchor.loc[common]
                       + (1 - DAILY_ANCHOR_WEIGHT) * regime.loc[common])
            # Hard daily-change clamp on the WHOLE daily index series (blended where
    # common, pure stability_filter otherwise). Iterating over the full
    # array -- not just the common slice -- catches transitions where the
    # blend ends and the index falls back to the macro-only stability_filter
    # value (otherwise those edges produce unsmoothed sawtooth jumps).
    # Multi-week moves still accumulate because each day's clamped delta is
    # at most BLEND_DAILY_CLAMP.
    limit = BLEND_DAILY_CLAMP
    arr = daily_anchor.to_numpy(dtype=float).copy()
    prev_valid = np.nan
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v):
            continue
        if np.isnan(prev_valid):
            prev_valid = v                # seed the first valid value
            continue
        d = v - prev_valid
        if d > limit:
            arr[i] = prev_valid + limit
        elif d < -limit:
            arr[i] = prev_valid - limit
        prev_valid = arr[i]
    daily_anchor[:] = arr
    return daily_anchor.dropna()


# Daily regime overlay weight: macro anchor 0.70, daily market regime 0.30.
# Tuned so the displayed daily index stays anchored to the macro reality
# (the validated monthly composite) while the daily overlay reacts to the
# tape. Push closer to 1.0 for a calmer chart, closer to 0 for a more
# SPX-correlated chart.
DAILY_ANCHOR_WEIGHT = 0.70

# Final hard clamp on |day-over-day Δ| of the blended displayed daily index.
# Trend still accumulates over weeks/months (the clamp is cumulative-friendly:
# a 0.5-pt daily move compounds to ~10pts over a month), but no single day
# can produce a sawtooth spike. Tuned so a typical regime day-over-day move
# (smoothed 15d) is allowed, but month-end anchor steps and any input glitch
# are squashed.
BLEND_DAILY_CLAMP = 1.2


def compute_daily_regime(hf: dict = None) -> pd.Series:
    """Daily 'market regime' 0-100 score from pure price/vix inputs.

    Three signals, each robust-z'd over a 252-day ( (1y) trailing window so
    the regime can swing with the tape while staying bounded:
      - SPX 20-day return (momentum / risk-on)
      - VIX level INVERTED (high VIX = fear = risk dropping)
      - SPX distance above its 200-day moving average (extension / froth)
    Equal-weighted average z -> short-EMA smoothed -> Φ(z) -> 0-100. The
    short EMA (5d) is what prevents the regime from zigzagging day to day
    when blended into the daily index. Clip [1, 99]. Falls back to an empty
    Series if no daily data; callers must handle that.

    This is the daily overlay blended into the displayed daily index so the
    chart tracks market moves. It does NOT replace the validated monthly
    composite -- the headline gauge and the backtest keep using the macro
    signal. The blend lives only in get_daily_scores.
    """
    if hf is None:
        hf = _get_hf_daily()
    if not hf:
        return pd.Series(dtype=float)
    spx = hf.get("spx")
    vix = hf.get("vix")
    if spx is None or spx.empty or vix is None or vix.empty:
        return pd.Series(dtype=float)

    mom_20 = spx.pct_change(20)
    mom_z = rolling_robust_z(mom_20, window=252, min_periods=60)
    vix_z = -rolling_robust_z(vix, window=252, min_periods=60)   # inverted
    ma200 = spx.rolling(200, min_periods=60).mean()
    ext = spx / ma200 - 1.0
    ext_z = rolling_robust_z(ext, window=252, min_periods=60)

    common = mom_z.index.intersection(vix_z.index).intersection(ext_z.index)
    common = common[~common.duplicated(keep="first")]
    if common.empty:
        return pd.Series(dtype=float)
    combined = ((mom_z.reindex(common).fillna(0)
                + vix_z.reindex(common).fillna(0)
                + ext_z.reindex(common).fillna(0)) / 3.0)
    # Short-EMA smoothing BEFORE Φ-mapping: kills the day-to-day jitter so the
    # blended daily index has a recognisable regime trend (week-to-week) rather
    # than a per-day sawtooth. 15d span (~7d half-life) gives the regime time
    # to settle into its new level after each move while still reacting to
    # multi-day trends within a session or two.
    combined_sm = combined.ewm(span=15, min_periods=1).mean()
    # robust-z -> 0-100 via standard normal CDF
    regime = (100.0 * _norm_cdf_arr(combined_sm.to_numpy())).clip(1, 99)
    return pd.Series(regime, index=common)


def risk_level(score: float) -> str:
    """Map a 0-100 score to its V2 risk-level label."""
    if pd.isna(score):
        return "Unknown"
    for lo, hi, _, label in RISK_BANDS:
        if lo <= score < hi:
            return label
    return RISK_BANDS[-1][3]


def _assemble_state(score_series: pd.Series, feat: pd.DataFrame,
                    meta: dict) -> dict:
    score_series = score_series.dropna()
    if score_series.empty:
        return {"score": np.nan, "status": "Unknown", "features": {},
                "modules": {}, "drivers": [], "hist_pct": np.nan,
                "source": meta.get("source", "unknown"), "as_of": None,
                "meta": meta}
    latest_date = score_series.index[-1]
    latest_score = float(score_series.iloc[-1])

    # granular factor detail (for the expandable breakdown). feat holds raw
    # Z-scores (V3); the dashboard cards expect a 0-100 percentile-style
    # reading, so we convert via z_display (100·Φ(z)) at the boundary.
    features = {}
    nonempty_feat = feat if feat is not None and not feat.empty else None
    for col in WEIGHTS:
        if nonempty_feat is not None and latest_date in nonempty_feat.index:
            val = nonempty_feat.loc[latest_date].get(col, np.nan)
        else:
            val = np.nan
        features[col] = {
            "score": None if pd.isna(val) else float(z_display(val)),
            "weight": WEIGHTS[col],
            "label": FEATURE_LABELS.get(col, col),
            "available": not pd.isna(val),
        }

    # ---- 5 module scores at the latest date (Z -> display 0-100) ----------
    modules = {}
    mod_df = None
    if nonempty_feat is not None:
        mod_df = compute_modules(nonempty_feat, tail_boost=meta.get("_tb"))
        for m in MODULE_WEIGHTS:
            v = mod_df[m].get(latest_date, np.nan)
            modules[m] = None if pd.isna(v) else float(z_display(v))

    # ---- historical percentile of the current reading --------------------
    hist_pct = float((score_series <= latest_score).mean() * 100.0)

    # ---- month-over-month drivers (which modules moved the score) ---------
    # Deltas are computed on the DISPLAY scale (0-100) so the panel reads in
    # the same units as the module cards.
    drivers = []
    if mod_df is not None and len(score_series) >= 2:
        prev_date = score_series.index[-2]
        for m in MODULE_WEIGHTS:
            cur = mod_df[m].get(latest_date, np.nan)
            prv = mod_df[m].get(prev_date, np.nan)
            if pd.notna(cur) and pd.notna(prv):
                drivers.append({"module": m,
                                "delta": float(z_display(cur) - z_display(prv)),
                                "weight": MODULE_WEIGHTS[m]})
        drivers.sort(key=lambda d: abs(d["delta"] * d["weight"]), reverse=True)

    return {"score": latest_score, "status": risk_level(latest_score),
            "features": features, "modules": modules, "drivers": drivers,
            "hist_pct": hist_pct, "source": meta.get("source", "unknown"),
            "as_of": latest_date, "meta": meta}


def get_latest_state(refresh: bool = False,
                      tail_boost: Optional[bool] = None) -> dict:
    """Latest score + module detail + drivers for the dashboard.

    Reads the SAVED cache written by ``get_monthly_scores`` so the dashboard
    never triggers a second network fetch; falls back to a direct compute only
    when no cache exists.
    """
    tb = TAIL_BOOST_ON if tail_boost is None else tail_boost
    if os.path.exists(CACHE_PATH):
        try:
            _fmt_ok = True
            if os.path.exists(META_PATH):
                try:
                    _fmt_ok = (json.load(open(META_PATH)).get("feat_format")
                               == FEAT_FORMAT)
                except Exception:
                    _fmt_ok = False
            else:
                _fmt_ok = False
            cached = pd.read_parquet(CACHE_PATH) if _fmt_ok else pd.DataFrame()
            fcols = [c for c in cached.columns if c.startswith("feat_")]
            if fcols:
                feat = cached[fcols].copy()
                feat.columns = [c[5:] for c in feat.columns]
                score_series = compute_composite(feat, tail_boost=tb)
                meta = {}
                if os.path.exists(META_PATH):
                    try:
                        meta = json.load(open(META_PATH))
                    except Exception:
                        pass
                meta["_tb"] = tb
                return _assemble_state(score_series, feat, meta)
        except Exception as exc:
            print(f"[state] cache read failed: {exc}")
    # No cache: compute once.
    score, meta = get_monthly_scores(refresh=refresh, tail_boost=tb)
    meta = dict(meta)
    meta["_tb"] = tb
    feat = None
    if os.path.exists(CACHE_PATH):
        try:
            cached = pd.read_parquet(CACHE_PATH)
            fcols = [c for c in cached.columns if c.startswith("feat_")]
            if fcols:
                feat = cached[fcols].copy()
                feat.columns = [c[5:] for c in feat.columns]
        except Exception:
            feat = None
    return _assemble_state(score, feat if feat is not None else pd.DataFrame(), meta)


def historical_benchmarks(score_series: pd.Series) -> dict:
    """Snapshot the calibrated score at the canonical bubble episodes, so the
    dashboard can show 'Compared with historical bubbles'.

    Returns {episode: {"date":..., "score":...}} using the max reading inside
    each episode window (data-driven, no hard-coded level).
    """
    out = {}
    episodes = {
        "dotcom_2000": ("2000-01-01", "2001-06-30"),
        "gfc_2007": ("2007-01-01", "2008-12-31"),
        "covid_pre": ("2019-10-01", "2020-02-29"),
        "bubble_2021": ("2021-01-01", "2022-01-31"),
    }
    s = score_series.dropna()
    for name, (a, b) in episodes.items():
        seg = s.loc[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
        if not seg.empty:
            idx = seg.idxmax()
            out[name] = {"date": idx.strftime("%Y-%m"), "score": float(seg.max())}
    return out


def opportunity_benchmarks(score_series: pd.Series) -> dict:
    """Mirror of historical_benchmarks but for the great ACCUMULATION windows —
    the local TROUGHS of the index inside each historical fear climax. These are
    the moments (2002, 2009, 2020, late-2022) when forward 12–24m returns were
    historically the strongest — the 'big buying opportunity' guidance.

    Returns {episode: {"date":..., "score":...}} using the min reading inside
    each window (data-driven, no hard-coded level).
    """
    out = {}
    episodes = {
        "dotcom_trough_2002": ("2002-07-01", "2003-05-31"),
        "gfc_trough_2009": ("2008-11-01", "2009-05-31"),
        "covid_trough_2020": ("2020-03-01", "2020-05-31"),
        "bear_trough_2022": ("2022-09-01", "2022-12-31"),
    }
    s = score_series.dropna()
    for name, (a, b) in episodes.items():
        seg = s.loc[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
        if not seg.empty:
            idx = seg.idxmin()
            out[name] = {"date": idx.strftime("%Y-%m"), "score": float(seg.min())}
    return out


# --- Event detection thresholds --------------------------------------------
RISK_EVENT_LEVEL = 75.0    # local peak >= this -> a speculative-risk climax
OPP_EVENT_LEVEL = 35.0     # local trough <= this -> an accumulation opportunity
EVENT_WIN = 45             # half-window (days) for the local-peak/trough test
EVENT_MIN_GAP = 150        # merge triggers closer than this into one event


def detect_events(score: pd.Series) -> dict:
    """Data-driven event flags on the (smoothed) score series.

    Returns {"risk": [(Timestamp, score), ...],
             "opportunity": [(Timestamp, score), ...]} where:
      * risk        = local maxima >= RISK_EVENT_LEVEL (risk climaxes to trim),
      * opportunity = local minima <= OPP_EVENT_LEVEL (fear climaxes to buy).

    The neighbourhood test uses the full window (past AND future) because these
    markers annotate HISTORY on the chart — they are context for the viewer,
    not a live trading signal. Triggers closer than EVENT_MIN_GAP days are
    merged so a single episode renders as one marker.
    """
    res = {"risk": [], "opportunity": []}
    s = score.dropna()
    if len(s) < 2 * EVENT_WIN + 5:
        return res
    vals = s.to_numpy(dtype=float)
    idx = s.index
    last_risk = None
    last_opp = None
    for i in range(len(vals)):
        v = vals[i]
        if np.isnan(v):
            continue
        lo = max(0, i - EVENT_WIN)
        hi = min(len(vals), i + EVENT_WIN + 1)
        seg = vals[lo:hi]
        seg = seg[~np.isnan(seg)]
        if seg.size == 0:
            continue
        d = idx[i]
        if v >= RISK_EVENT_LEVEL and v >= np.nanmax(seg):
            if last_risk is None or (d - last_risk).days > EVENT_MIN_GAP:
                res["risk"].append((d, float(v)))
                last_risk = d
        if v <= OPP_EVENT_LEVEL and v <= np.nanmin(seg):
            if last_opp is None or (d - last_opp).days > EVENT_MIN_GAP:
                res["opportunity"].append((d, float(v)))
                last_opp = d
    return res


# --- Confirmation-style BUY / SELL signals (V3.1) ---------------------------
# The user asked for signals with DURATION or SPEED confirmation instead of
# single-month spikes, grounded in historical behaviour:
#   SELL  = score stays ABOVE sell_level for >= sell_dur consecutive months
#           (sustained bubble zone — "连续超过一个区间多久")
#   BUY   = (a) score stays BELOW buy_level for >= buy_dur months (sustained
#           fear), OR (b) score collapses >= rapid_drop pts within
#           rapid_span months AND the trough is below rapid_end_level
#           ("从一个点短时间内快速下滑")
# A single continuous episode emits ONE signal at its CONFIRMATION month, so
# guidance is not spammed month after month. Dates are month-ends (no
# look-ahead: only past readings decide each signal).
SIGNAL_SELL_LEVEL = 78.0     # sustained above this -> de-risk confirmation
SIGNAL_SELL_DUR = 3          # consecutive months above -> sell signal
SIGNAL_BUY_LEVEL = 45.0      # sustained below this -> accumulate confirmation
SIGNAL_BUY_DUR = 2           # consecutive months below -> buy signal
SIGNAL_RAPID_DROP = 15.0     # |fall| >= this within the window -> rapid decline
SIGNAL_RAPID_SPAN = 3        # months
SIGNAL_RAPID_END = 60.0      # ...and the trough must be below this


def detect_signals(score: pd.Series) -> dict:
    """Confirmation-style buy/sell signals on the MONTHLY score series.

    Returns {"sell": [(date, score), ...], "buy": [(date, score, kind), ...]}
    where kind is "sustained" (extended fear) or "rapid" (fast collapse).
    Causal (a signal at month t uses readings <= t only) — safe to quote as
    "as of today" guidance, unlike detect_events which annotates history.
    """
    s = score.dropna()
    if len(s) < 6:
        return {"sell": [], "buy": []}
    vals = s.to_numpy(dtype=float)
    idx = s.index
    out_sell, out_buy = [], []

    # SELL: runs of score > SELL_LEVEL lasting >= SELL_DUR months
    run = 0
    for i, v in enumerate(vals):
        if v > SIGNAL_SELL_LEVEL:
            run += 1
            if run == SIGNAL_SELL_DUR:
                out_sell.append((idx[i], float(v)))
        else:
            run = 0

    # BUY (sustained): runs of score < BUY_LEVEL lasting >= BUY_DUR months
    run = 0
    for i, v in enumerate(vals):
        if v < SIGNAL_BUY_LEVEL:
            run += 1
            if run == SIGNAL_BUY_DUR:
                out_buy.append((idx[i], float(v), "sustained"))
        else:
            run = 0

    # BUY (rapid): |fall over <= RAPID_SPAN months| >= RAPID_DROP, trough < END
    if len(vals) >= SIGNAL_RAPID_SPAN:
        hi = pd.Series(vals, index=idx).rolling(SIGNAL_RAPID_SPAN).max()
        lo = pd.Series(vals, index=idx).rolling(SIGNAL_RAPID_SPAN).min()
        for i in range(SIGNAL_RAPID_SPAN - 1, len(vals)):
            d = idx[i]
            if hi.iloc[i] - lo.iloc[i] >= SIGNAL_RAPID_DROP and lo.iloc[i] < SIGNAL_RAPID_END:
                # trough date within the window
                win = vals[i - SIGNAL_RAPID_SPAN + 1: i + 1]
                tj = int(np.argmin(win))
                tdate = idx[i - SIGNAL_RAPID_SPAN + 1 + tj]
                out_buy.append((tdate, float(win[tj]), "rapid"))

    # ONE signal per episode: merge triggers closer than SIGNAL_MERGE_DAYS into
    # a single event (sell keeps the strongest peak, buy the deepest trough).
    out_sell = _merge_signals(out_sell, keep="max")
    out_buy = _merge_signals(out_buy, keep="min")
    return {"sell": out_sell, "buy": out_buy}


def _merge_signals(events, keep: str = "min", gap_days: int = 200):
    """Merge (date, score[, kind]) tuples closer than gap_days into one event,
    keeping the strongest reading ('min' = deepest trough, 'max' = highest
    peak). One continuous episode -> ONE signal, so guidance isn't spammed."""
    if not events:
        return events
    events = sorted(events, key=lambda x: x[0])
    merged = [list(events[0])]
    for e in events[1:]:
        if (e[0] - merged[-1][0]).days <= gap_days:
            if (keep == "min" and e[1] < merged[-1][1]) or \
               (keep == "max" and e[1] > merged[-1][1]):
                merged[-1] = list(e)
        else:
            merged.append(list(e))
    return [tuple(m) for m in merged]


def signal_stats(score: pd.Series, spx: Optional[pd.Series] = None) -> dict:
    """Historical grounding for the buy/sell signals: forward S&P returns after
    each signal vs the unconditional benchmark. Used by the guidance panel to
    show *why* the thresholds exist ("结合市场过往表现设定")."""

    def _fwd_ret(px: pd.Series, date, months: int) -> Optional[float]:
        try:
            a = px.asof(pd.Timestamp(date))
        except Exception:
            return None
        if a is None or pd.isna(a):
            return None
        end_dt = pd.Timestamp(date) + pd.DateOffset(months=months)
        tail = px[px.index <= end_dt]
        if tail.empty:
            return None
        b = float(tail.iloc[-1])
        return b / a - 1.0 if a else None

    px = spx
    if px is None:
        try:
            raw = _load_raw_cache()
            px = raw["spx"].dropna() if raw is not None and "spx" in raw else None
        except Exception:
            px = None
    base = {"sell": [], "buy_sustained": [], "buy_rapid": []}
    out = {k: {"count": 0, "fwd12": [], "fwd24": []} for k in base}
    if px is None or px.empty:
        out["_note"] = "SPX series unavailable for forward-return stats"
        return out
    sig = detect_signals(score)
    for d, _ in sig["sell"]:
        out["sell"]["count"] += 1
        for m in (12, 24):
            r = _fwd_ret(px, d, m)
            if r is not None:
                out["sell"][f"fwd{m}"].append(r)
    for d, _, kind in sig["buy"]:
        k = "buy_sustained" if kind == "sustained" else "buy_rapid"
        out[k]["count"] += 1
        for m in (12, 24):
            r = _fwd_ret(px, d, m)
            if r is not None:
                out[k][f"fwd{m}"].append(r)

    # unconditional benchmark: average forward returns across all months
    dates = list(score.dropna().index)
    bench = {12: [], 24: []}
    for d in dates[::3]:                       # sample every 3rd month (cheap)
        for m in (12, 24):
            r = _fwd_ret(px, d, m)
            if r is not None:
                bench[m].append(r)
    out["_benchmark"] = {
        f"fwd{m}": (float(np.mean(bench[m])) if bench[m] else None)
        for m in (12, 24)}

    def _sum(k):
        for m in (12, 24):
            v = out[k][f"fwd{m}"]
            out[k][f"fwd{m}"] = (round(float(np.mean(v)), 4) if v else None)
        out[k]["count"] = int(out[k]["count"])
    for k in ("sell", "buy_sustained", "buy_rapid"):
        _sum(k)
    return out


def get_price_series(ticker: str, start: str = LIVE_START,
                     refresh: bool = False) -> Optional[pd.Series]:
    """Monthly price series for the dashboard chart / backtest.

    Served from the unified raw cache (no extra network call); falls back to a
    direct fetch only if the ticker is absent from the cache.
    """
    key = RAW_TICKER_MAP.get(ticker, ticker)
    if not refresh and os.path.exists(CACHE_PATH):
        try:
            cached = pd.read_parquet(CACHE_PATH)
            if key in cached.columns:
                s = cached[key].dropna()
                if start:
                    s = s[s.index >= pd.Timestamp(start)]
                if not s.empty:
                    return s
        except Exception:
            pass
    s = _fetch_price(ticker, start=start)
    if s is None:
        return None
    return s.resample("ME").last()


def warm_cache() -> None:
    """Pre-compute and persist the feature cache (used at Docker BUILD time so
    FRED EMVMACROBUS and friends are baked into the image, and the first user
    request is instant). Failures are surfaced but never fatal.
    """
    try:
        s, m = get_monthly_scores(refresh=True)
        s = s.dropna()
        print(f"[warm] source={m.get('source')} live={m.get('available_count')}/8 "
              f"as_of={s.index[-1].date() if not s.empty else 'n/a'}")
    except Exception as exc:  # pragma: no cover
        print(f"[warm] pre-warm failed: {exc}")


if __name__ == "__main__":
    s, m = get_monthly_scores(refresh=True)
    s = s.dropna()
    print(f"Source : {m.get('source')}")
    print(f"Live   : {m.get('available_count')}/8 features")
    if m.get("features"):
        for k, v in m["features"].items():
            print(f"  {k:14s} {v}")
    print(f"As of  : {s.index[-1].date()}  Score = {s.iloc[-1]:.1f}  "
          f"({status_of(s.iloc[-1])})")
    print("Recent 12 months:")
    print(s.tail(12).round(1).to_string())
