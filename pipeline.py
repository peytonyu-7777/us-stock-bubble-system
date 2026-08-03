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

WINDOW_MONTHS = 240       # 20 years for the "trailing percentile" window
WINDOW_TECH_MONTHS = 36   # 3-year (~156-week) window for the tech-froth feature (F8)

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

# --- Stability layer (principle 1) -----------------------------------------
# --- K-line style two-timescale stability filter ---------------------------
# A slow EMA carries the mid/long-term macro trend; a damped fast component
# adds a SMALL, bounded short-term oscillation on top — like a stock K-line:
# a clear medium-term trend with minor daily wiggle. Never a violent sawtooth,
# never an over-smoothed flat line.
TREND_SPAN = 75              # slow EMA on the daily series (≈ one quarter)
OSC_SPAN = 8                 # fast EMA -> short-term component
OSC_GAIN = 0.55              # fraction of (fast - trend) kept as visible wiggle
OSC_MAX = 6.0                # hard cap on the short-term oscillation (points)
DAILY_CLAMP = 1.2            # max |Δ| per day in normal regimes
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

# Legacy toggle kept for API compatibility (controls valuation acceleration on/off)
TAIL_BOOST_ON = True

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
              timeout: int = FETCH_TIMEOUT) -> Optional[pd.Series]:
    """Fetch a FRED series as a monthly-end Series via the direct CSV endpoint.

    Uses the authenticated `api.stlouisfed.org` endpoint when FRED_API_KEY is
    set (reliable + server-side date filtering) and the keyless
    `fredgraph.csv` endpoint otherwise. Both honour `timeout` natively.
    Falls back to pandas_datareader (bounded by _run_with_timeout) only if the
    CSV endpoints fail.
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
        s = _parse_fred_csv(txt, series_id)
        if s is not None and not s.empty:
            return s

    # ---- pandas_datareader fallback (bounded so it can't hang the batch) ---
    return _run_with_timeout(
        lambda: _fred_pdr(series_id, start), timeout=FETCH_TIMEOUT)


def _parse_fred_csv(txt: str, series_id: str) -> Optional[pd.Series]:
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
        s = s.resample("ME").last()          # monthly month-end
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
    """Monthly-close via yfinance (timeout + bot-evading session), Stooq fallback.

    Yahoo frequently 429s from cloud IPs; Stooq is keyless and server-friendly,
    so prices stay REAL on deploy when Yahoo is blocked.
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
                return s.resample("ME").last()
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[yfinance] {ticker} failed: {exc}")
    st = _STOOQ_MAP.get(ticker)
    if st:
        return _stooq_daily(st, start=start)
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
        return _stooq_daily(st, start=start)
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
    """Return a dict of daily Series (keys: 'vix', 'spx') used for the first
    smoothing layer, with an incremental parquet cache so refreshes only pull
    the last ~30 days. Non-fatal: missing keys simply fall back to the monthly
    path inside compute_features_from_raw.

    Network is bounded by FETCH_TIMEOUT + the global deadline; any failure
    returns whatever subset is available (possibly empty).
    """
    cached = _load_hf_cache()
    df = cached if cached is not None else pd.DataFrame()
    out: dict = {}
    for tag, ticker in (("vix", "^VIX"), ("spx", "^GSPC")):
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


def compute_modules(feat_pct: pd.DataFrame,
                    tail_boost: Optional[bool] = None) -> pd.DataFrame:
    """Aggregate the granular percentile factors into the 5 V2 risk modules.

    Each module is the mean of its available sub-indicator percentiles. The
    valuation module runs its inputs through ``valuation_curve`` (non-linear
    froth acceleration) when ``tail_boost`` is ON; when OFF it uses the plain
    percentile so the dashboard can show the "raw" valuation contribution.

    A module with NO available sub-indicator is filled with neutral 50 (so a
    single missing series can never swing the blend), and coverage is recorded
    for the global gate.
    """
    if tail_boost is None:
        tail_boost = TAIL_BOOST_ON
    out = pd.DataFrame(index=feat_pct.index)
    coverage = {}
    for mod, cols in MODULE_SUBINDICATORS.items():
        present = [c for c in cols if c in feat_pct.columns
                   and feat_pct[c].notna().any()]
        coverage[mod] = (len(present) / len(cols)) if cols else 0.0
        if not present:
            out[mod] = 50.0
            continue
        sub = feat_pct[present]
        if mod == "valuation" and tail_boost:
            # apply the acceleration curve element-wise (vectorized)
            vals = sub.applymap(lambda v: valuation_curve(v)
                                if pd.notna(v) else np.nan)
            out[mod] = vals.mean(axis=1)
        else:
            out[mod] = sub.mean(axis=1)
    out.attrs["coverage"] = coverage
    return out


def historical_calibrate(raw: pd.Series) -> pd.Series:
    """Affine-calibrate the raw composite to the historical bubble scale.

    Pin the MAX raw reading inside ``HIST_PEAK_WINDOW`` (the dot-com episode)
    to ``HIST_PEAK_TARGET`` (97) and the MIN raw reading inside
    ``HIST_TROUGH_WINDOW`` (the GFC episode) to ``HIST_TROUGH_TARGET`` (12).
    Linear interpolation everywhere else preserves the relative macro wave and
    guarantees the index lands in a realistic [~12, ~97] band with today
    falling wherever the data puts it (no hard-coded "today" pin).
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
    """V2 Bubble Risk Score (0-100) from the granular percentile factors.

    Pipeline (no look-ahead, fully vectorized):
      1. Aggregate the 8 granular percentile factors into 5 risk MODULES.
      2. Weighted blend the modules (MODULE_WEIGHTS); a module below the
         coverage gate is neutralised (filled 50) so it can't distort.
      3. Historical affine calibration -> realistic [12, 97] bubble scale.
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
    blended = blended.where(total_cov >= MIN_VALID_WEIGHT, np.nan)

    score = historical_calibrate(blended)
    return score


def contribution_factor(score: float) -> float:
    """Monthly DCA multiplier from the Bubble Risk Score (per spec)."""
    if pd.isna(score):
        return 1.0
    if score < 40:
        return 2.0
    if score < 60:
        return 1.5
    if score < 80:
        return 1.0
    if score < 90:
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
    """Turn the raw monthly frame into the 8 risk-percentile features.

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
        parts.append(rolling_pct(cape))
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
        parts.append(rolling_pct(buffett.dropna()))

    primary = np.nanmean(parts, axis=0) if parts else None

    # Zero-fail fallback proxy: S&P 500 distance above its ~200-week MA.
    # A fat premium = expensive market = high risk (positive percentile).
    ma_pct = None
    if spx is not None and spx.notna().sum() >= 60:
        ma_long = spx.rolling(46).mean()        # 46 months ~ 200 weeks
        prem = (spx / ma_long - 1.0) * 100.0
        ma_pct = rolling_pct(prem.dropna())

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
            feat["F1_valuation"] = 100.0 - rolling_pct(sp500div)
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
        z = (log_cape - mu) / sd.replace(0, np.nan)
        cape_z = rolling_pct(z.clip(-4, 4))
        meta["F1b_cape_z"] = "CAPE z vs 10y (pct)"
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
            pct_ratio = rolling_pct(mg_ratio)
        else:
            pct_ratio = None
        pct_yoy = rolling_pct(mg_yoy)
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
        credit_ease = 100.0 - rolling_pct(baa10y)   # loose credit = high ease
        r1 = rolling_pct(spx_ret12)
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
            f6_src = rolling_pct(mom6)
            meta["F6_momentum"] = "SPX 10d-SMA -> 6m ann."
    if f6_src is None and spx is not None and spx.notna().any():
        mom6 = (spx / spx.shift(6)) ** (12.0 / 6.0) - 1.0
        f6_src = rolling_pct(mom6)
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
            f7_src = 100.0 - rolling_pct(vix_m)
            meta["F7_volatility"] = "VIX 10d-SMA (inv)"
    if f7_src is None:
        v = vix if (vix is not None and vix.notna().any()) else vixcls
        if v is not None and v.notna().any():
            f7_src = 100.0 - rolling_pct(v)
            meta["F7_volatility"] = "VIX/VIXCLS monthly (inv)"
    feat["F7_volatility"] = f7_src if f7_src is not None else np.nan
    if meta.get("F7_volatility") is None:
        meta["F7_volatility"] = "N"

    # ---- F3 Credit (credit spread INVERTED) -----------------------------
    # A compressed spread (blind risk-chasing, ultra-loose credit) is a bubble
    # signal; a wide spread marks panic (2008, 2020-03) — the opposite of froth.
    if baa10y is not None and baa10y.notna().any():
        feat["F3_credit"] = 100.0 - rolling_pct(baa10y)
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
        real_rate = rolling_pct(rr)
        meta["F3b_realrate"] = "FedFunds-CPI (pct)"
    if real_rate is None:
        feat["F3b_realrate"] = np.nan
        meta["F3b_realrate"] = "N"
    else:
        feat["F3b_realrate"] = real_rate          # already inverted (high = tight)

    # ---- F3c Yield Curve (10Y-3M spread, INVERTED) -----------------------
    # An inverted / flat curve is a classic late-cycle risk signal.
    yc = None
    if dgs10 is not None and dgs3mo is not None and dgs10.notna().any() and dgs3mo.notna().any():
        spread = (dgs10 - dgs3mo).replace([np.inf, -np.inf], np.nan)
        yc = rolling_pct(spread)
        meta["F3c_yield"] = "10Y-3M (pct)"
    if yc is None:
        feat["F3c_yield"] = np.nan
        meta["F3c_yield"] = "N"
    else:
        feat["F3c_yield"] = yc                      # already inverted (high = flat/inverted)

    # ---- F8 Liquidity (M2 YoY + Fed BS YoY) ------------------------------
    parts = []
    if m2 is not None and m2.notna().any():
        parts.append(rolling_pct(m2.pct_change(12) * 100.0))
    if walcl is not None and walcl.notna().any():
        parts.append(rolling_pct(walcl.pct_change(12) * 100.0))
    feat["F8_liquidity"] = np.nanmean(parts, axis=0) if parts else np.nan
    meta["F8_liquidity"] = (f"M2={'Y' if m2 is not None and m2.notna().any() else 'N'} "
                            f"FedBS={'Y' if walcl is not None and walcl.notna().any() else 'N'}")

    # ---- F4 Business sentiment (FRED EMVMACROBUS, INVERTED) --------------
    # Low index = complacency = bubble-prone -> invert. FRED series, so with a
    # FRED_API_KEY it is as stable as every other macro feature. Only if it is
    # entirely missing do we fall back to the keyless AAII bullish survey.
    if emv is not None and emv.notna().sum() >= 12:
        feat["F4_business"] = 100.0 - rolling_pct(emv)
        meta["F4_business"] = "EMVMACROBUS (FRED, inv)"
    else:
        aaii = _aaii_sentiment()
        if aaii is not None and aaii.notna().sum() >= 6:
            # High bullish (complacency) = risk -> positive percentile.
            feat["F4_business"] = rolling_pct(aaii)
            meta["F4_business"] = "AAII bullish (EMV fallback)"
        else:
            feat["F4_business"] = np.nan
            meta["F4_business"] = "N"

    # ---- F5 Tech froth (QQQ/SPY, 3-year (~156-week) rolling percentile) -
    if (qqq is not None and spy is not None
            and qqq.notna().any() and spy.notna().any()):
        ratio = qqq / spy
        feat["F5_tech"] = rolling_pct(ratio, window=WINDOW_TECH_MONTHS)
        meta["F5_tech"] = "Y"
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
        with open(META_PATH, "w") as f:
            json.dump(meta, f, default=str)
    except Exception:
        pass


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
    # 1) Instant path: serve the baked/on-disk cache without touching network.
    if not refresh:
        hit = _cached_scores(tail_boost)
        if hit is not None:
            return hit

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
    end = pd.Timestamp.today().normalize()
    daily_idx = pd.date_range(monthly.index.min(), end, freq="D")
    daily = monthly.reindex(daily_idx, method="ffill")

    # load the daily vol-regime signals for the stress-aware clamp
    hf = _get_hf_daily()
    vix_d = hf.get("vix") if hf else None
    spx_d = hf.get("spx") if hf else None
    # monthly BAA10Y for the credit-jump stress test
    baa = None
    if os.path.exists(CACHE_PATH):
        try:
            baa = pd.read_parquet(CACHE_PATH)["baa10y"]
        except Exception:
            baa = None
    daily = stability_filter(daily, vix=vix_d, spx=spx_d, baa=baa)
    return daily.dropna()


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

    # granular factor detail (for the expandable breakdown)
    features = {}
    nonempty_feat = feat if feat is not None and not feat.empty else None
    for col in WEIGHTS:
        if nonempty_feat is not None and latest_date in nonempty_feat.index:
            val = nonempty_feat.loc[latest_date].get(col, np.nan)
        else:
            val = np.nan
        features[col] = {
            "score": None if pd.isna(val) else float(val),
            "weight": WEIGHTS[col],
            "label": FEATURE_LABELS.get(col, col),
            "available": not pd.isna(val),
        }

    # ---- 5 module scores at the latest date -------------------------------
    modules = {}
    if nonempty_feat is not None:
        mod_df = compute_modules(nonempty_feat, tail_boost=meta.get("_tb"))
        for m in MODULE_WEIGHTS:
            v = mod_df[m].get(latest_date, np.nan)
            modules[m] = None if pd.isna(v) else float(v)

    # ---- historical percentile of the current reading --------------------
    hist_pct = float((score_series <= latest_score).mean() * 100.0)

    # ---- month-over-month drivers (which modules moved the score) ---------
    drivers = []
    if nonempty_feat is not None and len(score_series) >= 2:
        prev_date = score_series.index[-2]
        prev_mod = compute_modules(nonempty_feat, tail_boost=meta.get("_tb"))
        for m in MODULE_WEIGHTS:
            cur = mod_df[m].get(latest_date, np.nan)
            prv = prev_mod[m].get(prev_date, np.nan)
            if pd.notna(cur) and pd.notna(prv):
                drivers.append({"module": m, "delta": float(cur - prv),
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
            cached = pd.read_parquet(CACHE_PATH)
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
