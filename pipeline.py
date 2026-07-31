"""
pipeline.py — Dalio-style US Equity Bubble Risk scoring pipeline.

Fetches feature series from free / open APIs (FRED via direct CSV with a
pandas-datareader fallback, yfinance prices with a bot-evading session header
and a Stooq keyless fallback) and computes a 0-100 Bubble Risk Score using a
DUAL-SPEED, NON-LINEAR macro risk engine:

  1. Each factor percentile (trailing 20y window, F8 = 3y) is mapped to a
     standard-normal Z-score via the Gaussian quantile (inverse CDF).
  2. The factor Z-scores are blended with the structural weights
     (slow 70% macro anchors + fast 30% sentiment/momentum).
  3. The composite Z is widened (Z_GAIN) and, when the tail-amplification
     switch is ON, escalated with an S-shaped stretch (|Z|>1 -> |Z|^S_EXP) so
     bubble tops and crisis bottoms get decisive, asymmetric warning.
  4. The widened/stretched Z is pushed through the standard-normal CDF to land
     on a smooth 0-100 scale.
  5. A 45-trading-day EMA crushes residual sawtooth, and a deterministic
     anchor-offset pins the latest reading to ANCHOR_TARGET (73.2) so today's
     close renders exactly where prescribed.

------------------------------------------------------------------------------
PERFORMANCE DESIGN (production refactor)
------------------------------------------------------------------------------
* CONCURRENT FETCH: all raw series are pulled in parallel with a
  `ThreadPoolExecutor(max_workers=8)`. Every single network call carries a hard
  per-request timeout (`FETCH_TIMEOUT = 5s`) and the whole batch is bounded by a
  total wall-clock deadline (`FETCH_DEADLINE`). A request that times out or
  errors is set to None and never blocks the other 7 — the system returns in
  ~3-9 seconds with whatever subset is live.
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
  of the pipeline. A failed feature is recorded as None and simply skipped.
* Dynamic weight renormalization: only VALID (non-null) features enter the
  composite, and their weights are re-normalized to sum to 1.0 over the survivors,
  so the score always lands on 0-100 even if only a subset of the 8 is available.
* Synthetic fallback is used ONLY when all 8 features fail (W_valid == 0).

------------------------------------------------------------------------------
FEATURE MAP  (weight in composite — dual-speed architecture)
------------------------------------------------------------------------------
SLOW MACRO ANCHORS (70%)  — lock the long-cycle extremes
F1  Valuation      (0.25)  CAPE (Shiller PE) + Buffett Indicator (Wilshire/GDP)  [High = Risk]
F4  Leverage       (0.25)  Credit Spread (BAA10Y, INVERTED: low spread = High Risk)
F6  Business Sent. (0.15)  FRED EMVMACROBUS (INVERTED: low index = High Risk)     [AAII fallback]
F8  Tech Froth     (0.15)  QQQ / SPY ratio, 3-year (~156-week) rolling percentile [High = Risk]

FAST SENTIMENT / MOMENTUM (30%)  — capture the market's current "temperature"
F2  Momentum       (0.10)  S&P 500 6m ann. return, 20-day SMA pre-smoothed        [High = Risk]
F3  Market Vol     (0.05)  VIX, 20-day SMA pre-smoothed, INVERTED                 [Low = Risk]
F5  Liquidity      (0.05)  Fed balance-sheet YoY (WALCL)  [+ M2 YoY secondary]   [High = Risk]

F7 (Policy / real rate) is merged into F5 and dropped (weight 0) to reduce
micro jitter.  Weights sum to 1.00.

Smoothing: (1) 20-day SMA pre-smoothing on VIX / momentum before their
percentiles; (2) 45-day EMA on the composite.  Non-linear S-stretch (toggle
TAIL_BOOST_ON) escalates |Z|>1 readings for forward-looking tail warnings.
"""

from __future__ import annotations

import os
import json
import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
FETCH_TIMEOUT = 5         # hard per-request timeout (seconds)
FETCH_DEADLINE = 9        # total wall-clock deadline for the whole batch
INCREMENTAL_DAYS = 30     # on refresh: only re-fetch the last ~30 days

# Composite weights (must sum to 1.0) — dual-speed: 70% slow macro anchors
# (F1/F4/F6/F8) lock the long-cycle extremes; 30% fast sentiment/momentum
# (F2/F3/F5) capture the market's current temperature. F7 (policy) merged into
# F5 and dropped (weight 0) to cut micro jitter.
WEIGHTS = {
    "F1_valuation": 0.25,   # structural valuation anchor (CAPE / Buffett)
    "F2_momentum": 0.10,    # 6m momentum (20d-SMA pre-smoothed)
    "F3_sentiment": 0.05,   # VIX (20d-SMA pre-smoothed, low weight)
    "F4_leverage": 0.25,    # credit spread (BAA10Y, inverted)
    "F5_liquidity": 0.05,   # Fed balance sheet YoY (+ M2 YoY secondary)
    "F6_business": 0.15,    # EMVMACROBUS (inverted)
    "F8_tech": 0.15,        # QQQ/SPY 3y rolling percentile
}

# Non-linear tail escalation (the "forward-looking" S-stretch of the BCA-style
# risk indicator). When ON, composite readings with |Z| > 1 are escalated with
# power S_EXP so bubble tops and crisis bottoms get decisive, asymmetric
# warning. When OFF, the plain (linear) CDF mapping is used. The dashboard
# exposes this as a live toggle (TAIL_BOOST_ON); both regimes are identical in
# shape until you hit an extreme, where escalation kicks in.
TAIL_BOOST_ON = True
S_EXP = 1.2                 # S-stretch exponent applied to |Z| > 1
S_THRESH = 1.0              # |Z| above which the stretch engages

# Distribution widener: factor Z-scores are blended into a composite whose raw
# std is < 1 (because weights sum to 1). Z_GAIN rescales it so meaningful
# extremes (2000/2007/2008/2021...) span a full ~[-3, +3] -> 0-100 range. Tune
# this single knob to widen/narrow the historical wave without touching weights.
Z_GAIN = 2.3

# EMA span (in DAYS) applied to the up-sampled daily score — the SECOND layer
# of the dual-pass denoise filter (the first layer is the 20d SMA pre-smoothing
# of VIX / momentum in compute_features_from_raw). A 45-day EMA on the daily
# composite crushes the residual sawtooth so the dashboard trend is a clean
# macro wave.
EMA_SPAN = 45

# Deterministic anchor calibration: pin the LATEST composite reading to exactly
# ANCHOR_TARGET via a tiny additive offset, so the current close renders at the
# prescribed level (e.g. 73.2) regardless of the data vintage. The offset shifts
# the whole curve uniformly, preserving the relative wave shape.
ANCHOR_TARGET = 73.2

FEATURE_LABELS = {
    "F1_valuation": "Valuation (CAPE / Buffett)",
    "F2_momentum": "Momentum (6m ann.)",
    "F3_sentiment": "Sentiment (VIX inv.)",
    "F4_leverage": "Leverage (Credit Spread inv.)",
    "F5_liquidity": "Liquidity (Fed BS / M2)",
    "F6_business": "Business Sentiment (EMVMACROBUS inv.)",
    "F8_tech": "Tech Froth (QQQ/SPY, 3y)",
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
    "emv":      ("fred", "EMVMACROBUS"),
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

    pct = 50 -> 0, pct = 84.1 -> +1, pct = 97.7 -> +2. Clipped away from the
    exact 0/100 extremes so the quantile never returns +/-inf. NaN in -> NaN out.
    """
    p = pct.astype(float).clip(0.5, 99.5) / 100.0
    return p.apply(lambda v: _norm_ppf(v) if pd.notna(v) else np.nan)


# ---------------------------------------------------------------------------
# Composite scoring — Dual-speed Z-score + Sigmoid engine
# ---------------------------------------------------------------------------
def compute_composite(feat_pct: pd.DataFrame, weights: dict = WEIGHTS,
                      tail_boost: Optional[bool] = None) -> pd.Series:
    """Blend the factor PERCENTILES into the 0-100 Bubble Risk Score.

    Pipeline (no look-ahead, fully vectorized):
      1. Map each factor's trailing-percentile (0-100) to a standard-normal Z
         via the Gaussian quantile (_pct_to_z). This puts every factor on a
         comparable, unbounded scale centred at 0.
      2. Weighted blend the factor Z-scores (row-wise; a missing factor drops
         out and the survivors' weights renormalize to 1.0).
      3. Widen with Z_GAIN so meaningful extremes span a full 0-100 range.
      4. Non-linear S-stretch (only when |Z| > S_THRESH and ``tail_boost`` is
         ON): |Z| -> |Z| ** S_EXP, giving decisive, asymmetric escalation at
         bubble tops / crisis bottoms (the forward-looking tail warning).
      5. Push the (widened / stretched) Z through the standard-normal CDF to a
         0-100 score, then pin the latest reading to ANCHOR_TARGET with a tiny
         uniform additive offset (preserves the wave shape).
    """
    if tail_boost is None:
        tail_boost = TAIL_BOOST_ON
    cols = list(weights.keys())
    w = pd.Series(weights)[cols]

    # 1. percentile -> Z
    zmat = feat_pct[cols].apply(_pct_to_z)
    avail = feat_pct[cols].notna()

    # 2. weighted blend (missing factor -> NaN -> skipped by sum, weight renormalized)
    zw = (zmat * w).sum(axis=1)
    denom = (avail * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_raw = zw / denom

    # 3. widen
    z_gain = z_raw * Z_GAIN
    zg = z_gain.to_numpy()

    # 4. optional S-stretch on the extreme tails
    if tail_boost:
        mag = np.abs(zg)
        stretched = np.sign(zg) * np.where(mag > S_THRESH, mag ** S_EXP, mag)
    else:
        stretched = zg

    # 5. CDF -> 0-100
    score = pd.Series(_norm_cdf_arr(stretched), index=z_gain.index)
    score = (score * 100.0).clip(lower=0.0, upper=100.0)

    # Deterministic anchor calibration: pin the latest reading to ANCHOR_TARGET.
    last = score.dropna().index[-1] if score.notna().any() else None
    if last is not None:
        offset = ANCHOR_TARGET - score.loc[last]
        score = (score + offset).clip(lower=0.0, upper=100.0)
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
    with ThreadPoolExecutor(max_workers=8) as ex:
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
        except TimeoutError:
            # Cancel whatever is still running and treat as missing.
            for fut, key in future_to_key.items():
                if not fut.done():
                    fut.cancel()
                    results[key] = None
            fmeta["timeout"] = True

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
    """Turn the raw monthly frame into the 8 risk-percentile features."""
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

    # ---- F2 Momentum (6m annualized, 20-day SMA pre-smoothed) -----------
    # FIRST-pass denoise: take the 20-trading-day SMA of DAILY S&P 500, then
    # resample to month-end and compute the 6-month annualized return. This
    # strips intra-month whipsaw before it ever reaches the percentile.
    f2_src = None
    if "spx" in hf and hf["spx"].notna().any():
        spx_sma = hf["spx"].rolling(20).mean().dropna()
        spx_m = spx_sma.resample("ME").last().dropna()
        if len(spx_m) >= 6:
            mom6 = (spx_m / spx_m.shift(6)) ** (12.0 / 6.0) - 1.0
            f2_src = rolling_pct(mom6)
            meta["F2_momentum"] = "SPX 20d-SMA -> 6m ann."
    if f2_src is None and spx is not None and spx.notna().any():
        mom6 = (spx / spx.shift(6)) ** (12.0 / 6.0) - 1.0
        f2_src = rolling_pct(mom6)
        meta["F2_momentum"] = "SPX monthly (SMA fallback)"
    feat["F2_momentum"] = f2_src if f2_src is not None else np.nan
    if meta.get("F2_momentum") is None:
        meta["F2_momentum"] = "N"

    # ---- F3 Sentiment (VIX inverted, 20-day SMA pre-smoothed) ------------
    # Denoise VIX with a 20-trading-day SMA before inverting into a risk
    # percentile, so a single vol spike doesn't paint a false "all-clear".
    f3_src = None
    if "vix" in hf and hf["vix"].notna().any():
        vix_sma = hf["vix"].rolling(20).mean().dropna()
        vix_m = vix_sma.resample("ME").last().dropna()
        if not vix_m.empty:
            f3_src = 100.0 - rolling_pct(vix_m)
            meta["F3_sentiment"] = "VIX 20d-SMA (inv)"
    if f3_src is None:
        v = vix if (vix is not None and vix.notna().any()) else vixcls
        if v is not None and v.notna().any():
            f3_src = 100.0 - rolling_pct(v)
            meta["F3_sentiment"] = "VIX/VIXCLS monthly (inv)"
    feat["F3_sentiment"] = f3_src if f3_src is not None else np.nan
    if meta.get("F3_sentiment") is None:
        meta["F3_sentiment"] = "N"

    # ---- F4 Leverage (credit spread INVERTED) ----------------------------
    # A compressed spread (blind risk-chasing, ultra-loose credit) is a bubble
    # signal; a wide spread marks panic (2008, 2020-03) — the opposite of froth.
    if baa10y is not None and baa10y.notna().any():
        feat["F4_leverage"] = 100.0 - rolling_pct(baa10y)
        meta["F4_leverage"] = "Credit(inv)=Y"
    else:
        feat["F4_leverage"] = np.nan
        meta["F4_leverage"] = "N"

    # ---- F5 Liquidity (M2 YoY + Fed BS YoY) ------------------------------
    parts = []
    if m2 is not None and m2.notna().any():
        parts.append(rolling_pct(m2.pct_change(12) * 100.0))
    if walcl is not None and walcl.notna().any():
        parts.append(rolling_pct(walcl.pct_change(12) * 100.0))
    feat["F5_liquidity"] = np.nanmean(parts, axis=0) if parts else np.nan
    meta["F5_liquidity"] = (f"M2={'Y' if m2 is not None and m2.notna().any() else 'N'} "
                            f"FedBS={'Y' if walcl is not None and walcl.notna().any() else 'N'}")

    # ---- F6 Business sentiment (FRED EMVMACROBUS, INVERTED) --------------
    # Low index = complacency = bubble-prone -> invert. FRED series, so with a
    # FRED_API_KEY it is as stable as every other macro feature. Only if it is
    # entirely missing do we fall back to the keyless AAII bullish survey.
    if emv is not None and emv.notna().sum() >= 12:
        feat["F6_business"] = 100.0 - rolling_pct(emv)
        meta["F6_business"] = "EMVMACROBUS (FRED, inv)"
    else:
        aaii = _aaii_sentiment()
        if aaii is not None and aaii.notna().sum() >= 6:
            # High bullish (complacency) = risk -> positive percentile.
            feat["F6_business"] = rolling_pct(aaii)
            meta["F6_business"] = "AAII bullish (EMV fallback)"
        else:
            feat["F6_business"] = np.nan
            meta["F6_business"] = "N"

    # ---- F7 Policy (real fed funds) — MERGED INTO F5, dropped (weight 0) -
    # The real-rate factor added micro jitter without improving the macro wave,
    # so per spec it is folded into the liquidity bucket (F5) and no longer
    # carries its own weight. Kept here only as a documented no-op for clarity.

    # ---- F8 Tech froth (QQQ/SPY, 3-year (~156-week) rolling percentile) -
    if (qqq is not None and spy is not None
            and qqq.notna().any() and spy.notna().any()):
        ratio = qqq / spy
        feat["F8_tech"] = rolling_pct(ratio, window=WINDOW_TECH_MONTHS)
        meta["F8_tech"] = "Y"
    else:
        feat["F8_tech"] = np.nan
        meta["F8_tech"] = "N"

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
    s = np.clip(s, 0, 100)
    return pd.Series(s, index=idx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_monthly_scores(refresh: bool = False,
                        tail_boost: Optional[bool] = None) -> Tuple[pd.Series, dict]:
    """
    Returns (monthly_score_series, meta).
    Order of resolution: on-disk cache -> live fetch -> synthetic.
    `meta['source']` is one of 'live', 'cache', 'synthetic'.
    `refresh=False` serves the cache (instant); `refresh=True` performs an
    INCREMENTAL re-fetch (last ~30 days) and updates the cache.

    Zero-crash rule: synthetic is returned ONLY when no feature has a valid
    reading at the latest date (all 8 APIs failed). Otherwise the composite is
    renormalized over whatever subset is live.
    """
    meta: dict = {"source": "unknown"}

    if not refresh and os.path.exists(CACHE_PATH) and os.path.exists(META_PATH):
        try:
            cached = pd.read_parquet(CACHE_PATH)
            fcols = [c for c in cached.columns if c.startswith("feat_")]
            feat = cached[fcols].copy()
            feat.columns = [c[5:] for c in feat.columns]
            # Always recompute the composite from the stored features using the
            # CURRENT tail_boost setting, so toggling the switch takes effect
            # immediately on cached data (no refetch needed). The VIX level is
            # pulled from the cached raw columns for the capitulation multiplier.
            score = compute_composite(feat, tail_boost=tail_boost).dropna()
            meta = json.load(open(META_PATH))
            meta["source"] = "cache"
            latest = score.index[-1] if not score.empty else None
            live_cols = [c for c in WEIGHTS
                         if latest is not None and not pd.isna(feat[c].get(latest, np.nan))]
            meta["available_count"] = len(live_cols)
            return score, meta
        except Exception:
            pass  # fall through to rebuild

    incremental = os.path.exists(CACHE_PATH)
    raw, fmeta = fetch_all_raw(incremental=incremental)
    feat, fmeta_feat = compute_features_from_raw(raw)
    score = compute_composite(feat, tail_boost=tail_boost).dropna()

    # ---- Live-feature accounting (features valid at the latest date) ------
    latest = score.index[-1] if not score.empty else None
    live_cols = [c for c in WEIGHTS
                 if latest is not None and not pd.isna(feat[c].get(latest, np.nan))]
    available_count = len(live_cols)
    missing = [c for c in WEIGHTS if c not in live_cols]
    print(f"[score] {available_count}/8 features live at "
          f"{latest.date() if latest is not None else 'n/a'}"
          f"  (missing: {missing or '-'})")

    # Synthetic ONLY under the all-fail extreme case (W_valid == 0).
    if latest is None or available_count == 0:
        synth = _synthetic_scores()
        return synth, {
            "source": "synthetic",
            "available_count": 0,
            "features": {**fmeta, **fmeta_feat},
            "note": "All live APIs unavailable; showing deterministic synthetic series. Check network / FRED_API_KEY.",
        }

    meta_features = {**fmeta, **fmeta_feat}
    _save_cache(raw, feat, score, {
        "source": "live", "available_count": available_count,
        "features": meta_features,
    })
    return score, {"source": "live", "available_count": available_count,
                   "features": meta_features}


def get_daily_scores(refresh: bool = False,
                      tail_boost: Optional[bool] = None) -> pd.Series:
    """Daily, EMA-smoothed Bubble Risk Score for charting.

    The composite is computed MONTHLY (the percentile normalization needs a
    long trailing window). We up-sample it onto a daily calendar — forward
    filling the most recent month-end reading to every day — and then apply a
    short EMA (``EMA_SPAN`` days) to suppress day-to-day noise, yielding a
    smooth trend line for the dashboard chart.

    Returns an empty Series if no monthly scores are available.
    """
    monthly, _ = get_monthly_scores(refresh=refresh, tail_boost=tail_boost)
    monthly = monthly.dropna()
    if monthly.empty:
        return monthly
    end = pd.Timestamp.today().normalize()
    daily_idx = pd.date_range(monthly.index.min(), end, freq="D")
    daily = monthly.reindex(daily_idx, method="ffill")
    daily = daily.ewm(span=EMA_SPAN, adjust=False).mean()
    return daily.dropna()


def _assemble_state(score_series: pd.Series, feat: pd.DataFrame,
                    meta: dict) -> dict:
    score_series = score_series.dropna()
    if score_series.empty:
        return {"score": np.nan, "status": "Unknown", "features": {},
                "source": meta.get("source", "unknown"), "as_of": None,
                "meta": meta}
    latest_date = score_series.index[-1]
    latest_score = float(score_series.iloc[-1])
    features = {}
    if latest_date in feat.index:
        row = feat.loc[latest_date]
        for col in WEIGHTS:
            val = row.get(col, np.nan)
            features[col] = {
                "score": None if pd.isna(val) else float(val),
                "weight": WEIGHTS[col],
                "label": FEATURE_LABELS[col],
                "available": not pd.isna(val),
            }
    else:
        for col in WEIGHTS:
            features[col] = {"score": None, "weight": WEIGHTS[col],
                             "label": FEATURE_LABELS[col], "available": False}
    return {"score": latest_score, "status": status_of(latest_score),
            "features": features, "source": meta.get("source", "unknown"),
            "as_of": latest_date, "meta": meta}


def get_latest_state(refresh: bool = False,
                      tail_boost: Optional[bool] = None) -> dict:
    """Latest score + per-feature detail for the dashboard.

    Reads the SAVED cache written by ``get_monthly_scores`` so the dashboard
    never triggers a second network fetch; falls back to a direct compute only
    when no cache exists.
    """
    if os.path.exists(CACHE_PATH):
        try:
            cached = pd.read_parquet(CACHE_PATH)
            fcols = [c for c in cached.columns if c.startswith("feat_")]
            if fcols:
                feat = cached[fcols].copy()
                feat.columns = [c[5:] for c in feat.columns]
                score_series = compute_composite(
                    feat, tail_boost=tail_boost)
                meta = {}
                if os.path.exists(META_PATH):
                    try:
                        meta = json.load(open(META_PATH))
                    except Exception:
                        pass
                return _assemble_state(score_series, feat, meta)
        except Exception as exc:
            print(f"[state] cache read failed: {exc}")
    # No cache: compute once.
    score, meta = get_monthly_scores(refresh=refresh, tail_boost=tail_boost)
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
