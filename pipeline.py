"""
pipeline.py — Dalio-style US Equity Bubble Risk scoring pipeline.

Fetches 8 feature series from free / open APIs (FRED via pandas_datareader,
yfinance prices, pytrends Google Trends) and computes a 0-100 Bubble Risk
Score with rolling 20-year (default) percentile normalization.

------------------------------------------------------------------------------
DESIGN PRINCIPLES
------------------------------------------------------------------------------
* Every feature is normalized so that HIGHER value == MORE bubble risk.
  For "good when low" indicators (VIX, real rate) we invert inside the
  feature builder, so the percentile is always "risk percentile".
* Each feature becomes a percentile rank (0-100) within a TRAILING window
  (default 20 years). Because the window only looks backwards, the score
  carries NO look-ahead bias and can be re-computed for any historical date.
* The composite score is a weighted blend of the 8 per-feature percentiles.
  If a feature is unavailable (API failure / no history) its weight is
  redistributed across the remaining features, so the score stays on 0-100.
* A disk cache (parquet) avoids hammering APIs on every run. A deterministic
  synthetic generator guarantees the app still renders when every API fails
  (clearly flagged as "synthetic" in the UI).

------------------------------------------------------------------------------
FEATURE MAP  (weight in composite)
------------------------------------------------------------------------------
F1  Valuation      (0.20)  CAPE (Shiller PE) + Buffett Indicator (Wilshire/GDP)   [High = Risk]
F2  Momentum       (0.10)  S&P 500 6-month annualized return                      [High = Risk]
F3  Market Vol     (0.10)  VIX (INVERTED: low VIX = complacency = High Risk)
F4  Leverage       (0.15)  Margin Debt / Mkt Cap  +  Credit Spread (BAA10Y, INVERTED: low spread = High Risk)
F5  Liquidity      (0.10)  Fed balance-sheet YoY (WALCL)  [+ M2 YoY secondary]    [High = Risk]
F6  Business Sent. (0.15)  FRED EMVMACROBUS (INVERTED: low index = High Risk)      [AAII/FINRA fallback]
F7  Policy Stance  (0.05)  Real Fed Funds (FEDFUNDS - CPI YoY, INVERTED: low real rate = High Risk)
F8  Tech Froth     (0.15)  QQQ / SPY ratio, 3-year (156-week) rolling percentile  [High = Risk]

Weights sum to 1.00. Any feature that fails to load has its weight redistributed
across the survivors (zero-crash renormalization), so the score always lands on
0-100 even if only a subset of the 8 is available.
"""

from __future__ import annotations

import os
try:
    from dotenv import load_dotenv
    load_dotenv()   # pull FRED_API_KEY (and friends) from a local .env file
except Exception:
    pass
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from io import StringIO

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # optional; many series work w/o key
CACHE_PATH = os.getenv("BUBBLE_CACHE", "bubble_cache.parquet")
PRICES_CACHE = os.getenv("BUBBLE_PRICES_CACHE", "prices_cache.parquet")
WINDOW_MONTHS = 240       # 20 years for the "trailing percentile" window
WINDOW_TECH_WEEKS = 156    # 3 years (156 weeks) for the tech-froth feature (F8)
WINDOW_TECH_MONTHS = 36    # 3-year monthly fallback if weekly prices fail

# Composite weights (must sum to 1.0)
WEIGHTS = {
    "F1_valuation": 0.20,
    "F2_momentum": 0.10,
    "F3_sentiment": 0.10,
    "F4_leverage": 0.15,
    "F5_liquidity": 0.10,
    "F6_business": 0.15,
    "F7_policy": 0.05,
    "F8_tech": 0.15,
}

FEATURE_LABELS = {
    "F1_valuation": "Valuation (CAPE / Buffett)",
    "F2_momentum": "Momentum (6m ann.)",
    "F3_sentiment": "Sentiment (VIX inv.)",
    "F4_leverage": "Leverage (Margin / Credit)",
    "F5_liquidity": "Liquidity (Fed BS / M2)",
    "F6_business": "Business Sentiment (EMVMACROBUS inv.)",
    "F7_policy": "Policy (Real Fed Funds)",
    "F8_tech": "Tech Froth (QQQ/SPY)",
}

HISTORY_START = "1990-01-01"   # long history so the 20y window is "full" by 2010
LIVE_START = "2000-01-01"      # backtest / reporting start


# ---------------------------------------------------------------------------
# Low-level fetchers
# ---------------------------------------------------------------------------
def _fred(series_id: str, start: str = HISTORY_START) -> Optional[pd.Series]:
    """Fetch a FRED series as a monthly-end Series. Returns None on failure."""
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return None
    try:
        df = pdr.get_data_fred(series_id, start, api_key=FRED_API_KEY or None)
        if df is None or df.empty:
            return None
        s = df[df.columns[0]].dropna()
        s = s.resample("ME").last()
        return s
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[fred] {series_id} failed: {exc}")
        return None


_STOOQ_MAP = {"^GSPC": "SPX.US", "SPY": "SPY.US", "QQQ": "QQQ.US", "^IXIC": "IXIC.US"}


def _http_get(url: str, timeout: int = 20) -> Optional[str]:
    """Keyless HTTP GET with a browser UA (Stooq / AAII block empty UAs)."""
    try:
        hdr = {"User-Agent": "Mozilla/5.0 (compatible; bubble-monitor/1.0)"}
        r = requests.get(url, headers=hdr, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        print(f"[http] {url} failed: {exc}")
        return None


def _stooq_daily(symbol: str, start: str = HISTORY_START) -> Optional[pd.Series]:
    """Keyless, datacenter-friendly DAILY price source (stooq.com CSV)."""
    txt = _http_get(f"https://stooq.com/q/d/l/?s={symbol}&i=d")
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


def _stooq_monthly(symbol: str, start: str = HISTORY_START) -> Optional[pd.Series]:
    """Month-end close from Stooq daily (used as Yahoo fallback)."""
    d = _stooq_daily(symbol, start=start)
    if d is None:
        return None
    s = d.resample("ME").last()
    return s if not s.empty else None


def _yf_weekly(ticker: str, start: str = HISTORY_START) -> Optional[pd.Series]:
    """Weekly-close via yfinance, with a Stooq fallback.

    Used for the F8 tech-froth feature so the 156-week rolling window really
    spans ~3 years of weekly observations. Yahoo 429s from cloud IPs; Stooq is
    keyless and server-friendly, so prices stay REAL on deploy when Yahoo is
    blocked.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, auto_adjust=True, actions=False, progress=False)
        if hist is not None and not hist.empty:
            s = hist["Close"].dropna().resample("W-FRI").last()
            if not s.empty:
                return s
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[yfinance-weekly] {ticker} failed: {exc}")
    st = _STOOQ_MAP.get(ticker)
    if st:
        d = _stooq_daily(st, start=start)
        if d is not None:
            return d.resample("W-FRI").last().dropna()
    return None


def _yf_monthly(ticker: str, start: str = HISTORY_START) -> Optional[pd.Series]:
    """Monthly adjusted-close via yfinance, with a Stooq fallback.

    yfinance (Yahoo) frequently 429s from cloud / datacenter IPs (Render, HF
    Spaces). Stooq is keyless and server-friendly, so it keeps prices REAL on
    deploy when Yahoo is blocked.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, auto_adjust=True, actions=False, progress=False)
        if hist is not None and not hist.empty:
            s = hist["Close"].dropna().resample("ME").last()
            if not s.empty:
                return s
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[yfinance] {ticker} failed: {exc}")
    st = _STOOQ_MAP.get(ticker)
    if st:
        return _stooq_monthly(st, start=start)
    return None


def _google_trends(keyword: str = "Buy Stocks",
                   start: str = "2015-01-01") -> Optional[pd.Series]:
    """Monthly mean of Google Trends interest (0-100). None on failure."""
    try:
        from pytrends.request import TrendReq
    except Exception:
        return None
    try:
        pytrends = TrendReq(timeout=(10, 25))
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        pytrends.build_payload([keyword], timeframe=f"{start} {end}")
        data = pytrends.interest_over_time()
        if data is None or data.empty:
            return None
        s = data[keyword].dropna().resample("ME").mean()
        return s if not s.empty else None
    except Exception as exc:  # pragma: no cover - network / rate-limit dependent
        print(f"[pytrends] '{keyword}' failed: {exc}")
        return None


def _aaii_sentiment(start: str = "1987-01-01") -> Optional[pd.Series]:
    """AAII Investor Sentiment Survey — % bullish (REAL, keyless, 1987+).

    High retail bullishness == euphoria == bubble risk, so we use the level
    directly (no inversion). 35+ years of weekly history makes the 20-year
    percentile meaningful and lets the 2000->today backtest run on real data.
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


def _finra_retail_share(start: str = "2019-01-01") -> Optional[pd.Series]:
    """FINRA Retail Trading Data — share of total US equity volume from retail.

    REAL, keyless, monthly, published by FINRA. The downloadable workbook
    carries the FULL historical series (every month since inception) in a
    single sheet, so fetching the latest file yields the whole history.

    The bubble-relevant metric is the percentage of total US equity trading
    volume attributable to retail investors (FINRA labels the column
    "Retail Volume %" / "Retail Share"). We auto-detect the column so the
    code keeps working if FINRA renames it.

    FINRA publishes ~1 month after month-end; the file lives under an upload
    month folder. We try a sliding window of recent (data_month, upload_month)
    candidates so the fetcher self-heals as months roll over.
    """
    import io

    today = pd.Timestamp.today()
    candidates = []
    for back in range(0, 18):                      # last ~18 data months
        data_month = today - pd.DateOffset(months=1 + back)
        for up_off in (1, 2, 0):                   # upload folder offset
            up = data_month + pd.DateOffset(months=up_off)
            folder = up.strftime("%Y-%m")
            fname = data_month.strftime("%Y_%m")
            candidates.append(
                f"https://www.finra.org/sites/default/files/{folder}/"
                f"retail_trading_data_{fname}.xlsx"
            )

    raw = None
    for url in candidates:
        try:
            hdr = {"User-Agent": "Mozilla/5.0 (compatible; bubble-monitor/1.0)"}
            r = requests.get(url, headers=hdr, timeout=12)
            if r.status_code != 200 or not r.content:
                continue
            raw = r.content
            break
        except Exception as exc:
            print(f"[finra] GET {url} failed: {exc}")
            continue

    if raw is None:
        print("[finra] no retail trading workbook reachable")
        return None

    try:
        xls = pd.ExcelFile(io.BytesIO(raw))

        # FINRA workbooks sometimes carry a banner/title row; try a few header
        # configs until we land on a sheet that has the retail + share columns.
        df = None
        for cfg in (dict(), dict(header=1), dict(skiprows=1)):
            for sh in xls.sheet_names:
                tmp = xls.parse(sh, **cfg)
                cols_l = [str(c).lower() for c in tmp.columns]
                has_retail = any("retail" in c for c in cols_l)
                has_share = any(
                    ("volume" in c and "%" in c) or "share" in c
                    for c in cols_l
                )
                if has_retail and has_share:
                    df = tmp
                    break
            if df is not None:
                break
        if df is None:
            df = xls.parse(xls.sheet_names[0])

        # date / period column
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("month", "date", "period"):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

        # retail share column
        share_col = None
        for c in df.columns:
            cl = str(c).lower()
            if "retail" in cl and ("%" in cl or "share" in cl or "volume %" in cl):
                share_col = c
                break
        if share_col is None:
            for c in df.columns:
                cl = str(c).lower()
                if "retail" in cl and "volume" in cl:
                    share_col = c
                    break
        if share_col is None:
            print("[finra] could not locate retail share column; columns="
                  f"{list(df.columns)}")
            return None

        out = df[[date_col, share_col]].copy()
        # Robust date parsing: FINRA mixes "2019-10-01" and "Oct-19" styles.
        parsed = pd.to_datetime(out[date_col], errors="coerce")
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(out[date_col], errors="coerce", format="%b-%y")
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(out[date_col], errors="coerce", format="mixed")
        out[date_col] = parsed
        out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
        out[share_col] = pd.to_numeric(out[share_col], errors="coerce")
        s = out[share_col].dropna().resample("ME").last()
        s = s[s.index >= pd.Timestamp(start)]
        # normalize to a fraction if expressed as a percent (e.g. 21.3 -> 0.213)
        if s.max() > 1.5:
            s = s / 100.0
        return s if not s.empty else None
    except Exception as exc:
        print(f"[finra] parse failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Rolling percentile (no look-ahead)
# ---------------------------------------------------------------------------
def rolling_pct(series, window: int = WINDOW_MONTHS,
                min_periods: int | None = None) -> pd.Series:
    """
    Percentile rank (0-100) of each value within its TRAILING `window`
    observations.

    Vectorized via pandas rolling rank (C-level, ~100x faster than the naive
    per-point loop). For each date the output is the percentile of the current
    value among all values in the trailing window, so it is purely
    backwards-looking (no look-ahead bias) and safe to re-compute for any
    historical date.

    `min_periods` keeps the first stretch (before a full window has
    accumulated) as NaN rather than a noisy single-point percentile.
    """
    s = pd.Series(series, dtype="float64")
    if min_periods is None:
        min_periods = max(24, window // 4)
    rp = s.rolling(window, min_periods=min_periods).rank(pct=True, method="average")
    return rp * 100.0


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------
def compute_composite(feat_pct: pd.DataFrame, weights: dict = WEIGHTS) -> pd.Series:
    """
    Weighted blend of available feature percentiles at each date.

    Fully vectorized: at every date the missing features have their weight
    redistributed across the present ones, so the output always lives on 0-100
    even when some F1-F8 features are NaN.
    """
    cols = list(weights.keys())
    w = pd.Series(weights)[cols]
    vals = feat_pct[cols]
    avail = vals.notna()
    # weighted sum of available percentiles / sum of available weights
    numer = (vals * w).sum(axis=1)
    denom = (avail * w).sum(axis=1)
    return numer / denom


def contribution_factor(score: float) -> float:
    """Monthly DCA multiplier from the Bubble Risk Score (per spec)."""
    if pd.isna(score):
        return 1.0  # neutral default when score is unknown
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
# Feature construction
# ---------------------------------------------------------------------------
def build_features() -> Tuple[pd.DataFrame, dict]:
    """
    Fetch raw series, build the 8 risk-percentile features, and return:
      * feat_pct : monthly DataFrame, columns F1..F8 (0-100)
      * meta     : dict with per-feature availability / source notes
    """
    meta: dict = {}

    # ---- Raw underlying series -------------------------------------------
    cape = _fred("CAPE")                                   # Shiller CAPE (monthly)
    wilshire = _fred("WILL5000INDFC")                      # Wilshire 5000 mkt cap ($M)
    gdp = _fred("GDP")                                     # quarterly -> ffill
    vix = _fred("VIXCLS")                                  # daily VIX -> monthly mean
    margin = _fred("MARGINSL")                             # margin debt ($M, monthly)
    baa_tsy = _fred("BAA10Y")                              # credit spread (monthly)
    m2 = _fred("M2SL")                                     # M2 (monthly)
    fed_bs = _fred("WALCL")                                # Fed assets (weekly)
    cpi = _fred("CPIAUCSL")                                # CPI (monthly)
    ffr = _fred("FEDFUNDS")                                # Fed funds (monthly)
    spx = _yf_monthly("^GSPC")                             # S&P 500
    spy = _yf_monthly("SPY")                               # (for ratio if needed)
    qqq = _yf_monthly("QQQ")                               # Nasdaq 100 ETF
    trends = _google_trends("Buy Stocks")                  # 2015+ only (secondary)
    aaii = _aaii_sentiment()                               # 1987+ real retail sentiment

    # Quarterly GDP -> monthly
    if gdp is not None:
        gdp = gdp.resample("ME").ffill()

    # Weekly Fed BS -> monthly
    if fed_bs is not None:
        fed_bs = fed_bs.resample("ME").last()

    # VIX daily -> monthly mean
    if vix is not None:
        vix = vix.resample("ME").mean()

    # ---- Common monthly index --------------------------------------------
    idx = pd.date_range(HISTORY_START, pd.Timestamp.today(), freq="ME")
    def align(s: Optional[pd.Series]) -> Optional[pd.Series]:
        if s is None:
            return None
        return s.reindex(idx).ffill(limit=6)  # tolerate small gaps

    cape = align(cape)
    wilshire = align(wilshire)
    gdp = align(gdp)
    vix = align(vix)
    margin = align(margin)
    baa_tsy = align(baa_tsy)
    m2 = align(m2)
    fed_bs = align(fed_bs)
    cpi = align(cpi)
    ffr = align(ffr)
    spx = align(spx)
    qqq = align(qqq)
    spy = align(spy)
    trends = align(trends)
    aaii = align(aaii)

    feat = pd.DataFrame(index=idx)

    # ---- F1 Valuation ----------------------------------------------------
    parts = []
    if cape is not None:
        parts.append(rolling_pct(cape))
    if wilshire is not None and gdp is not None:
        buffett = (wilshire * 1e6) / (gdp * 1e9)  # ratio, unitless
        parts.append(rolling_pct(buffett))
    feat["F1_valuation"] = np.mean(parts, axis=0) if parts else np.nan
    meta["F1_valuation"] = f"CAPE={'Y' if cape is not None else 'N'} " \
                           f"Buffett={'Y' if wilshire is not None and gdp is not None else 'N'}"

    # ---- F2 Momentum (6m annualized S&P return) --------------------------
    if spx is not None:
        mom6 = (spx / spx.shift(6)) ** (12.0 / 6.0) - 1.0
        feat["F2_momentum"] = rolling_pct(mom6)
        meta["F2_momentum"] = "Y"
    else:
        feat["F2_momentum"] = np.nan
        meta["F2_momentum"] = "N"

    # ---- F3 Sentiment (VIX inverted) -------------------------------------
    if vix is not None:
        # low VIX == complacency == risk -> invert the percentile
        feat["F3_sentiment"] = 100.0 - rolling_pct(vix)
        meta["F3_sentiment"] = "Y"
    else:
        feat["F3_sentiment"] = np.nan
        meta["F3_sentiment"] = "N"

    # ---- F4 Leverage (margin/mktcap + credit spread) ---------------------
    # Credit spread (BAA10Y) is INVERTED: a *compressed* spread (investors
    # blindly chasing risk, ultra-loose credit) is a bubble signal, whereas a
    # *wide* spread marks panic / liquidity stress (2008, 2020-03) and is the
    # opposite of froth. So low spread -> high risk percentile (100 - pct).
    parts = []
    if margin is not None and wilshire is not None:
        margin_ratio = margin / wilshire  # both in $M -> ratio
        parts.append(rolling_pct(margin_ratio))
    if baa_tsy is not None:
        parts.append(100.0 - rolling_pct(baa_tsy))   # inverted credit spread
    feat["F4_leverage"] = np.mean(parts, axis=0) if parts else np.nan
    meta["F4_leverage"] = f"Margin={'Y' if margin is not None and wilshire is not None else 'N'} " \
                          f"Credit(inv)={'Y' if baa_tsy is not None else 'N'}"

    # ---- F5 Liquidity (M2 YoY + Fed BS YoY) ------------------------------
    parts = []
    if m2 is not None:
        m2_yoy = m2.pct_change(12) * 100.0
        parts.append(rolling_pct(m2_yoy))
    if fed_bs is not None:
        bs_yoy = fed_bs.pct_change(12) * 100.0
        parts.append(rolling_pct(bs_yoy))
    feat["F5_liquidity"] = np.mean(parts, axis=0) if parts else np.nan
    meta["F5_liquidity"] = f"M2={'Y' if m2 is not None else 'N'} " \
                           f"FedBS={'Y' if fed_bs is not None else 'N'}"

    # ---- F6 Business sentiment (FRED EMVMACROBUS, INVERTED) --------------
    # EMVMACROBUS = "Equity Market Volatility Tracker: Macroeconomic News &
    # Outlook: Business Investment And Sentiment" (Baker/Bloom/Davis via FRED,
    # monthly, 1985+). Per the spec, a LOW index = complacency / low perceived
    # business risk = bubble-prone, so we INVERT: low -> high risk percentile.
    # Because it is a FRED series (same source as F1/F3/F4/F5/F7), with a
    # FRED_API_KEY it is as stable as every other macro feature — no scraper,
    # no second key. Only if FRED is entirely unavailable do we fall back to the
    # keyless FINRA/AAII/Trends blend so the feature never silently drops.
    emv = _fred("EMVMACROBUS")
    if emv is not None and emv.notna().sum() >= 12:
        emv = align(emv)
        feat["F6_business"] = 100.0 - rolling_pct(emv)   # inverted
        meta["F6_business"] = "EMVMACROBUS (FRED, inv)"
    else:
        # ---- fallback: FINRA retail volume share + AAII bullish blend -----
        # High retail participation / bullishness == euphoria == bubble risk;
        # the level is used directly (no inversion). FINRA only reaches back to
        # ~2019, so we BLEND with the AAII % Bullish survey (real, 1987+) for
        # 2000-2019, and Google Trends (2015+) as last-resort filler. Each
        # series is percentile-ranked in its OWN scale first (different units)
        # then merged with combine_first, yielding one continuous F6.
        finra = _finra_retail_share()
        finra = align(finra)
        f6_sources = []
        if finra is not None and finra.notna().sum() >= 6:
            f6_sources.append(("FINRA", finra))
        if aaii is not None:
            f6_sources.append(("AAII", aaii))
        if trends is not None:
            f6_sources.append(("Trends", trends))

        if not f6_sources:
            feat["F6_business"] = np.nan
            meta["F6_business"] = "N"
        else:
            blended = None
            for _name, s in f6_sources:
                pc = rolling_pct(s)
                blended = pc if blended is None else blended.combine_first(pc)
            feat["F6_business"] = blended
            finra_start = None
            if finra is not None and finra.notna().any():
                finra_start = finra.index[finra.notna()][0].date()
            meta["F6_business"] = "blend " + "+".join(n for n, _ in f6_sources) + \
                (f" (FINRA {finra_start}+)" if finra_start else "") + " (EMV fallback)"

    # ---- F7 Policy (real fed funds, inverted) ----------------------------
    if ffr is not None and cpi is not None:
        cpi_yoy = cpi.pct_change(12) * 100.0
        real_ffr = ffr - cpi_yoy
        feat["F7_policy"] = 100.0 - rolling_pct(real_ffr)
        meta["F7_policy"] = "Y"
    else:
        feat["F7_policy"] = np.nan
        meta["F7_policy"] = "N"

    # ---- F8 Tech froth (QQQ/SPY, 156-week rolling percentile) -----------
    # 156 weeks (~3 years): tech bubbles build over 2-3 years, and a shorter
    # window would mark the froth "cleared" during a long high-level
    # consolidation. We compute the ratio on WEEKLY bars (so 156 observations
    # really span ~3 years) then roll up to month-end for the composite. If
    # weekly prices fail we fall back to a 36-month (≈3y) monthly window.
    qqq_w = _yf_weekly("QQQ")
    spy_w = _yf_weekly("SPY")
    if qqq_w is not None and spy_w is not None:
        ratio_w = (qqq_w / spy_w).dropna()
        ratio_w = ratio_w[ratio_w.index >= pd.Timestamp(HISTORY_START)]
        f8_weekly = rolling_pct(ratio_w, window=WINDOW_TECH_WEEKS)
        feat["F8_tech"] = f8_weekly.resample("ME").last()
        meta["F8_tech"] = "Y (156w weekly)"
    elif qqq is not None and spy is not None:
        ratio = qqq / spy
        feat["F8_tech"] = rolling_pct(ratio, window=WINDOW_TECH_MONTHS)
        meta["F8_tech"] = "Y (36m fallback)"
    elif qqq is not None and spx is not None:
        ratio = qqq / spx
        feat["F8_tech"] = rolling_pct(ratio, window=WINDOW_TECH_MONTHS)
        meta["F8_tech"] = "Y (36m vs SPX)"
    else:
        feat["F8_tech"] = np.nan
        meta["F8_tech"] = "N"

    return feat, meta


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

    bumps = (peak("2000-03-01", 18, 5)   # dot-com peak  (~84)
             + peak("2007-10-01", 11, 7)  # pre-GFC       (~57)
             + peak("2021-12-01", 26, 6)  # COVID tech    (~92)
             + peak("2025-01-01", 6, 9))  # recent
    rng = np.random.RandomState(7)
    s = base + bumps + rng.normal(0, 2.0, len(idx))
    s = np.clip(s, 0, 100)
    return pd.Series(s, index=idx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_monthly_scores(refresh: bool = False) -> Tuple[pd.Series, dict]:
    """
    Returns (monthly_score_series, meta).
    Order of resolution: on-disk cache -> live fetch -> synthetic.
    `meta['source']` is one of 'live', 'cache', 'synthetic'.
    """
    meta = {"source": "unknown"}

    if not refresh and os.path.exists(CACHE_PATH):
        try:
            cached = pd.read_parquet(CACHE_PATH)
            score = cached["score"]
            meta = cached.attrs.get("meta", {"source": "cache"})
            meta["source"] = "cache"
            return score, meta
        except Exception:
            pass  # fall through to rebuild

    feat, fmeta = build_features()
    score = compute_composite(feat, WEIGHTS)

    # ---- Live-feature accounting (zero-crash weight renormalization) ------
    # Count features that actually contributed data (non-NaN anywhere), NOT the
    # meta string prefix, so partial features (e.g. F4 "Margin=N Credit=Y")
    # are scored correctly. The composite already redistributes the missing
    # weight, so the score stays on 0-100 regardless of how many are live.
    live_cols = [c for c in WEIGHTS if feat[c].notna().any()]
    available_count = len(live_cols)
    missing = [c for c in WEIGHTS if c not in live_cols]
    print(f"[score] {available_count}/8 features live"
          f"  (missing: {missing or '-'})")
    if available_count >= 4:  # enough signal to be "live"
        meta = {"source": "live", "features": fmeta,
                "available_count": available_count}
        try:
            out = feat.copy()
            out["score"] = score
            out.attrs["meta"] = meta
            out.to_parquet(CACHE_PATH)
        except Exception:
            pass
        return score, meta

    # Not enough live data -> synthetic, clearly flagged
    synth = _synthetic_scores()
    meta = {"source": "synthetic",
            "features": fmeta,
            "available_count": len(available),
            "note": "Live APIs unavailable; showing deterministic synthetic series."}
    return synth, meta


def get_latest_state(refresh: bool = False) -> dict:
    """Convenience bundle for the dashboard (latest score + per-feature)."""
    score, meta = get_monthly_scores(refresh=refresh)
    score = score.dropna()
    if score.empty:
        return {"score": np.nan, "status": "Unknown", "features": {},
                "source": meta.get("source", "unknown"), "as_of": None,
                "meta": meta}

    latest_date = score.index[-1]
    latest_score = float(score.iloc[-1])

    # Per-feature detail — recompute features quickly from cache if present
    feat = None
    if os.path.exists(CACHE_PATH):
        try:
            cached = pd.read_parquet(CACHE_PATH)
            if "score" in cached.columns:
                feat = cached.drop(columns=["score"])
        except Exception:
            feat = None

    features = {}
    if feat is not None and latest_date in feat.index:
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

    return {
        "score": latest_score,
        "status": status_of(latest_score),
        "features": features,
        "source": meta.get("source", "unknown"),
        "as_of": latest_date,
        "meta": meta,
    }


def get_price_series(ticker: str, start: str = LIVE_START,
                      refresh: bool = False) -> Optional[pd.Series]:
    """Monthly price series used by the dashboard main chart and backtest.

    Disk-cached per ticker (``prices_cache.parquet``) so cloud deploys
    (Render / HuggingFace Spaces) do NOT re-pull 25+ years of daily data on
    every cold start or page load. With cache present the first request is
    instant and free-tier health checks don't time out.
    """
    if not refresh and os.path.exists(PRICES_CACHE):
        try:
            pc = pd.read_parquet(PRICES_CACHE)
            if ticker in pc.columns:
                s = pc[ticker].dropna()
                if not s.empty:
                    return s[s.index >= pd.Timestamp(start)] if start else s
        except Exception:
            pass

    s = _yf_monthly(ticker, start=start)
    if s is None:
        return None

    try:
        pc = pd.DataFrame()
        if os.path.exists(PRICES_CACHE):
            pc = pd.read_parquet(PRICES_CACHE)
        pc[ticker] = s
        pc.to_parquet(PRICES_CACHE)
    except Exception:
        pass
    return s


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
