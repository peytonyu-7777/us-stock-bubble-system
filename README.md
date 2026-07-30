# US Equity Bubble Risk Monitor (Dalio-style)

A zero-cost, open-data system that scores US equity "bubble risk" on a **0–100**
scale using Ray Dalio's 8-feature framework, backtests a bubble-aware dollar-cost
averaging (DCA) strategy against buy-&-hold, and serves everything through a
Streamlit dashboard deployable to **Render** or **HuggingFace Spaces** in one click.

---

## 1. The 8 features & data sources

| # | Feature | Weight | Proxy / Source | Direction |
|---|---------|--------|----------------|-----------|
| F1 | Valuation | 0.20 | CAPE (FRED `CAPE`) + Buffett Indicator (Wilshire `WILL5000INDFC` / `GDP`) | high = risk |
| F2 | Momentum | 0.10 | S&P 500 6-month annualized return (`yfinance ^GSPC`) | high = risk |
| F3 | Sentiment | 0.10 | VIX, inverted (FRED `VIXCLS`) | low VIX = risk |
| F4 | Leverage | 0.20 | Margin Debt / Mkt Cap (`MARGINSL`) + Credit Spread (`BAA10Y`) | high = risk |
| F5 | Liquidity | 0.10 | M2 YoY (`M2SL`) + Fed BS YoY (`WALCL`) | high = risk |
| F6 | Retail inflows | 0.15 | **FINRA Retail Volume Share** (real, keyless, monthly, ~2019+) + AAII % Bullish fallback | high = risk |
| F7 | Policy stance | 0.05 | Real Fed Funds (`FEDFUNDS` − CPI YoY), inverted | low real rate = risk |
| F8 | Tech froth | 0.10 | QQQ / SPY ratio, 52-week percentile | high = risk |

Each feature is converted to a **percentile rank (0–100) within a trailing 20-year
window** (no look-ahead bias), then blended with the weights above. If any feature
is unavailable, its weight is redistributed so the score always stays on 0–100.

### Risk bands → DCA rule
| Score | Zone | DCA multiplier (per rebalance period) |
|-------|------|------------------------|
| 0–40 | Cooling | 2.0× |
| 40–60 | Normal | 1.5× |
| 60–80 | Watch | 1.0× |
| 80–90 | Elevated | 0.5× |
| 90–100 | Bubble Warning | 0× + move 20% equity → cash |

---

## 2. Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Data fetchers + rolling-percentile scoring + cache + synthetic fallback. Public API: `get_monthly_scores()`, `get_latest_state()`. |
| `backtest.py` | Buy-&-Hold vs Bubble-DCA (2000→today), **weekly rebalancing by default** (`--freq M` for monthly). Prints CAGR, Max DD, Sharpe, Calmar + drawdown comparison for 2000/2008/2021. |
| `app.py` | Streamlit dashboard: gauge, 8 feature cards, S&P/Nasdaq + score history with >80 zone shaded. |
| `requirements.txt`, `Dockerfile`, `render.yaml` | Zero-cost deploy config. |

---

## 3. Local run

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env   # then edit .env and paste your free FRED_API_KEY
python pipeline.py     # score history -> bubble_cache.parquet
python backtest.py          # weekly rebalance strategy comparison (default)
python backtest.py --freq M   # monthly rebalance instead
python backtest.py --refresh  # force re-fetch score history
streamlit run app.py   # LIVE dashboard on http://localhost:8501
python report.py       # static report.html (open directly in a browser, no server)
```

### The ONLY credential you need to provide: a free FRED API key
FRED covers **F1** (valuation/CAPE/Buffett), **F3** (VIX), **F4** (margin debt +
credit spread), **F5** (M2 + Fed balance sheet) and **F7** (real fed funds).
Everything else — yfinance/Stooq prices, **FINRA** retail volume share, AAII
sentiment fallback — is **keyless and free**.

```bash
export FRED_API_KEY=your_key_here      # free: https://fredaccount.stlouisfed.org/apikeys
```
On Render/HuggingFace: add `FRED_API_KEY` as an environment variable in the
dashboard. Without it the app still runs on cache or a clearly-flagged
**synthetic** series — it never silently presents fake numbers as if real.

**Local fill-in file:** copy `.env.example` → `.env` and paste your key there
(`pipeline.py` auto-loads it via `python-dotenv`; `.env` is git-ignored).
This is the only file you edit to supply the API key.

---

## 4. Deploy to Render (free)

1. Push this folder to a GitHub repo (init the git repo **inside this folder**
   so `render.yaml` sits at the repo root — the Blueprint reads it from root).
2. Go to **render.com → New → Blueprint**, connect the repo (it reads `render.yaml`).
3. In the service settings add the env var **`FRED_API_KEY`** (optional but recommended).
4. Deploy. The `Dockerfile` pre-fetches the score + price history **at build time**
   and bakes `bubble_cache.parquet` + `prices_cache.parquet` into the image, so the
   first request is instant (no slow cold-start fetch, health check won't time out).

> Free web tier: the service sleeps after ~15 min idle and takes ~30 s to spin
> back up. If a zero-sleep free option is preferred, use **HuggingFace Spaces**.

## Deploy to HuggingFace Spaces (free)

1. New Space → SDK **Streamlit**, pick a repo name.
2. Upload `app.py`, `pipeline.py`, `backtest.py`, `requirements.txt`, `Dockerfile`.
3. (Optional) add `FRED_API_KEY` under Settings → Variables.

> Both platforms allow outbound HTTPS. On Render the caches are baked at build;
> on HF the app fetches live on first load and writes `bubble_cache.parquet` /
> `prices_cache.parquet` to the Space's persistent storage afterwards.

---

## 5. Caveats / honesty notes

* **F6 (retail)** is the **FINRA Retail Trading Data** share of total US equity
  volume from retail investors (REAL, keyless, monthly). The fetcher tries a
  sliding window of recent (data-month, upload-month) workbook URLs and
  auto-detects the retail-share column, so it self-heals as months roll over.
  FINRA only publishes back to ~2019, so **before 2019 F6 is dropped by
  renormalization** — the 2000/2008 signals come from F1–F5/F7/F8. If FINRA is
  unreachable, it falls back to the **AAII % Bullish** series (real, 1987+).
* **F8** uses a 52-week window (per the brief) while F1–F7 use 20 years, so tech
  froth reacts faster than the slow valuation metrics — by design.
* The **2000 / 2021 reference scores (83.2 / 92.1)** are illustrative anchors from
  the Dalio framing; the live model's own values will differ and are what's plotted.
* The strategy assumes cash (SHY) earns its prevailing yield; the conservative
  default is 0 if SHY can't be fetched.
* This is a research/educational tool, **not investment advice**.
