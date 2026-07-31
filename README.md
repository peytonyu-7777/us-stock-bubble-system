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
| F4 | Leverage | 0.15 | Credit Spread (`BAA10Y`, **inverted**: low spread = high risk) | high = risk |
| F5 | Liquidity | 0.10 | Fed BS YoY (`WALCL`, primary) + M2 YoY (`M2SL`, secondary) | high = risk |
| F6 | Business Sentiment | 0.15 | **FRED `EMVMACROBUS`** (Equity Market Volatility Tracker — Business & Sentiment), **inverted** (low index = risk). Keyless **AAII** % Bullish survey used only as fallback when FRED is unavailable | low index = risk |
| F7 | Policy stance | 0.05 | Real Fed Funds (`FEDFUNDS` − CPI YoY), inverted | low real rate = risk |
| F8 | Tech froth | 0.15 | QQQ / SPY ratio, **3-year (156-week) rolling percentile** | high = risk |

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
| `pipeline.py` | **Concurrent** data fetchers (ThreadPoolExecutor, 5s per-request timeout, hard total deadline) + vectorized rolling-percentile scoring + **incremental** parquet cache (`bubble_cache.parquet`) + synthetic fallback (only when all 8 APIs fail). Public API: `get_monthly_scores()`, `get_latest_state()`. |
| `backtest.py` | Buy-&-Hold vs Bubble-DCA (2000→today), **weekly rebalancing by default** (`--freq M` for monthly). Prints CAGR, Max DD, Sharpe, Calmar + drawdown comparison for 2000/2008/2021. |
| `app.py` | Streamlit dashboard: gauge, 8 feature cards, S&P/Nasdaq + score history with >80 zone shaded. **Mobile-responsive** (feature cards reflow, columns stack on phones). |
| `requirements.txt`, `Dockerfile`, `render.yaml` | Zero-cost deploy config. |

---

## 3. Local run

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env   # then edit .env and paste your free FRED_API_KEY
python pipeline.py     # score history -> bubble_cache.parquet (full fetch first run)
python backtest.py          # weekly rebalance strategy comparison (default)
python backtest.py --freq M   # monthly rebalance instead
python backtest.py --refresh  # force re-fetch score history
streamlit run app.py   # LIVE dashboard on http://localhost:8501
python report.py       # static report.html (open directly in a browser, no server)
```

### The ONLY credential you need to provide: a free FRED API key
FRED covers **F1** (valuation/CAPE/Buffett), **F3** (VIX), **F4** (credit spread),
**F5** (M2 + Fed balance sheet), **F6** (EMVMACROBUS business
sentiment) and **F7** (real fed funds). Everything else — yfinance/Stooq prices,
and the keyless **AAII** fallback for F6 — is **keyless and free**. Because F6
now comes from FRED too, a single `FRED_API_KEY` powers all six macro features
(incl. the business-sentiment signal), so the pipeline is effectively a
one-key, 100%-API-stable system when the key is set.

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
3. In the service settings add the env var **`FRED_API_KEY`** (recommended).
   **Set it as BOTH a runtime AND a build environment variable** (Render lets you
   toggle "Build" vs "Runtime" for each env var). The `Dockerfile` pre-fetches the
   score + price history **at build time** and bakes `bubble_cache.parquet`
   into the image, so the first request is instant (no slow cold-start fetch,
   health check won't time out). Baking it as a *build* var is what lets **F6
   (EMVMACROBUS)** and the other FRED features land in the image — otherwise the
   runtime key alone can't refresh the baked cache.
   If you only set it as a runtime var (no build var), F6 will still render using
   the real keyless AAII fallback — just not the EMVMACROBUS series.

> Free web tier: the service sleeps after ~15 min idle and takes ~30 s to spin
> back up. If a zero-sleep free option is preferred, use **HuggingFace Spaces**.

## Deploy to HuggingFace Spaces (free)

1. New Space → SDK **Streamlit**, pick a repo name.
2. Upload `app.py`, `pipeline.py`, `backtest.py`, `requirements.txt`, `Dockerfile`.
3. (Optional) add `FRED_API_KEY` under Settings → Variables.

> Both platforms allow outbound HTTPS. On Render the cache is baked at build;
> on HF the app fetches live on first load and writes `bubble_cache.parquet`
> to the Space's persistent storage afterwards. Refresh only re-fetches the last
> ~30 days (incremental), so even a cold HF load is fast.

---

## 5. Caveats / honesty notes

* **F4 (leverage) — credit spread is inverted.** A *compressed* BAA10Y spread
  (investors blindly chasing risk, ultra-loose credit) is the bubble signal; a
  *wide* spread marks panic/liquidity stress (2008, 2020-03) and is the opposite
  of froth. The code therefore feeds `100 − percentile(BAA10Y)` into F4, so a
  historically low spread contributes a *high* risk score.
* **F6 (business sentiment)** is now the **FRED `EMVMACROBUS`** series (Equity
  Market Volatility Tracker — Business Investment & Sentiment, Baker/Bloom/Davis,
  monthly, 1985+), **inverted** so a *low* index (complacency / low perceived
  business risk) scores *high* bubble risk. It is a FRED series like F1/F3/F4/F5/F7,
  so with `FRED_API_KEY` set it is as stable as every other macro feature — no
  scraper, no second key. Only if FRED is entirely unavailable does F6 fall back
  to the keyless **AAII** % Bullish survey, so the feature never silently drops and
  the weight is renormalized away when even that fallback fails.
* **F8 (tech froth)** now uses a **3-year (156-week) rolling percentile** computed
  on *weekly* QQQ/SPY bars (rolled up to month-end) instead of 52 weeks. Tech
  bubbles build over 2–3 years, and a 1-year window would mark
  the froth "cleared" during a long high-level consolidation — the longer window
  captures the structural deviation.
* The **2000 / 2021 reference scores (83.2 / 92.1)** are illustrative anchors from
  the Dalio framing; the live model's own values will differ and are what's plotted.
* The strategy assumes cash (SHY) earns its prevailing yield; the conservative
  default is 0 if SHY can't be fetched.
* This is a research/educational tool, **not investment advice**.
