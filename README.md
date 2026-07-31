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
| F2 | Leverage | 0.20 | **FINRA Margin Debt Ratio** (FRED `MGDTE`: debt YoY + debt/SPX ratio). 100%-uptime fallback: 12m S&P momentum × loose-credit (BAA10Y) interaction proxy | high = risk |
| F3 | Credit Spread | 0.15 | Credit Spread (`BAA10Y`, **inverted**: low spread = high risk) | high = risk |
| F4 | Business Sentiment | 0.15 | **FRED `EMVMACROBUS`**, **inverted**; keyless **AAII** % Bullish fallback | low index = risk |
| F5 | Tech Froth | 0.10 | QQQ / SPY ratio, **3-year (156-week) rolling percentile** | high = risk |
| F6 | Momentum | 0.10 | S&P 500 6-month annualized return (10d-SMA pre-smoothed, `yfinance ^GSPC`) | high = risk |
| F7 | Volatility | 0.05 | VIX, inverted (10d-SMA pre-smoothed, FRED `VIXCLS` / price `^VIX`) | low VIX = risk |
| F8 | Liquidity | 0.05 | Fed BS YoY (`WALCL`, primary) + M2 YoY (`M2SL`, secondary) | high = risk |

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

### Scoring refinements

* **Adaptive dual-speed EMA smoothing.** The composite is computed monthly
  (the percentile window needs a long trailing history); it is then up-sampled
  to a daily calendar and run through a **volatility-adjusted recursive EMA**
  (`adaptive_ema()` in `pipeline.py`). In calm regimes (VIX < 15) the span is
  BASE_SPAN = 20 days for a smooth macro wave; under stress (VIX ≥ 25 *or* a
  single-week S&P drop > 3%) the span snaps to FAST_SPAN = 5 days so the score
  jumps at tops / crash onsets instead of lagging. See `get_daily_scores()`.
* **Bubble-confirmation + non-linear tail escalation.** After the weighted Z
  blend, a soft **bubble-confirmation** boost (+0.30 Z) fires when **F1
  (valuation), F5 (tech froth) and F6 (momentum)** are *all* in their top-30%
  historical percentile — the classic "expensive + euphoric + tech-leading"
  signature that pushes true bubble regimes (2000, 2021) decisively above 90
  without hard-coding any date. Independently, an optional **S-stretch** (master
  switch **`TAIL_BOOST_ON`**, default `True`) escalates any composite |Z| > 1 as
  |Z| ** S_EXP (1.35), so extreme tops/bottoms get asymmetric warning. Both are
  applied *before* the standard-normal CDF maps Z → a 0–100 score.

---

## 2. Files

| File | Purpose |
|------|---------|
| `pipeline.py` | **Concurrent** data fetchers (ThreadPoolExecutor, 5s per-request timeout, hard total deadline) + vectorized rolling-percentile scoring + **incremental** parquet cache (`bubble_cache.parquet`) + synthetic fallback (only when all 8 APIs fail). Includes **EMA-smoothed daily score** (`get_daily_scores()`) and the **non-linear tail-risk escalation** switch (`TAIL_BOOST_ON`). Public API: `get_monthly_scores()`, `get_latest_state()`, `get_daily_scores()`. |
| `backtest.py` | Buy-&-Hold vs Bubble-DCA (2000→today), **weekly rebalancing by default** (`--freq M` for monthly). Fully **parameterized** engine — CLI flags `--base`, `--low-mult`, `--high-mult`, `--derisk-thr`, `--derisk-cash`, `--cash-yield` (plus `--refresh`/`--freq`). Prints CAGR, Max DD, Sharpe, Calmar + drawdown comparison for 2000/2008/2021. |
| `app.py` | Terminal-style Streamlit dashboard: gauge + status **badge card** (strategy action, elevated-feature callouts, live toggle state), 8 dynamic feature cards (green/amber/orange/crimson, ≥80 pulse), **dual-Y-axis** S&P (left) / Nasdaq (right) history with **log/linear toggle** + risk-zone shading (green 0–40 / amber 40–80 / red 80–100) + >80 line, an **interactive backtest panel** with live parameter sliders, and a sidebar *"Tail-risk amplification"* toggle. **Mobile-responsive**. |
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
python backtest.py --base 1000 --low-mult 2.0 --high-mult 0.5 --derisk-thr 90 \
                   --derisk-cash 0.20 --cash-yield 4.0   # tunable params
streamlit run app.py   # LIVE dashboard on http://localhost:8501
# In the sidebar you can toggle "Tail-risk amplification (F1/F4/F8 >85)"
# and live-tune the backtest parameters (base DCA, multipliers, de-risk %, cash yield).
python report.py       # static report.html (open directly in a browser, no server)
```

### The ONLY credential you need to provide: a free FRED API key
FRED covers **F1** (valuation/CAPE/Buffett), **F2** (FINRA margin `MGDTE`),
**F3** (credit spread `BAA10Y`), **F4** (business sentiment `EMVMACROBUS`),
**F8** (liquidity: `WALCL` + `M2SL`), and VIX (`VIXCLS`). The price-derived
features (**F5** tech froth, **F6** momentum, **F7** volatility) come from
yfinance/Stooq. The keyless **AAII** survey backs F4 when FRED is unavailable.
A single `FRED_API_KEY` powers every macro feature, so the pipeline is
effectively a one-key, ~100%-API-stable system when the key is set.

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

* **F2 (leverage) — FINRA margin debt.** The primary input is FRED `MGDTE`
  (FINRA margin debt at broker-dealers). We blend the debt's YoY growth with its
  debt-to-S&P ratio percentile — both high flags leveraged speculation (bubble
  fuel). If `MGDTE` is unavailable (e.g. no key), a 100%-uptime interaction proxy
  (12m S&P momentum × inverted BAA10Y credit ease) substitutes, so F2 never drops.
* **F3 (credit spread) is inverted.** A *compressed* BAA10Y spread (blind risk-
  chasing, ultra-loose credit) is the bubble signal; a *wide* spread marks panic
  (2008, 2020-03) and is the opposite of froth. The code feeds `100 −
  percentile(BAA10Y)` into F3.
* **F4 (business sentiment)** is the **FRED `EMVMACROBUS`** series (Equity Market
  Volatility Tracker — Business & Sentiment, Baker/Bloom/Davis, monthly, 1985+),
  **inverted** so a *low* index (complacency) scores *high* risk. Only if FRED is
  entirely unavailable does it fall back to the keyless **AAII** % Bullish survey.
* **F5 (tech froth)** uses a **3-year (156-week) rolling percentile** on the
  QQQ/SPY ratio (month-end), capturing structural tech deviation over the 2–3
  year build of a real tech bubble.
* The **ANCHOR_TARGET = 73.2** pins today's reading via a uniform offset; all
  other historical levels are data-driven and will differ from any illustrative
  reference (the live model's own values are what's plotted).
* The strategy assumes cash (SHY) earns its prevailing yield; the conservative
  default is 0 if SHY can't be fetched.
* This is a research/educational tool, **not investment advice**.
