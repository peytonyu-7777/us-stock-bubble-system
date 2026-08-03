# US Equity Bubble Risk Monitor (Dalio-style)

A zero-cost, open-data system that scores US equity "bubble risk" on a **0–100**
scale using a **5-module framework** (Valuation, Sentiment, Leverage, Structure,
Macro), backtests a bubble-aware dollar-cost averaging (DCA) strategy against
buy-&-hold, and serves everything through a Streamlit dashboard deployable to
**Render** or **HuggingFace Spaces** in one click.

It is a **RISK-ACCUMULATION INDICATOR** — it measures how much speculative risk
has built up, *not* a crash forecast.

---

## 1. The 5 modules & data sources

Every indicator is first converted to a **trailing robust-Z score** (MAD-based
z of the last 20 years, clipped to ±4σ — no look-ahead, no saturation). This
replaces the old trailing-percentile design, which pinned any record-breaking
reading at 100 for as long as it stayed the record (the "index stuck at max"
pathology). Robust-Z features feed five modules, each aggregated as the mean of
its sub-indicator Z-scores (a module with <50% coverage is neutralised to 0 =
median, not guessed):

| Module | Weight | Sub-indicators (proxy / source) | Direction |
|--------|--------|--------------------------------|-----------|
| **A. Valuation** | 30% | CAPE percentile (FRED `CAPE`); Buffett Indicator (Wilshire / GDP); S&P P/E log-z vs 10y mean | high = risk |
| **B. Sentiment** | 20% | AAII % Bullish (keyless) + FRED `EMVMACROBUS` (inverted); VIX-inverted | low caution = risk |
| **C. Leverage** | 20% | **FINRA Margin Debt Ratio** (`MGDTE`: YoY + debt/SPX); credit spread `BAA10Y` (inverted) | high = risk |
| **D. Structure** | 15% | QQQ/SPY tech-froth percentile (3y); S&P 6m momentum; equal-weight / S&P | high = risk |
| **E. Macro** | 15% | Real fed-funds rate (FedFunds − CPI YoY); yield-curve slope (`DGS10`−`DGS3MO`); Fed BS / M2 YoY | loose = risk |

Valuation applies an **acceleration curve** — flat below the 50th percentile,
linear to the 80th, convex 80–95th (power 1.8), then ramp to 100 — so
"expensive" and "true bubble" are cleanly separated. If a sub-indicator is
unavailable its module weight is redistributed so the score always stays 0–100.

### Risk bands (display)

| Score | Zone |
|-------|------|
| 0–40 | Cheap / Fear |
| 40–60 | Normal |
| 60–75 | Expensive |
| 75–90 | Bubble Risk |
| 90–100 | Extreme Bubble |

### Bubble-DCA rule (backtest de-risk)

The strategy **never changes your monthly outflow** (same $ as plain DCA) — the
index only decides how much of the cash on hand to *deploy* into SPY:

| Score | Deployment action |
|-------|-------------------|
| < 40 | **Deploy base + draw the stockpiled reserve** (up to 2× base) |
| 40–50 | Deploy base + up to 0.5× reserve |
| 50–80 | Deploy the base amount only |
| 80–95 | **Taper: deploy 0.5×, stockpile 0.5× as cash reserve** (earns SHY/FRED 3M yield) |
| ≥ 95 | Deploy 0 × + move 15% of the portfolio to cash (true extremes only) |

Total invested therefore equals plain DCA's exactly — the IRR comparison is a
pure measure of the index's *timing* skill (stockpile at bubble highs, deploy at
deep-fear lows). The legacy multiplier mode (scale the contribution itself) is
still available in the sidebar.

*(Band thresholds are strategy parameters, independent of the display risk
bands above.)*

### Scoring refinements (V3)

* **Fixed-gain calibration, not data-anchored affine.** The module blend Z is
  mapped with a deterministic linear map `score = 50 + 28 × blend_z`, clipped to
  [1, 99]. The old affine re-derived its scale from the dot-com window MAX on
  every run; when feature availability depressed that anchor (e.g. CAPE missing)
  the whole scale compressed and dozens of months clipped at 97–99 — the
  "always at maximum" pathology. The fixed gain is anchor-free, so it is immune
  to feature availability, and linear-in-Z keeps the top-end spacing so 2000 /
  2007 / 2021 / today differentiate instead of saturating.
* **Robust-Z features, no percentile saturation.** Every feature is a MAD-based
  trailing 240-month Z (clip ±4σ; std fallback when MAD = 0). A record-breaking
  reading moves the Z continuously instead of pinning at 100 for years.
* **F5 structure uses the 20y window + 6-month EMA pre-smooth** on one
  consistent IXIC/SPX ratio (FRED `NASDAQCOM`/`SP500`, 1971+; QQQ/SPY fallback).
  The old 3-year window made it a fast momentum gauge that whipsawed the whole
  index (z 1.2 → 3.4 → 1.2 within weeks); the long window keeps the dot-com
  extreme as the reference max, so later readings are calm and comparable.
* **FRED-backed price fallback chain.** Prices resolve via
  `FRED (SP500/NASDAQCOM/VIXCLS) → yfinance → Stooq → FRED` — so the Nasdaq and
  S&P series survive Yahoo 429s and the Stooq anti-bot wall, and the daily index
  chart is always available.
* **K-line stability filter (trend + bounded oscillation).** The daily index is
  decomposed into a slow-EMA trend (TREND_SPAN = 75d — the mid/long-term macro
  wave) plus a small, hard-capped short-term oscillation — like a stock K-line:
  a clear medium-term trend with minor daily wiggle, never a violent sawtooth,
  never an over-smoothed flat line. A **stress-aware daily clamp** (≤ 0.6 pts
  normally, ≤ 6 pts under a *stress flag*: VIX > 40 *or* a 21-day S&P drop <
  −15% *or* a BAA10Y month-over-month jump > 0.5) lets genuine risk events break
  out fast. All steps are causal (no look-ahead).
* **Event detection + guidance.** Local peaks ≥ 75 are flagged as risk climaxes
  (trim); local troughs ≤ 35 as accumulation windows (buy) — the deep-fear zone
  has historically marked the strongest forward 12–24m returns (2002-09,
  2009-03, 2020-03, 2022-10). The dashboard renders these as chart markers and a
  current-guidance panel.
* **Crash-proof, cache-first loading.** `get_monthly_scores()` never raises:
  fresh parquet cache is served instantly; a live incremental refresh is
  attempted behind a hard deadline; any failure falls back to the cache, then to
  a flagged synthetic series. The concurrent fetch catches
  `concurrent.futures.TimeoutError` explicitly (a **distinct class** from the
  builtin on Python ≤ 3.10 — the exact bug that crashed Render) and shuts the
  executor down with `wait=False, cancel_futures=True` so a hung request can
  never hold the page hostage.
* **Honest backtest metrics.** The DCA backtest reports **money-weighted return
  (IRR)**, total invested, final value, max drawdown and a contribution-stripped
  Sharpe — *not* a naive end/start equity "Total Return", which would conflate
  ongoing contributions with investment growth. The reserve-recycle mode keeps
  total invested identical to DCA, isolating the index's timing skill.
* **On-page diagnostics.** A "🔧 数据诊断" panel shows source, feature
  availability, FRED_API_KEY presence, per-series Y/N, episode calibration
  (dot-com/GFC/COVID/2021 peak scores — verifies the V3 calibration with the
  container's real data) and a live connectivity probe — so a failing deploy is
  diagnosable from the page itself.

---

## 2. Files

| File | Purpose |
|------|---------|
| `pipeline.py` | **Crash-proof concurrent** data fetchers (ThreadPoolExecutor, per-request timeout, hard total deadline, `FuturesTimeoutError` caught + non-blocking shutdown) + vectorized robust-Z scoring (V3) + **FRED-backed price fallback chain** (`SP500`/`NASDAQCOM`/`VIXCLS`) + **incremental** parquet cache (`bubble_cache.parquet`, format-versioned) + **cache-first, never-raises** resolution (cache → live → synthetic). Includes the **K-line two-timescale daily filter** (`get_daily_scores()`), **event detection** (`detect_events()`), `historical_benchmarks()` / `opportunity_benchmarks()` and `probe_connectivity()`. Public API: `get_monthly_scores()`, `get_latest_state()`, `get_daily_scores()`. |
| `backtest.py` | Buy-&-Hold vs Bubble-DCA (2000→today), monthly by default. Fully **parameterized** engine with a **reserve-recycle default mode** (same $ outflow as DCA — taper & stockpile at high risk, deploy the reserve at deep-fear lows; bands 40/50/80/95, de-risk ≥ 95). Prints **money-weighted return (IRR)**, total invested, final value, max drawdown, Sharpe + drawdown comparison for 2000/2008/2021. `run_backtest()` returns `(metrics_df, chart_fig)` for the dashboard; `main()` returns the per-side metric dicts for `report.py`. |
| `app.py` | Terminal-style Streamlit dashboard: gauge + status **badge card**, **🧭 current-guidance panel** (zone → posture, last risk climax / buying window), 5-module radar + monthly drivers, **historical comparison** (risk climaxes 🔴 vs accumulation troughs 🟢, incl. the late-2022 big-buy window), **dual-Y-axis** S&P/Nasdaq history with **event markers**, V3 risk-band shading, an **interactive backtest panel** (reserve-recycle default) with honest IRR metrics, an **on-page diagnostics panel** (source, feature availability, episode calibration, connectivity probe), and a sidebar *"Valuation acceleration"* toggle. All data loads are guarded — the page degrades to cache/synthetic with a notice instead of ever crashing. **Mobile-responsive**. |
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
* **F5 (tech froth)** uses the **20-year robust-Z of one consistent IXIC/SPX
  ratio** (FRED `NASDAQCOM` / `SP500`, 1971+; QQQ/SPY fallback), pre-smoothed
  with a 6-month EMA — capturing structural tech deviation with the dot-com
  extreme as the long-term reference, without the old 3-year window's whipsaw.
* The composite uses the **V3 fixed-gain calibration** (`score = 50 + 28 ×
  blend_z`, clip [1, 99]) — deterministic and immune to feature availability;
  all levels are data-driven and will differ from any illustrative reference
  (the live model's own values are what's plotted).
* The reserve-recycle strategy assumes cash (SHY / FRED 3M T-bill) earns its
  prevailing yield; the conservative default is 0 if neither can be fetched.
* This is a research/educational tool, **not investment advice**.
