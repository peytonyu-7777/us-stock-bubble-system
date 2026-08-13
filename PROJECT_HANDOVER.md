# US Equity Bubble Risk System — 项目交接文档（整合版）

> 写给下一个 WorkBuddy 会话的完整项目上下文。读完本文档即可独立维护、调试、升级本系统。
> 最后更新：2026-08-13 ｜ 最近一次重构：**数据一致性 / 可靠性 / 时滞性全面优化**（详见 §3.3）

---

## 目录

1. [项目是什么](#1-项目是什么)
2. [文件技术路径与上传部署](#2-文件技术路径与上传部署)
3. [当前最重要的问题](#3-当前最重要的问题接手必读)
4. [代码架构（文件地图）](#4-代码架构文件地图)
5. [关键设计决策](#5-关键设计决策改代码前必读)
6. [可调参数速查](#6-可调参数速查pipelinepy-顶部)
7. [已知陷阱](#7-已知陷阱踩过的坑务必避免)
8. [测试与验证命令](#8-测试与验证命令本地)
9. [下一步优化建议](#9-下一步优化建议按优先级)
10. [工作流建议](#10-工作流建议给新会话)

---

## 1. 项目是什么

一个**美股泡沫风险指数仪表盘**（Streamlit 应用），线上地址：
**https://us-stock-bubble-system.onrender.com/**（Render 免费版，自动从 GitHub 部署）

核心功能：
- **Bubble Risk Index（0-100）**：5 大模块加权合成（估值30% / 情绪20% / 杠杆20% / 结构15% / 宏观15%），判断美股当前泡沫风险
- **合并图**：指数日线 + S&P 500 / Nasdaq（rebased 到 100）三线对比，支持日期级选择
- **操作指引卡**：按风险区给出"3× 定投 / 1.5× / 1× / 0.5× / 0× + 现金"建议
- **策略回测**：Bubble-DCA（倍率模式默认） vs 定投基准，支持可选时间区间
- **信号系统**：sell（>78 持续 3 月）/ buy（<45 持续 2 月）及历史前向收益统计

评分方法学（V3）：每个指标做 **20 年滚动 robust-Z**（无前视偏差）→ 模块加权 → **fixed-gain 校准**到 0-100 → K-line 稳定性滤波（慢 EMA 趋势 + 应力感知日限幅）。

---

## 2. 文件技术路径与上传部署

### 2.1 本地技术路径（Windows）

```
项目根目录 : C:\Users\yuyipeng\WorkBuddy\2026-07-30-16-19-06\us-stock-bubble-system
venv python: C:\Users\yuyipeng\.workbuddy\binaries\python\envs\bubble\Scripts\python.exe
              （所有本地运行/测试都用这个解释器，已装 streamlit/plotly/kaleido）
缓存文件   : bubble_cache.parquet   （月度原始+特征+评分，构建时烘培）
             bubble_cache_meta.json  （meta：source/written_at/features）
             hf_daily.parquet        （日线 vix/spx/ndx/tnx/hyg/lqd）
Git 仓库   : https://github.com/peytonyu-7777/us-stock-bubble-system.git
              （local branch: main → origin/main 自动触发 Render 部署）
部署 runtime: Python3 native（非 Docker）—— 见 §3.1，Docker egress 被 FRED/Stooq
              屏蔽，Python3 runtime 的 egress IP 池不在黑名单（用户同账户的
              Liquidly_dashboard_cloud 用 Python3 runtime 一直能访问 FRED）
```

### 2.2 上传 / 部署方式（改完代码如何上线）

```bash
# 1) 进入项目目录
cd C:/Users/yuyipeng/WorkBuddy/2026-07-30-16-19-06/us-stock-bubble-system

# 2) 本地验证（必须先过编译 + 冒烟）
C:/Users/yuyipeng/.workbuddy/binaries/python/envs/bubble/Scripts/python.exe -m py_compile app.py pipeline.py backtest.py
C:/Users/yuyipeng/.workbuddy/binaries/python/envs/bubble/Scripts/python.exe test_benchmarks.py

# 3) 提交并推送（推送即触发 Render 自动部署）
git add -A
git commit -m "描述本次改动"
git push origin main

# 4) 验证线上健康
curl -s https://us-stock-bubble-system.onrender.com/_stcore/health   # 期望 200
```

### 2.3 关键环境变量（Render Dashboard → Environment）

| 变量 | 值 | 说明 |
|---|---|---|
| `FRED_API_KEY` | FRED 密钥 | **必须同时设为 build 和 runtime**，否则构建时 warm_cache 拿不到 FRED 数据 |
| `PORT` | （Render 注入） | 容器监听端口，不要手动改 |

> ⚠️ **FRED 屏蔽问题的根治路径**：Render Dashboard → Environment 把 `FRED_API_KEY` 同时加为 build + runtime 变量，然后 **Manual Deploy → Clear build cache & deploy**（清掉旧的 synthetic 烘培缓存，重新构建）。构建成功且网络恢复后，诊断面板 FRED 探测应显示 ✅，`source` 回到 `live`/`cache`。

---

## 3. 当前最重要的问题（接手必读）

### 3.1 问题 A：Render Docker runtime 屏蔽了 FRED（核心瓶颈，已根治）

**症状**：页面诊断面板（🔧 数据诊断 → 连通性探测）显示：

| 数据源 | 状态 |
|---|---|
| FRED 无密钥 CSV | ❌ 超时/屏蔽（~6-21s） |
| FRED API（带 key） | ❌ 返回错误 ZIP 包（PK 开头） |
| Stooq 日线 | ❌ 返回 HTML（反爬墙） |
| **yfinance chart API**（query1.finance.yahoo.com） | ✅ **可用**（~1.6s） |

**根因（已查明）**：Render 的 **Docker runtime 和 native Python3 runtime 走的是不同的 egress IP 池**。FRED/Stooq 对 Docker 的 IP 段做了限流/封禁，而 Python3 native 的 IP 段不被拦。用户同账户的 `Liquidly_dashboard_cloud` 服务用 Python3 runtime 一直能正常访问 FRED——这是关键对照。

**根治方案（2026-08-13 commit `ac91162`）**：`render.yaml` 把 `runtime: docker` 改为 `runtime: python`，新增 `buildCommand` 在构建时跑 `pipeline.warm_cache()` 把 FRED 数据烘焙进 slug，`startCommand` 启动 streamlit。Dockerfile 不再被引用，但保留在仓库里以备回退。

**部署后验证清单**：
- [ ] Render 控制台 Services 列表里 us-stock-bubble-monitor 的 Runtime 列变成 Python（不再是 Docker）
- [ ] 诊断面板 FRED 探测恢复 ✅（FRED 无密钥 CSV + FRED API 都应 ✅）
- [ ] `meta.source` 回到 `live`（不再是 resilient / synthetic）
- [ ] 估值模块用上真 CAPE/Buffett，杠杆模块用上真 FINRA MGDTE，指数质量回到 V3 验证版

如果切到 Python3 runtime 后**仍然**显示 FRED ❌，说明 Render 把 Python3 IP 池也加入了黑名单（概率很低）——届时再迁到 HuggingFace Spaces（参考 §9.2）。

**原"已实施的应对"（Docker 时代）**：
1. **resilient 模式**（`523bf83`）：检测到 `source == "synthetic"` 时自动切换到 yfinance-only 实时指数。Python3 时代如果 FRED 通了，resilient 不再被触发（逻辑只在 source==synthetic 时触发）。
| **yfinance chart API**（query1.finance.yahoo.com） | ✅ **可用**（~1.6s） |

**影响**：月度宏观特征（CAPE、MGDTE 保证金、EMVMACROBUS、GDP、M2SL、WALCL 等）**只有 FRED 有**——FRED 挂了，V3 复合分全部失效 → 曾长期降级到合成数据（页面出现黄色 synthetic 警告）。

**已实施的应对**：
1. **resilient 模式**（`523bf83`）：检测到 `source == "synthetic"` 时，自动切换到 yfinance-only 实时指数（Valuation=SPX-vs-200dMA、Sentiment=VIX、Leverage=HYG/LQD、Structure=NDX/SPX、Macro=^TNX，按同权重合成）。页面显示蓝色 info 提示，**不再出现合成数据**。注意：此模式精度低于 V3（无 CAPE/保证金/EMV）。
2. **FRED 24h 节流**（`2ba0d33`）：`CACHE_MAX_AGE_HOURS = 24`，一天最多刷新一次 FRED，防止触发限流。
3. **FRED 探测门**（`c3d8ef1`）：刷新开头 3s 探测 FRED，失败则整轮跳过 FRED 调用（不再逐个 15s×3 次超时），改用构建时烘培缓存。
4. **yfinance 优先**（`c3d8ef1`→`2ba0d33`）：^IXIC/^VIX 走 FRED（权威）、其余价格走 yfinance 优先。

**TODO（关键！）**：如果用户重新部署时把 `FRED_API_KEY` 同时设为 **build 和 runtime** 环境变量，且 Render 网络恢复，应验证：
- [ ] 诊断面板 FRED 探测恢复 ✅
- [ ] `meta.source` 回到 `live` / `cache`（不再是 resilient）
- [ ] 仪表盘数值回到 V3 复合分（resilient 只是临时替代）
- 若 FRED 恢复，**resilient 模式不会被使用**（逻辑只在 source==synthetic 时触发），无需额外处理。

### 3.2 问题 B：数据"及时性 / 一致性 / 可用性"三层现状

| 维度 | 现状 | 机制 |
|---|---|---|
| **及时性** | 价格/指数日线 → 最新交易日；月度宏观 → 最新月末 | 缓存超 24h 自动增量刷新（`get_monthly_scores`）；`get_daily_price` 自愈（hf 缓存 >2 交易日陈旧即强制刷新） |
| **一致性** | 图表/仪表盘/指引读同一 series | resilient 模式下全部用 `scores`（同一 Series）；正常模式下图表用 `get_daily_scores`（月度锚+日度体制混合），仪表盘用月度分（有意为之，见 §5.1） |
| **可用性** | 永不硬崩；FRED 挂了有 resilient 兜底；页面加载 20-40s 内完成 | 全部 fetch 带超时+重试+截止；`get_*` 均 try/except 降级 |

### 3.3 问题 C（本次已修复）：指数时滞 + 三处一致性/可靠性缺陷

**根因 1 — 致命 bug：日度体制 overlay 被算出来却从未生效。** `get_daily_scores` 里
`blended = 0.6*anchor + 0.4*regime` 计算后被丢弃，函数最终返回的是**纯月度锚**（`daily_anchor`），
日度价格/VIX 体制从未参与。这就是"泡沫指数相对价格时滞极强"的直接原因——红线只是
月度宏观分的慢 EMA，天然滞后一个月。**已修复**：overlay 现在真正 blend 进日度指数，
并统一了钳制逻辑。

**根因 2 — 仪表盘/指引与图表口径不一致。** 仪表盘/指引卡读**月度分**，图表红线读**日度混合分**，
极端行情可差几十分。**已修复**：`app.py` 现在把 gauge / 状态卡 / 指引 / 历史对照 / 合并图
全部统一到**同一个规范化日度指数**（`daily`，即图表红线）；月度宏观分保留给回测/信号/模块拆解。

**根因 3 — 可靠性：缓存写 + 重复拉取。** 并发写可能产生半写坏文件；同一次页面加载里
`load_scores` 和 `load_daily_scores` 会**重复触发整批网络拉取**。**已修复**：原子写
（temp+`os.replace`）、进程内文件锁、60s memo 去重、FRED 探测结果 5 分钟缓存。

**调参（降时滞，保稳定）**：`DAILY_ANCHOR_WEIGHT` 0.60→0.50，`BLEND_DAILY_CLAMP`
1.2→1.5，regime 的 5d/6d 激进提速被回退为 10d/10d（经实验，5d/6d 会让 regime 单日摆动
30+ 点、指数出现锯齿；真正的问题是 overlay 没生效，不是窗口不够短）。

---

## 4. 代码架构（文件地图）

```
us-stock-bubble-system/
├── app.py            # Streamlit UI 主入口（~57KB，顶部大段 CSS）
├── pipeline.py       # 核心：数据获取、特征计算、评分、缓存、信号（~113KB 最大）
├── backtest.py       # 回测引擎（run_backtest → (metrics_df, chart_fig)）
├── report.py         # CLI 报告生成
├── test_benchmarks.py# 冒烟测试
├── Dockerfile        # Render 部署：构建时 warm_cache() 烘培缓存（关键！）
├── requirements.txt  # 依赖
├── README.md         # 用户向文档
└── PROJECT_HANDOVER.md  # 本文档（维护者交接）
```

### 4.1 关键函数（pipeline.py）

| 函数 | 作用 |
|---|---|
| `get_monthly_scores(refresh, tail_boost)` | 评分入口。cache-first；超 24h 自动增量刷新；FRED 失败降级 cache/synthetic |
| `get_daily_scores(refresh, tail_boost)` | 日线指数 = 月度锚（0.60）+ 日度体制（0.40）混合，尾部钳制 |
| `compute_daily_regime(hf)` | 日度市场体制（SPX 10d 动量 + VIX + 200dMA 延伸），10d EMA 平滑 |
| `compute_resilient_index(hf)` | **yfinance-only 兜底指数**（FRED 挂时用） |
| `get_daily_price(ticker)` | 日线价格，**自愈**（陈旧>2 交易日自动触发刷新） |
| `_get_hf_daily()` | 维护 hf_daily.parquet：vix/spx/ndx/tnx/hyg/lqd 6 个日线序列 |
| `fetch_all_raw(incremental)` | 并行拉取原始数据；**FRED 探测门**（失败跳过全部 FRED） |
| `cache_info()` | 缓存写入时间/新鲜度（驱动仪表盘"最近更新"时间戳） |
| `probe_connectivity()` | 连通性探测（诊断面板） |
| `compute_composite(feat)` | 5 模块 → 0-100 复合分 |

### 4.2 关键函数（app.py）

| 函数 | 作用 |
|---|---|
| `main()` | 页面主流程（侧边栏→诊断→仪表盘→指引→模块→历史对照→合并图→回测） |
| `history_fig(scores, spx, ndx, ...)` | 合并图（指数左轴 + 价格右轴 rebase 100） |
| `guidance_panel(state, daily, monthly)` | 操作指引卡（3×/1.5×/1×/0.5×/0×） |
| `backtest_panel(scores, params, meta)` | 回测面板（日期区间可调） |
| `gauge_fig(score)` / `status_card(...)` | 仪表盘/状态卡 |

---

## 5. 关键设计决策（改代码前必读）

1. **单一规范化日度指数（本次重构后）**：
   - **仪表盘 / 指引 / 状态卡 / 历史对照 / 合并图红线** 全部读**同一个日度指数** `get_daily_scores`（= 月度宏观锚 50% + 日度价格/VIX 体制 50%，经日限幅钳制）——口径统一，永不自相矛盾
   - **回测 / 买卖信号 / 模块拆解 / drivers** 用**月度宏观分**（V3 验证版，CAPE/保证金/EMV 锚定，月度节奏天然适合回测与确认信号）
   - **不要**把日度体制权重加到让红线变成 SPX 镜像（会失去宏观意义）；调 `DAILY_ANCHOR_WEIGHT`
   - resilient 模式例外：全部用同一个 yfinance 兜底 series（保一致性）

2. **日线尾部钳制** `BLEND_DAILY_CLAMP = 1.5`：日度指数单日变化 ≤ 1.5 分，**防锯齿**（历史上锯齿问题根因：OSC 项 + 未平滑 regime + 钳制只覆盖部分区间，已全部修复；另有 overlay 被丢弃的 bug，见 §7）。调大 = 更灵敏但更锯齿。

3. **OSC_GAIN = 0**：稳定性滤波的振荡项已禁用——日度体制提供了短期变化，OSC 叠上去只产生锯齿。

4. **日线对齐**：`get_daily_scores` 的日线尾部截止到**最新日度价格收盘日**（不是中国日历"今天"），保证红线不超出价格线。

5. **FRED 是权威源但被屏蔽**：所有价格/宏观，能 yfinance 就 yfinance（不占 FRED 配额）；FRED 保留给唯一来源的月度特征 + ^IXIC/^VIX。`CACHE_MAX_AGE_HOURS=24` 控制 FRED 调用频率。

---

## 6. 可调参数速查（pipeline.py 顶部）

| 常量 | 当前值 | 含义 |
|---|---|---|
| `FETCH_TIMEOUT` | 15s | 单请求超时（原 8s→15s，慢 egress 也能成功） |
| `FETCH_DEADLINE` | 30s | 整批拉取硬截止 |
| `INCREMENTAL_DAYS` | 30 | 增量刷新窗口（只拉最近 30 天） |
| `CACHE_MAX_AGE_HOURS` | 24.0 | 缓存超过此值自动刷新（FRED 节流的关键） |
| `FRED_REFRESH_HOURS` | 24.0 | FRED 系列刷新频率上限 |
| `DAILY_ANCHOR_WEIGHT` | 0.50 | 日线指数中宏观锚的权重（1=纯宏观，0=纯 SPX） |
| `BLEND_DAILY_CLAMP` | 1.5 | 日线指数单日最大变化（防锯齿） |
| `REGIME_MOM_WINDOW` | 10 | 日度体制的 SPX 动量窗口（交易日） |
| `REGIME_EMA_SPAN` | 10 | 日度体制的 EMA 平滑跨度（≈7 日半衰期） |
| `OSC_GAIN` | 0.0 | 稳定性滤波振荡项（已禁用） |
| `MODULE_WEIGHTS` | 30/20/20/15/15 | 五大模块权重 |

---

## 7. 已知陷阱（踩过的坑，务必避免）

1. **Plotly `add_hrect` 的 `exclude_empty_subplots`**：默认 True，在添加任何 trace **之前**调用会被**静默丢弃**——必须传 `exclude_empty_subplots=False`（否则色带不显示且不报错）。
2. **Plotly `add_shape`/`add_annotation` 传 `row/col` 会覆盖 `xref="paper"`**：变成数据坐标，`x=0.985` 被当作 1970 年毫秒时间戳，**把 x 轴拖到 1960**（图表左侧大片空白）。修复：不传 row/col，用 `yref="y2"` 指到分数轴。
3. **空子图会把共享 x 轴拉到 1960**：make_subplots 里某个 row 没有任何 trace 时，autorange 算出的 x 起点 ≈ 1960。倍率模式下不渲染空现金储备行（单行布局）。
4. **Render 免费版临时磁盘**：运行时的 parquet 缓存**每次冷启动（闲置 15min 后）重置**为构建时烘培的版本 → 首次访问触发刷新。这是特性不是 bug，别试图"修"它。
5. **yfinance 在云 IP 常 429**：所有 fetch 都走"yfinance → FRED → Stooq"链，失败静默降级，**绝不抛错**。
6. **Streamlit 无 background 线程**：别用 `threading` 后台刷新，页面会崩。刷新只能靠"页面加载时同步执行"（已实现：缓存陈旧→spinner→增量刷新）。
7. **日度 overlay 被丢弃的坑（本次修复的致命 bug）**：`get_daily_scores` 曾先算 `blended = w*anchor + (1-w)*regime`，随后却 clamp 并 return 了 `daily_anchor`（纯月度锚），overlay 从未生效 → 指数滞后一个月。改这段代码时**务必确认 return 的是 `blended` 而不是 `daily_anchor`**。`test_benchmarks.py` 第 5 项回归检查会拦住这个回归（要求 2024 起 `|daily − monthly-anchor|` 均值 ≥ 1 分）。
8. **缓存写必须原子**：直接用 `df.to_parquet(path)` 会在并发/崩溃时留下半写坏文件。统一走 `_atomic_write_parquet`（temp + `os.replace`）。

---

## 8. 测试与验证命令（本地）

```bash
# venv（Windows）
C:/Users/yuyipeng/.workbuddy/binaries/python/envs/bubble/Scripts/python.exe

# 冒烟测试
python test_benchmarks.py

# 手动验证评分/缓存
python -c "import pipeline as p; s,m=p.get_monthly_scores(refresh=False); print(m.get('source'), s.dropna().index[-1])"

# 验证日线指数尾部对齐 + 单日变化
python -c "import pipeline as p; d=p.get_daily_scores(refresh=False).dropna(); print(d.index[-1]); print(d.diff().abs().max())"

# 验证 resilient 兜底
python -c "import pipeline as p; r=p.compute_resilient_index(); print(r.dropna().index[-1], r.dropna().iloc[-1])"

# 本地跑页面
streamlit run app.py
```

注意：本地网络访问 FRED 常被墙（会走 20-30s 超时降级），验证以 **Render 线上**为准。

---

## 9. 下一步优化建议（按优先级）

1. **【验证】FRED 恢复后的 V3 分**：部署时设好 build FRED_API_KEY，检查诊断面板 FRED ✅，确认 `source=live`，对比 resilient 与 V3 数值差异。（本次已把一致性/时滞/可靠性修好；FRED 屏蔽仍是外部基建问题，见 §3.1）
2. **【数据质量】CAPE / 保证金缺口的替代源**：本地缓存 `cape/wilshire/mgdte` 常为 N（FRED 被墙），估值锚退化到 SPX-200wMA 代理、杠杆锚退化到 SPX12m/BAA 代理——精度低于 V3 全量。可调研 multpl.com 的 CAPE 镜像或其它非 FRED 源作为 F1 的第三层兜底。
3. **【体验】回测面板的日期选择**：现在回测用年 select_slider（与合并图的日期选择器不一致），可统一为 date_input + 预设按钮。
4. **【可用性】resilient 模式的历史标注**：图表 hover 提示区分"V3 验证分"与"resilient 兜底分"，避免用户误读。
5. **【性能】合并图 13300+ 日点**：Plotly 全量渲染在低端设备可能卡，可考虑降采样（>5000 点时抽稀）。

---

## 10. 工作流建议（给新会话）

1. **用户报"数据没更新/不一致"** → 先看线上诊断面板连通性探测 + 数据时间线 → 判断是 FRED 屏蔽、yfinance 429、还是缓存冷启动。
2. **改 chart 前**先看 §5 设计决策和 §7 陷阱，避免重蹈 1960/锯齿/色带消失。
3. **每次改完**：py_compile 全过 + `test_benchmarks.py` + 本地起页面截图关键面板 + git commit/push + curl 健康检查。
4. **记录**：每天的实质工作追加到 `.workbuddy/memory/2026-08-*.md`（已有完整历史），关键决策写进本文档 §5。
