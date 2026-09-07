# swingbot

A broker-independent **weekly swing-trading** research and signal system for Indian (NSE)
and US equities, with news-assisted return prediction.

No intraday. Decisions are made once a week at the close, filled at the next open, and
held for a configurable number of sessions (five by default).

---

## What this is, and what it is not

This is not a bot that promises returns. It is the machinery that tells you honestly
whether a weekly signal is real: point-in-time correctness, purged cross-validation,
realistic transaction costs, and null-model baselines you have to beat before believing
anything.

Generating a signal is the easy part. Knowing whether it survives costs, overlapping
labels, survivorship bias and your own multiple testing is the hard part, and that is
what most of this codebase is.

**It never touches a broker.** The core computes target positions and writes order files.
A single protocol, `ExecutionAdapter`, is the only place a broker could ever attach, and
a test fails the build if anything else imports a broker SDK.

---

## The 90/10 design: what the AI does

An LLM asked to predict a price is doing the thing it is worst at, and the result cannot
be backtested honestly because the outcome of any historical article is already inside
the language model's training data. A backtest over past news would measure memorisation,
not edge.

So the forecast is an explicit two-block blend:

```
score = 0.90 * rank(price_model.predict(price + technical + chart features))
      + 0.10 * rank(news_model.predict(news features))
```

- The **price block** does the real work. Momentum, mean reversion, volatility, and chart
  structure across the whole universe, trained on forward returns.
- The **news block** is a language model reading articles and emitting typed facts:
  event type, sentiment magnitude, expected direction, confidence, affected entity, and
  whether the item is speculative or a rehash. Text in, JSON out. No prices in the
  prompt, no knowledge of what happened afterwards.

Keeping the blocks separate is what makes the split auditable. The report prints each
block's realised contribution, and the ablation measures whether news earns its 10
percent at all.

### Not bound to any one AI provider

The news layer is a protocol with four interchangeable backends, chosen in config:

| Backend | Needs a key | Notes |
| --- | --- | --- |
| `lexicon` | no | Financial sentiment lexicon, offline, the default |
| `openai_compat` | yes | Any OpenAI-shaped endpoint: OpenAI, Groq, Together, OpenRouter, or a local Ollama or LM Studio server via `base_url` |
| `anthropic` | yes | Claude |
| `finbert` | no | Local transformer, optional extra, fully offline |

All four return the same `NewsAnalysis` object, so swapping one never touches anything
downstream. **The system works out of the box with no API key.**

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# optional extras
pip install -e ".[gbdt]"        # LightGBM, recommended
pip install -e ".[yahoo]"       # live price download
pip install -e ".[anthropic]"   # Claude news backend
pip install -e ".[openai]"      # any OpenAI-compatible backend
```

Without LightGBM the system falls back to scikit-learn's gradient booster automatically.

---

## Quick start

Everything below runs offline on generated data, so you can see the whole pipeline work
before wiring up a data source.

```bash
# 1. Generate a synthetic market with a known, injected signal
swingbot demo --market us --years 8

# 2. Build features and labels
swingbot featurize --market us

# 3. Walk-forward train and backtest, with every null model for comparison
swingbot backtest --market us --ablation

# 4. This week's target book, as order files
swingbot signal --market us

# 5. The report
swingbot report --market us --open
```

`swingbot doctor` checks the environment, data coverage, calendar sanity and cache state.

### Real data

```bash
swingbot fetch prices --market india --start 2015-01-01
swingbot fetch news   --market india --lookback 30d
swingbot analyze news --market india --backend lexicon
```

Price providers are tried in order and fall back: `csv`, `yahoo`, `stooq`, `synthetic`.
To use your own data, drop CSVs with columns `session,open,high,low,close,volume` into
`data/<market>/csv/<TICKER>.csv` and the `csv` provider picks them up. That path always
works, whatever your broker or vendor.

---

## The weekly loop

```bash
swingbot run-weekly --market india
```

Fetches the incremental week, analyses new articles, builds the feature row, loads the
pinned model, constructs the target book, diffs it against current positions, writes
`orders.csv` and `targets.json`, sends Telegram and email, and regenerates the report.

Schedule it after Friday's close:

```cron
# 17:00 IST Friday
0 17 * * 5 cd /path/to/swingbot && .venv/bin/swingbot run-weekly --market india
```

It refuses to run on stale data, on a feature-schema mismatch, or while the drawdown
kill-switch is armed.

---

## Rigor, concretely

This is the part that matters.

**Point in time.** Every row carries `available_at`. A `PITGuard` wraps every feature
transformer and raises if any input is newer than the decision time. News availability is
`max(published_at, first_seen_at)`, so a source that backdates its stamps cannot hand the
backtest free information. An article stamped exactly at the cutoff is excluded, and that
boundary has its own test.

**The look-ahead regression test.** For sampled decision dates, features are built from
the full panel and again from a panel physically truncated after that date, then asserted
identical. It is parametrised over every registered transformer individually. A
full-sample scaler, a rank across all dates, a retroactively adjusted price, or a
forward-fill that crosses the cutoff all fail it. Transformers are auto-discovered, so a
leaky feature cannot be added unnoticed.

**Leak canaries.** Shuffled labels must produce an information coefficient near zero. A
deliberately cheating variant, news shifted one week forward, must score materially
higher than the honest one, proving the news path is live rather than inert. The honest
variant must score above zero, proving the pipeline is not silently broken.

**Purged walk-forward.** Every sample carries its label span. Training samples whose span
overlaps the test window are dropped, and an embargo is applied afterwards because
features are serially correlated and leak backwards too. Overlapping labels get
uniqueness weights so the same week is not counted five times.

**Costs are per-market and real.** India charges brokerage, STT both sides, exchange
fees, stamp duty and GST, and prices its shorts as single-stock futures with roll and
carry. The US charges commission, regulatory sell fees and stock borrow. Both add a
half-spread estimated from the stock's own high-low range and a square-root impact term
scaled by participation in average daily volume. The report includes a sensitivity sweep
at 0.5x, 1x, 2x and 3x. A strategy that dies at 2x is not real.

**Shorting is modelled honestly.** NSE cash-segment delivery cannot be held short
overnight, so the India profile expresses shorts as single-stock futures and says so in
every report. Set `short_instrument: none` for long-only.

**Survivorship.** The universe is a membership file with start and end dates that retains
delisted names, and a delisting books a terminal loss. If you supply only a current
snapshot, the report is stamped survivorship-biased rather than quietly inflated.

**Null models.** `swingbot backtest --ablation` runs the whole matrix in one command:
benchmark buy-and-hold, equal-weight, flat, random-sign at matched turnover, shuffled
labels, momentum-only, price-only, price-plus-news, news-only. The headline number for
whether news helps is the Sharpe *difference* between price-plus-news and price-only with
a bootstrap confidence interval, never the absolute Sharpe of the full model.

**Multiple testing.** A trial ledger records every backtest ever run with its config hash
and result. Deflated Sharpe is computed against that real count, so the configurations
that did not work cannot be quietly forgotten.

**Risk.** Inverse-volatility sizing scaled to a target portfolio volatility, a fractional
Kelly cap, hard per-name and sector caps, gross and net exposure limits, a no-trade band
that skips small rebalances, and a drawdown kill-switch that halves gross at 8 percent
and flattens at 15 percent. The kill-switch runs inside the backtest loop, so its cost is
measured rather than assumed.

---

## Layout

```
config/          layered YAML: base, per-market, per-profile, universe membership
src/swingbot/
  config.py      layered config, secrets from env only, reproducibility hash
  pit.py         the point-in-time firewall
  calendars.py   the weekly decision grid
  data/          price providers, PIT universe, corporate actions, resampling
  news/          RSS ingest, dedupe, the four analyzer backends
  features/      price, technical, chart patterns, cross-sectional, news, labels
  model/         ridge, gradient boosting, baselines, the 90/10 blender
  validation/    purged walk-forward, metrics, information coefficient, deflated Sharpe
  portfolio/     sizing, long-short construction, risk limits, capacity
  backtest/      the weekly fill engine, per-market costs, ablation
  report/        HTML tearsheet
  execution/     ExecutionAdapter, CSV and JSON output, paper book
tests/           look-ahead, leak canaries, purge correctness, cost, limits, architecture
```

---

## Configuration

Layers, later wins:

```
config/base.yaml -> config/markets/<market>.yaml -> config/profiles/<profile>.yaml
                 -> environment -> --set
```

```bash
swingbot backtest --market india --set portfolio.n_long=10 --set model.blend.news_weight=0.2
SWINGBOT__PORTFOLIO__TARGET_VOL_ANNUAL=0.10 swingbot backtest --market us
```

Secrets come from the environment only and never enter a config file or the config hash:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`SMTP_USERNAME`, `SMTP_PASSWORD`.

---

## Expectations, stated up front

With roughly 500 weekly observations and cross-sectionally correlated names, the
effective sample is small. A gross information coefficient of 0.02 to 0.04 is a *good*
result at this horizon, and costs plus turnover can consume most of it. That is exactly
why the no-trade band, the cost sweep, the price-only denominator and the deflated Sharpe
exist rather than being bolted on at the end.

Nothing here is investment advice. Run it on paper for a long time before it touches
money.

## Licence

MIT.
