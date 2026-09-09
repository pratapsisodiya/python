# swingbot

A **weekly swing-trading** signal system for Indian (NSE) and US equities. It reads price
history and news, and once a week tells you what to buy and sell. It does not place trades
for you, does not connect to your broker, and never needs your broker password or API keys.
You take its output and place the trades yourself, on whatever platform you already use.

No intraday. It decides once a week (Friday close), you trade the next session's open, and
you hold for about a week before the next signal replaces it.

If you've never used a command line before, this doc explains everything from scratch.

---

## What this actually does, in plain terms

Once a week, you run one command. It looks at:

- **Price history** — how each stock has been trending, its momentum, volatility, and
  chart patterns (moving averages, RSI, breakouts, and so on).
- **Recent news** — headlines about each company, scored for whether they're good or bad
  news, and how big a deal they are.

It combines the two — about 90% weight on price/chart signals, 10% on news — and picks a
handful of stocks to go long (buy) and, optionally, a handful to go short (bet against). It
writes that list to a plain CSV file. **You** read that file and place the trades on your
own broker's app or website. The program never touches your money or your account.

**Why isn't it fully automatic?** Because full automation to a live broker is a much bigger
trust step than most beginners should take with a system they just downloaded. This gives
you the signal and a clear checklist; you stay in control of every trade.

---

## Before you start: read this

- **This is not investment advice**, and it is not a guaranteed money-maker. It is a
  structured way to generate and check trading ideas. Treat every number it prints with
  suspicion until you've watched it on paper for a while.
- **Start on paper.** Run it for several weeks writing down what it would have told you to
  do, without placing real trades, so you can see whether you'd have followed it and how it
  behaves.
- **Nothing here needs your broker login, password, or API key.** If a tool ever asks you
  to type your broker credentials into this system, that is not this project — stop and
  check.
- **Weekly, not daily.** If you're expecting to watch a screen all day, this isn't that. You
  check it once a week, place a handful of trades, and leave it alone until next week.

---

## What you need before you begin

- A computer (Mac, Windows, or Linux) you can install software on.
- **Python 3.11 or newer.** Check with `python3 --version` in a terminal. If you don't have
  it, install it from [python.org](https://www.python.org/downloads/) — on Windows, tick
  "Add Python to PATH" during install.
- **Git**, to download the code. Check with `git --version`. Get it from
  [git-scm.com](https://git-scm.com/downloads) if needed.
- A brokerage account with **any** broker — this system doesn't care which one.
- About 15 minutes for the first-time setup.

Every command below is typed into a terminal (Terminal on Mac, PowerShell or Command
Prompt on Windows, any shell on Linux) — not into the code editor, not into your browser.

---

## Step 1: Get the code and install it

```bash
git clone https://github.com/pratapsisodiya/python swingbot
cd swingbot

# Create an isolated Python environment so this doesn't interfere with anything else
# on your machine, and activate it.
python3 -m venv .venv
source .venv/bin/activate          # on Windows (PowerShell): .venv\Scripts\Activate.ps1

# Install swingbot and its dependencies
pip install -e ".[dev,gbdt]"
```

`.venv/bin/activate` needs to be run again every time you open a new terminal window to
work on this. You'll know it worked because your terminal prompt will show `(.venv)` at
the start of the line.

Check it installed correctly:

```bash
swingbot doctor --market us
```

This prints a checklist — installed packages, data status, cost assumptions — and should
run without errors even before you've added any data.

---

## Step 2: See it work, with no real data (5 minutes)

Before you touch real money or real market data, run the whole pipeline on made-up data so
you can see what each step produces. This is entirely offline and completely safe —
nothing here is a real prediction, it's a demonstration.

```bash
# Generates a fake market and fake news with a made-up pattern baked in, then confirms
# the system can find that pattern. This step can take a few minutes.
swingbot demo --market us --years 8

# Backtests the strategy against several "dumb" comparisons (a coin flip, holding
# everything equally, etc.) so you can see whether it's actually better than nothing.
swingbot backtest --market us --ablation

# Generates this week's fake target trade list.
swingbot signal --market us --no-notify
```

That last command writes files into a new folder under `runs/` — look for
`runs/<timestamp>-us-signal-.../orders.csv`. Open it in Excel, Google Sheets, or a text
editor. That CSV is exactly the file format you'll use with real money later, just built
from fake data. Get comfortable reading it now.

Also open the `tearsheet.html` file from the `backtest` step in a web browser — it's a
plain HTML file, just double-click it. It shows charts and a table comparing the strategy
against simple baselines, and a plain-English verdict at the top telling you whether the
result looks real or looks like noise.

---

## Step 3: Point it at a real market

Two markets are supported out of the box: `us` (US stocks) and `india` (NSE stocks). Pick
one with `--market us` or `--market india` on every command (`in` and `nse` also work for
India).

### India: it fetches real NSE data by itself

Nothing to download. The India profile talks to an Indian broker's public price API, so
this just works:

```bash
swingbot instruments --market india    # lot sizes and ISINs from the exchange list
swingbot fetch --market india          # ~8 years of real NSE daily history
swingbot backtest --market india --ablation --sensitivity
```

The first command is worth running before anything else on India. It fetches the F&O
**lot sizes**, and those decide whether you can short at all — see the warning below.

### Other markets: bring your own CSVs

The simplest and most reliable way, whatever broker or data source you have access to, is a
CSV file per stock:

1. Export or download daily price history for each stock you're interested in. You need
   at least these columns: **date, open, high, low, close, volume.** Almost any broker,
   data vendor, or free source (Yahoo Finance's historical-data export, for example) can
   give you this as a CSV or spreadsheet.
2. Save one file per stock at `data/<market>/csv/<TICKER>.csv` — for example
   `data/us/csv/AAPL.csv` or `data/india/csv/RELIANCE.csv`. The ticker in the filename
   must match the ticker in `config/universe/sp500.csv` (US) or
   `config/universe/nifty200.csv` (India).
3. You need **at least 2 years** of daily history per stock for the models to have enough
   to learn from; more is better.

Then build features and run a real backtest:

```bash
swingbot backtest --market us --ablation --sensitivity
```

Read the verdicts printed at the end before doing anything else. If it says the pipeline
looks like it's leaking information from the future, or that the strategy doesn't beat
simple momentum, or that costs eat all the profit — believe it. That is the entire point
of this system: it is built to tell you when an idea doesn't hold up, not to talk you into
trading it anyway.

### India: how much money you need before you can short

This one surprises people, so it is worth knowing before you start rather than after a
rejected order.

On NSE you cannot hold a short in the cash segment overnight. A weekly short therefore has
to be a **single-stock future** — and futures trade in fixed lots set by the exchange, not
in single shares. Those lots are big, and they vary enormously: RELIANCE is 500 shares a
lot, IOC is 4,875.

One lot of the *cheapest* F&O name in the Nifty 200 is around **₹3 lakh**. The system caps
any single position at 12% of your account. Put those together:

| Your account | Shorts you can actually place |
| --- | --- |
| ₹5,00,000 | **none at all** |
| ₹25,00,000 | 1 or 2 |
| ₹40,00,000 | about 7 — enough for the strategy as configured |
| ₹1,00,00,000 | essentially the whole list |

Below roughly ₹36 lakh the model will still *tell* you to short things, and not one of
those orders can be placed. So run it long-only until your account is big enough:

```bash
swingbot signal --market india --set market_profile.short_instrument=none
```

`swingbot doctor --market india` prints your own numbers for this, and the weekly signal
explains any position it had to drop, with the arithmetic.

**About the included stock lists:** `config/universe/sp500.csv` and
`config/universe/nifty200.csv` are provided so you have something to start with, but they
are *today's* company lists, not history of which companies were in the index at each
past date. That makes a backtest look a little better than reality would have been,
because it silently excludes companies that later went bankrupt or got removed. The
tearsheet says this explicitly ("survivorship-biased"). It doesn't stop you from using the
system, but don't take the backtest numbers as gospel until you fix this.

---

## Step 4: Get your weekly trade list, and place it with your broker

This is the part you'll actually do every week, and the part that connects to your trading
platform. Run this after the market closes on Friday (or your market's last trading day of
the week):

```bash
swingbot signal --market us --equity 50000 --no-notify
```

`--equity` is roughly how much money (in your market's currency) you want this strategy to
manage — the position sizes scale to it. Adjust it to a small, comfortable amount while
you're starting out; you are not required to size positions to your full account.

This prints a table of positions and writes an `orders.csv` file — the path is printed at
the end, something like `runs/20260907-us-signal-.../orders.csv`. Open it. It looks like
this:

| ticker | side | quantity | order_type | instrument | est_price | est_value |
|---|---|---|---|---|---|---|
| AAPL | buy | 34 | market | equity | 190.50 | 6,477.00 |
| XYZ | sell | 12 | market | equity | 88.10 | 1,057.20 |

Here's what each column means, in plain terms:

- **ticker** — the stock's symbol, exactly as your broker lists it.
- **side** — `buy` to open or add to a long position, `sell` to close a long or open a
  short.
- **quantity** — how many shares. Already rounded to a whole share count for you.
- **instrument** — almost always `equity`, meaning a normal stock trade. On the India
  market a short position shows `futures` instead — see the callout below, this changes
  *where* you place that trade.
- **est_price / est_value** — the price used to size the trade and the approximate money
  amount. Your broker's actual fill price may differ slightly; that's normal and expected.

### Placing these trades with your broker

Every broker is different, but they all support one of these two ways, and this file works
for both:

**Option A — bulk / basket order upload.** Many brokers let you upload a spreadsheet of
orders instead of clicking through each one (sometimes called a "basket order", "bulk
order", or "bracket upload"). Look for this in your broker's order screen or app settings.
Their upload almost certainly expects different column names or an order than
`orders.csv` uses, so open both `orders.csv` and your broker's template side by side in a
spreadsheet, and copy the ticker/side/quantity values across into your broker's expected
columns. Check your broker's help pages for "bulk order upload" or "basket order CSV" to
find their exact template.

**Option B — enter each trade by hand.** This always works, on every broker, with no setup.
Open `orders.csv`, and for each row: open your broker's trade screen, enter the ticker,
choose buy or sell, type in the quantity, and place a market order (or a limit order near
`est_price` if you prefer more control over the fill price). With 10–15 positions this
takes a few minutes.

Either way, place the trades as close to the next session's open as you reasonably can —
that's the price the whole system's math is built around.

> **Tip.** The browser extension (see Step 5) turns this list into a checklist next to
> your broker's tab, with copy buttons for the ticker and the quantity. It makes the
> one-by-one route considerably less error-prone than reading numbers off a terminal.

### If you see `instrument: futures` (India only)

Indian stock exchanges don't allow you to hold a plain stock (equity) short overnight — you
can only sell a stock short if you already own it, or close it the same day. So when the
strategy wants to bet *against* a stock for a week, it has to do that using that stock's
**futures contract** instead of the plain stock. This shows up as `instrument: futures` with
a negative quantity/short side in your target book.

To place this trade, you need **F&O (futures & options) trading enabled** on your account
— it's a separate permission from plain stock trading that most Indian brokers require you
to activate (usually a quick form plus meeting an income/net-worth eligibility check). Then
place the trade on your broker's **futures** order screen for that stock, not the regular
equity screen — searching the ticker there will show you the current-month contract.

If you don't want to deal with futures at all, that's completely fine — just skip every row
where `instrument` is `futures`, or turn shorting off entirely so the strategy only ever
gives you plain stock buys:

```bash
swingbot signal --market india --set market_profile.short_instrument=none
```

### Keeping track from week to week

Every time you run `swingbot signal`, it remembers what it told you to hold last time (in
`runs/<...>/positions.json`) and only tells you what *changed* — new positions to open,
old ones to close, and any resized. It assumes you actually placed last week's trades. If
you skipped one, or your broker filled you at a very different price, place a comment in
your own notes; the file just assumes you followed through.

---

## Step 5: the web dashboard (optional, but much nicer)

Everything above works from the terminal. If you'd rather *look* at it — a page you open,
with the week's orders as a checklist, buttons to run things, and the honesty checks laid
out — there's a small web app included.

It runs **on your own computer**. There is no account, no cloud, and no broker connection.

```bash
# One extra install, for the web parts only
pip install -e '.[web]'

# Start it
swingbot serve --market us
```

Then open <http://127.0.0.1:8765> in your browser. Five tabs:

| Tab | What's on it |
| --- | --- |
| **This week** | The latest signal: orders to place with a tick box for each, the target book, and the portfolio's own notes (a risk limit that bound, orders trimmed for liquidity, and so on). |
| **Run it** | Buttons for the demo, a backtest, this week's signal, and the weekly refresh. The pipeline's log streams live while it works, so a slow backtest shows you what it's doing. |
| **Honesty checks** | The verdicts from the most recent backtest — the leak check, the "does news actually help" comparison, the deflated Sharpe, the overfitting probability. **Read this tab before you trust any number on the others.** |
| **Costs** | What your orders really filled at, against what the backtest assumed they would. Type the fill price into the **This week** table and this tab pools it across weeks. It is the only place the system checks itself against reality rather than against its own model — and since the cost assumption is what decides whether this strategy makes or loses money, it is worth the ten seconds a week. It refuses to draw a conclusion from fewer than four weeks. |
| **Run history** | Every run you've ever done, with a link to its report and a one-click copy of the `--use-model` command that reproduces it exactly. |
| **Health** | Where your prices came from and how many of them nobody actually observed, data coverage, the trading calendar, costs per round trip, how much capital shorting needs before it's even possible, which AI backend is set, and the survivorship-bias warning if it applies. |

Ticking an order off marks it in the run folder, so the record of what you actually placed
lives alongside everything else about that week.

**A note on the address.** It binds to `127.0.0.1`, which means only your own computer can
reach it. That matters: the page has buttons that run code, and there is no login. If you
change `--host` to anything else, the command tells you so — don't, unless you know exactly
why you're doing it.

### The browser extension

There's also a small browser extension in `extension/` that puts the same order checklist
in your toolbar, next to your broker's tab. Click **copy** on a row and it gives you the
ticker, click again and it gives you the quantity — which is exactly what an order ticket
asks for, in that order. Tick the row when it's placed.

See [`extension/README.md`](extension/README.md) for the two-minute install. It reads from
the dashboard above, so `swingbot serve` needs to be running (there's a paste-the-CSV
fallback if it isn't).

It deliberately does **not** type into your broker's page for you. That would need
permission to run code on live brokerage sites, it would break whenever a broker changed
its layout, and a bug in that code would place a wrong trade with real money. Copying two
fields is a few seconds slower and cannot misfire.

---

## The weekly habit

Once you're comfortable with the manual steps above, this is the whole loop, every week:

```bash
swingbot run-weekly --market us
```

This refreshes prices, pulls in the latest news, and runs `signal` in one step. Do this
once, after the close, on your market's last trading day of the week. Or press **Run
weekly** in the dashboard, which does the same thing.

### Letting your computer do it

Running it by hand for the first several weeks — and reading the output each time — is the
better habit while you're learning how it behaves. Once you trust it, hand it to your
operating system's scheduler.

**Mac or Linux** (`crontab -e`). This example runs at 17:15 on Fridays; adjust for your
market's close and your own timezone:

```cron
15 17 * * 5 cd /path/to/swingbot && /path/to/.venv/bin/swingbot run-weekly --market us >> ~/swingbot-weekly.log 2>&1
```

**Windows** — Task Scheduler, "Create Basic Task", weekly on Friday, action "Start a
program":

```
Program:   C:\path\to\.venv\Scripts\swingbot.exe
Arguments: run-weekly --market us
Start in:  C:\path\to\swingbot
```

Two things worth knowing. The scheduler only runs while your computer is on and awake, so
a laptop shut in a bag on Friday evening won't produce a signal. And a scheduled run still
only *writes files* — it emails or messages you (if you configured that) and waits. Nothing
places a trade.

---

## Safety checklist before you use real money

- [ ] Ran the demo (Step 2) and understood what `orders.csv` and the tearsheet show.
- [ ] Backtested on real historical data for your market (Step 3) and read the verdicts —
      especially the "leak check" and "does news help" lines.
- [ ] Understand the survivorship-bias warning and, ideally, replaced the universe file
      with one that includes delisted companies before trusting the numbers.
- [ ] Paper-traded (wrote down, but didn't place) at least 4–6 weeks of signals and
      compared them to what actually happened.
- [ ] Started with a small `--equity` amount, well below your full account.
- [ ] Understand that a short position on India shows as `futures` and needs F&O trading
      enabled, or have turned shorting off.
- [ ] Never given this system, or anyone claiming to be it, your broker password or API
      key.
- [ ] If using the dashboard, left it on `127.0.0.1` — it has no login, and it can start
      jobs on your machine.

---

## Everything else (for when you want to go deeper)

The sections below are for understanding *why* the system is built the way it is, and how
to change its settings. You don't need any of this to place your first trade.

### It never touches a broker — by design

The core computes target positions and writes plain files. A single interface,
`ExecutionAdapter`, is the only place a broker connection could ever attach, and an
automated test fails the project's build if any other part of the code even imports a
broker library. That's not a policy, it's enforced by a test every time the code changes.

### The 90/10 design: what the AI does and doesn't do

An AI model asked to predict a stock's price directly is doing the thing large language
models are worst at, and you couldn't trust a backtest of it anyway — the model may already
"remember" what happened to a stock after an old news article, from its own training data.

So the forecast is two separate pieces added together, not one model looking at everything:

```
score = 0.90 * rank(price_model.predict(price + technical + chart features))
      + 0.10 * rank(news_model.predict(news features))
```

- The **price piece** does the real work — momentum, mean reversion, volatility, chart
  structure — trained on what actually happened next, historically.
- The **news piece** is a language model reading articles and pulling out structured facts:
  what kind of event this is, how positive or negative, how big a deal, whether it's
  confirmed or just a rumour. Text in, structured facts out — never a price prediction.

The report always shows how much each piece actually contributed, and the backtest
specifically checks whether the news piece is earning its 10% weight or just adding noise.

### Not locked to any one AI provider

The news-reading step works with any of four interchangeable options, picked in config:

| Backend | Needs an API key | Notes |
| --- | --- | --- |
| `lexicon` | no | Built-in financial word list, fully offline — **this is the default** |
| `openai_compat` | yes | Any OpenAI-style API: OpenAI itself, or a local model server like Ollama |
| `anthropic` | yes | Claude |
| `finbert` | no | A local, free sentiment-reading model, fully offline |

The system works immediately with no API key at all, using the built-in word list. The
other options can read news more subtly, at the cost of needing an account and key.

### Layout

```
config/          settings: overall defaults, per-market rules, stock lists
src/swingbot/
  config.py       settings loader
  pit.py          the "no looking into the future" safety check
  calendars.py    the weekly decision schedule
  data/           price loading, stock list handling, corporate actions
  news/           news fetching, de-duplication, the four AI reading options
  features/       price/chart/technical signals, news signals, labels
  model/          the prediction models and the 90/10 combiner
  validation/      the "is this actually working" statistical tests
  portfolio/      turns predictions into position sizes, applies risk limits
  backtest/       the historical simulation engine and cost model
  report/         the HTML report you open in a browser
  execution/      writes orders.csv / targets.json — the only broker-facing part
  service.py      the commands themselves; both the terminal and the web app call these
  web/            the local dashboard (optional install)
extension/        the browser extension for placing orders
tests/            the automated checks that catch bugs like this
```

The `service.py` layer is worth one sentence: the terminal and the dashboard are both thin
skins over it, so a number shown on the web page and the same number printed in a terminal
came out of one function rather than two implementations that happen to agree today.

### Configuration

Settings are layered — later ones win:

```
config/base.yaml -> config/markets/<market>.yaml -> config/profiles/<profile>.yaml
                 -> environment variables -> --set on the command line
```

Example — trade more/fewer stocks, or change the news weight, without editing any files:

```bash
swingbot backtest --market india --set portfolio.n_long=10 --set model.blend.news_weight=0.2
```

API keys and any other secrets are read only from your terminal's environment variables,
never written into a settings file:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`SMTP_USERNAME`, `SMTP_PASSWORD`.

### Why the rigor matters — the checks this system runs on itself

Generating a buy/sell list is the easy part of a project like this. The hard part —
and most of this codebase — is checking honestly whether that list means anything:

- **No looking into the future.** Every price and news item is timestamped, and the system
  physically cannot see anything dated after the moment it's making a decision for. This
  is checked by an automated test that rebuilds every signal with future data deleted, and
  confirms the answer doesn't change.
- **Fake-signal checks.** The system is tested against data with the "correct answer"
  scrambled, and it must find nothing. It's also tested against data with a fake, known
  pattern deliberately inserted, and it must find that. Both checks currently pass.
- **Realistic trading costs.** Brokerage, taxes, and the cost of your own trade moving the
  price are all modelled per-market, not ignored. The report also shows what happens if
  real costs turn out to be two or three times the assumption — a strategy that only
  works at today's exact assumed cost isn't a real strategy.
- **Compared against doing nothing clever.** Every backtest also runs a coin-flip
  portfolio, a buy-everything-equally portfolio, and a couple of other "dumb" baselines,
  so you can see whether the strategy is actually earning its complexity.
- **Honest about small sample sizes.** A few hundred weeks of history isn't a lot of data,
  statistically, and the report's confidence intervals and significance numbers reflect
  that rather than overstating certainty.
- **Honest about *choosing*, too.** Run `swingbot backtest --pbo` and the report adds a
  "probability of backtest overfitting" number. It re-runs the history many different
  ways, and on each one picks whichever model setup looked best, then checks where that
  choice actually landed. Above 0.50 means picking the best-looking setup is worse than
  picking at random — a genuinely common outcome, and one most backtests never check for.
- **Every setting has to prove it does something.** A separate test takes each setting in
  the config file, runs the pipeline twice with two different values, and fails the build
  if the output is identical. A setting that quietly does nothing is worse than no setting,
  because you'll believe you're protected by it. This test found four controls that were
  documented, configurable, and completely inert.
- **The dashboard cannot grow a trade button.** A test reads the web code and fails the
  build if it so much as imports the part of the system that talks to a venue. That's why
  the web app needs no broker password: there is no code path that could use one, and it
  stays that way by force rather than by good intentions.

None of this guarantees profit. It's the difference between a system that can tell you
"this doesn't work, don't trade it" and one that will always say yes.

### Explaining a trade after the fact

Every `swingbot signal` run saves the exact model that produced it into its run folder,
alongside the settings and the data fingerprints. Months later you can re-run that same
week with the same model instead of a freshly fitted one:

```bash
swingbot signal --market us --use-model 20260904T163000-us-signal-3f2a1b8c
```

That reproduces the original book exactly rather than approximately, which is the
difference between explaining a past trade and guessing at it. The command refuses if the
saved model's features no longer match the current ones, or if the model is older than
`model.max_model_age_weeks` — reproducing an old decision is fine, but trading this week
on a year-old fit should have to be asked for out loud.

### What it actually did on real Indian data

Not a demo. 272,615 real NSE daily bars, 128 Nifty 200 names, 452 weeks from January 2018
to September 2026, fetched by the command in Step 3.

| What you'd do | Sharpe | Return a year |
| --- | --- | --- |
| Just hold the whole list, equally | **+0.98** | **+18.3%** |
| This strategy, long-only | +0.54 | +3.4% |
| Plain momentum, no machine learning | +0.91 | +5.9% |
| This strategy, long **and** short | −0.14 | −0.9% |
| Coin flip at the same turnover | −0.97 | −3.7% |

**Buying the whole list and doing nothing beat everything else.** The system says so
itself, in the verdicts it prints:

> The model (Sharpe 0.54) does not beat plain momentum and reversal. The added complexity
> is not earning anything.
>
> The strategy (Sharpe 0.54) does not beat simply holding the universe (Sharpe 0.98).
>
> Deflated Sharpe: does not survive deflation; likely selection, not skill.

Two things are worth understanding about *why*, because they are not the same as "the idea
was stupid".

**The signal is genuinely there. Costs eat it.** The model's predictions have an
information coefficient of +0.026 with a t-statistic of 3.4 over 296 out-of-sample weeks —
a small but statistically real ability to rank stocks. The leak check passes, so this is
not the harness fooling itself. The problem is arithmetic:

```
gross return   +7.4% a year
trading costs  -8.3% a year   (49% of the book turned over every week)
net            -0.9% a year
```

An edge this size cannot pay for weekly turnover at Indian cash-segment charges. At *half*
the assumed cost it makes +0.29 Sharpe; at double it makes −1.20. A strategy that lives
entirely inside its cost assumption is not one to trade.

**Shorting a rising market cost the rest.** The Nifty compounded at 18% a year over this
period. Being short anything into that is a headwind the model has to overcome before it
earns a rupee. Long-only takes the Sharpe from −0.14 to +0.54 — and it is also the only
version most accounts can place at all, for the lot-size reason above.

**So: don't trade this.** Not yet. What it is good for right now is the machinery — a
pipeline that fetches real data, respects point-in-time correctness, models real costs, and
tells you honestly when an idea does not work. That last part is the hard part, and it is
working exactly as intended: it just told you not to trade its own strategy.

If you want to make it work, the numbers say where to push: cut turnover hard (the
`no_trade_band` and `n_long`/`n_short` settings), or find a signal several times stronger.
Fishing through settings until one shows a positive Sharpe is exactly what the deflated
Sharpe and the PBO check exist to catch you doing.

### Expectations, honestly

With a few hundred weeks of data and stocks that move together to some degree, the
effective amount of independent information is small. A "good" result at this timeframe
looks modest by design, and trading costs can eat a meaningful chunk of any edge. That's
why the safety checks above exist rather than being an afterthought.

**Nothing in this project is investment advice.** Paper trade for a meaningful stretch of
time before it touches real money, and never risk more than you can afford to lose.

## Licence

MIT.
