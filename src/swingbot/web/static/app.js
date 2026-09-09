/*
 * Dashboard front end. No framework and no build step, on purpose: the README's audience
 * is following copy-paste install steps, and "now install Node" would break that promise.
 *
 * Everything rendered here comes from the API, which serialises the same result objects
 * the CLI prints. If a number on this page disagreed with the terminal, that would be a
 * bug in one shared code path rather than a difference between two implementations.
 */

const $ = (id) => document.getElementById(id);

const state = {
  market: $("market").value,
  jobId: null,
  logLines: 0,
  pollTimer: null,
  week: null,
};

// ------------------------------------------------------------------------- formatting

const NBSP_DASH = "—";

const num = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value)
    ? NBSP_DASH
    : Number(value).toFixed(digits);

const pct = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? NBSP_DASH
    : `${(Number(value) * 100).toFixed(digits)}%`;

function money(value, currency) {
  if (value === null || value === undefined || Number.isNaN(value)) return NBSP_DASH;
  const symbol = currency === "INR" ? "₹" : currency === "USD" ? "$" : "";
  return `${symbol}${Math.round(Number(value)).toLocaleString()}`;
}

const shortStamp = (iso) => (iso ? String(iso).slice(0, 16).replace("T", " ") : NBSP_DASH);

/**
 * Format a number for a table whose columns we do not know in advance.
 *
 * Precision scaled to magnitude, trailing zeros trimmed. A fixed four decimals put "504
 * weeks" on the ablation table as `504.0000` and a hit rate of 72% as `0.7210`, which
 * makes a panel meant to be read carefully harder to read than it needs to be.
 */
function autoNum(value) {
  if (!Number.isFinite(value)) return NBSP_DASH;
  if (Number.isInteger(value)) return value.toLocaleString();
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 1 : abs >= 1 ? 2 : 4;
  // Group the thousands here too. A column of notionals where 490320 reads "490,320" and
  // 4453333.3 reads "4453333.3" is a column you cannot scan: the eye compares digit
  // counts, and the one row that happens to be fractional is the one it misreads.
  return Number(value.toFixed(digits)).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

let toastTimer = null;
function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 2200);
}

/** Build an element with text content, avoiding innerHTML for anything data-derived. */
function cell(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

async function api(path, { method = "GET", body = null } = {}) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    // The API sends a human-readable `detail` for anything the user can act on, so it is
    // shown as-is rather than replaced with a generic failure message.
    throw new Error(payload?.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

/**
 * A generic table renderer for API rows whose columns we do not want to hard-code.
 *
 * An empty result hides the container rather than drawing an empty bordered box with
 * "nothing to show" in it. The checks panel had two of those stacked under a heading, and
 * they read as though something had gone wrong when in fact the run simply had no ablation
 * — which the verdict above already says, in words.
 */
function renderRowsTable(container, rows, { limit = 200 } = {}) {
  container.textContent = "";
  if (!rows || !rows.length) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  const columns = Object.keys(rows[0]);
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) headRow.append(cell("th", column.replace(/_/g, " ")));
  head.append(headRow);
  const tbody = document.createElement("tbody");
  for (const row of rows.slice(0, limit)) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const value = row[column];
      const isNumber = typeof value === "number";
      tr.append(cell("td", isNumber ? autoNum(value) : (value ?? NBSP_DASH), isNumber ? "num" : ""));
    }
    tbody.append(tr);
  }
  table.append(head, tbody);
  container.append(table);
}

function kpi(container, label, value, { className = "", sub = "" } = {}) {
  const box = document.createElement("div");
  box.className = "kpi";
  box.append(cell("div", label, "label"));
  box.append(cell("div", value, `value ${className}`.trim()));
  if (sub) box.append(cell("div", sub, "sub"));
  container.append(box);
}

// ------------------------------------------------------------------------ this week

async function loadWeek() {
  let payload;
  try {
    payload = await api(`/api/orders/latest?market=${encodeURIComponent(state.market)}`);
  } catch (error) {
    toast(error.message);
    return;
  }

  if (!payload.run_id) {
    state.week = null;
    $("week-empty").hidden = false;
    $("week-content").hidden = true;
    return;
  }

  // The orders endpoint is intentionally lean — it is what the extension reads. The book
  // and the exposure numbers live in the run detail, fetched only for this view.
  const detail = await api(`/api/runs/${encodeURIComponent(payload.run_id)}`).catch(() => null);
  state.week = { payload, detail };

  $("week-empty").hidden = true;
  $("week-content").hidden = false;
  renderWeek();
}

function renderWeek() {
  const { payload, detail } = state.week;
  // book.json holds the target book and the portfolio's notes; targets.json is the
  // execution adapter's file and holds the orders. Runs made before book.json existed
  // still render — just without the book table.
  const book = detail?.book || {};
  const positions = book.positions || [];

  const kpis = $("week-kpis");
  kpis.textContent = "";
  kpi(kpis, "Enter at the open on", payload.entry_session || NBSP_DASH);
  kpi(kpis, "Orders", String(payload.orders.length));
  kpi(kpis, "Positions", positions.length
    ? `${positions.length}`
    : NBSP_DASH, {
    sub: positions.length ? `${book.n_long ?? 0} long / ${book.n_short ?? 0} short` : "",
  });
  kpi(kpis, "Gross exposure", pct(book.gross), {
    sub: book.net !== undefined ? `net ${pct(book.net)}` : "",
  });
  kpi(kpis, "Account equity", money(payload.equity, payload.currency));
  if (book.risk_scale !== undefined && book.risk_scale !== null && book.risk_scale < 1) {
    kpi(kpis, "Risk scale", pct(book.risk_scale, 0), {
      className: "neg", sub: "drawdown kill-switch is active",
    });
  }

  const notes = [...(payload.notes || []), ...(payload.caveats || [])];
  const notesNode = $("week-notes");
  notesNode.textContent = "";
  if (notes.length) {
    const heading = cell("div", "The portfolio explains itself:", "muted small");
    const list = document.createElement("ul");
    for (const note of notes) list.append(cell("li", note));
    notesNode.append(heading, list);
    notesNode.hidden = false;
  } else {
    notesNode.hidden = true;
  }

  const placed = new Set(payload.placed || []);
  const hasFutures = payload.orders.some((o) => String(o.instrument || "").includes("futures"));
  $("week-futures").hidden = !hasFutures;

  const body = $("orders-body");
  body.textContent = "";
  for (const order of payload.orders) {
    const id = order.client_order_id || `${order.ticker}-${order.side}-${order.quantity}`;
    const tr = document.createElement("tr");
    if (placed.has(id)) tr.classList.add("done");

    const tick = document.createElement("td");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = placed.has(id);
    box.title = "Mark as placed at your broker";
    box.addEventListener("change", () => togglePlaced(payload.run_id, id, box.checked));
    tick.append(box);

    tr.append(tick);
    tr.append(cell("td", order.ticker, "strong"));
    tr.append(cell("td", String(order.side).toUpperCase(), order.side === "buy" ? "side-buy" : "side-sell"));
    tr.append(cell("td", Number(order.quantity).toLocaleString(), "num"));

    const instrument = document.createElement("td");
    const tag = cell("span", order.instrument || "equity", `tag${String(order.instrument || "").includes("futures") ? " futures" : ""}`);
    instrument.append(tag);
    tr.append(instrument);

    tr.append(cell("td", money(order.est_price, payload.currency), "num"));
    tr.append(cell("td", money(order.est_value, payload.currency), "num"));
    body.append(tr);
  }

  const done = payload.orders.filter((o) =>
    placed.has(o.client_order_id || `${o.ticker}-${o.side}-${o.quantity}`)).length;
  $("week-placed-count").textContent = payload.orders.length
    ? `— ${done} of ${payload.orders.length} placed`
    : "";

  const bookBody = $("book-body");
  bookBody.textContent = "";
  for (const row of positions) {
    const tr = document.createElement("tr");
    tr.append(cell("td", row.ticker, "strong"));
    tr.append(cell("td", pct(row.weight, 2), `num ${row.weight < 0 ? "neg" : "pos"}`));
    tr.append(cell("td", row.weight < 0 ? "short" : "long", row.weight < 0 ? "side-sell" : "side-buy"));
    tr.append(cell("td", row.sector || NBSP_DASH));
    tr.append(cell("td", row.instrument || "equity"));
    tr.append(cell("td", num(row.score, 4), "num"));
    bookBody.append(tr);
  }

  const link = $("week-run-link");
  link.textContent = "";
  link.append(cell("span", `Run ${payload.run_id} — `));
  const report = document.createElement("a");
  report.href = `/api/runs/${encodeURIComponent(payload.run_id)}/report`;
  report.target = "_blank";
  report.rel = "noreferrer";
  report.textContent = "open the full report";
  link.append(report);
  if (detail?.has_model) {
    link.append(cell("span", ` · reproduce it exactly: swingbot signal --market ${state.market} --use-model ${payload.run_id}`));
  }
}

async function togglePlaced(runId, orderId, checked) {
  const placed = new Set(state.week.payload.placed || []);
  if (checked) placed.add(orderId); else placed.delete(orderId);
  try {
    const saved = await api(`/api/orders/${encodeURIComponent(runId)}/placed`, {
      method: "POST",
      body: { placed: [...placed] },
    });
    state.week.payload.placed = saved.placed;
  } catch (error) {
    toast(error.message);
    return;
  }
  renderWeek();
}

// ---------------------------------------------------------------------------- jobs

async function startJob(kind) {
  const body = { kind, market: state.market, notify: false };
  if (kind === "backtest") {
    body.ablation = $("bt-ablation").checked;
    body.sensitivity = $("bt-sensitivity").checked;
    body.pbo = $("bt-pbo").checked;
  } else if (kind === "demo") {
    body.years = Number($("demo-years").value) || 8;
  } else if (kind === "signal" || kind === "run-weekly") {
    const equity = Number($("signal-equity").value);
    if (equity > 0) body.equity = equity;
  }

  let job;
  try {
    job = await api("/api/jobs", { method: "POST", body });
  } catch (error) {
    toast(error.message);
    return;
  }

  state.jobId = job.id;
  state.logLines = 0;
  $("job-log").textContent = "";
  setButtonsBusy(true);
  pollJob();
  loadJobs();
}

function setButtonsBusy(busy) {
  for (const button of document.querySelectorAll("button[data-job]")) button.disabled = busy;
}

async function pollJob() {
  if (!state.jobId) return;
  let job;
  try {
    job = await api(`/api/jobs/${state.jobId}?log_from=${state.logLines}`);
  } catch (error) {
    toast(error.message);
    setButtonsBusy(false);
    return;
  }

  if (job.log?.length) {
    const pane = $("job-log");
    // Appending, and only auto-scrolling when the user is already at the bottom, so
    // scrolling back to read a fold summary is not yanked away on the next poll.
    const atBottom = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 40;
    pane.textContent += job.log.join("\n") + "\n";
    state.logLines = job.n_log;
    if (atBottom) pane.scrollTop = pane.scrollHeight;
  }

  const status = $("job-status");
  status.className = `job-status ${job.state === "failed" ? "failed" : job.state === "running" ? "running" : "muted"}`;
  status.textContent =
    job.state === "queued" ? `${job.kind}: queued behind another job`
    : job.state === "running" ? `${job.kind}: running…`
    : job.state === "failed" ? `${job.kind}: failed — ${job.error}`
    : `${job.kind}: done${job.run_id ? ` — run ${job.run_id}` : ""}`;

  if (job.state === "queued" || job.state === "running") {
    state.pollTimer = setTimeout(pollJob, 1200);
    return;
  }

  setButtonsBusy(false);
  state.jobId = null;
  loadJobs();
  if (job.state === "done") {
    // Refresh whatever the job just changed, so the result is visible without a reload.
    if (job.kind === "signal" || job.kind === "run-weekly") loadWeek();
    if (job.kind === "backtest") loadChecks();
    loadHistory();
    loadHealth();
  }
}

async function loadJobs() {
  const { jobs } = await api("/api/jobs").catch(() => ({ jobs: [] }));
  const body = $("jobs-body");
  body.textContent = "";
  for (const job of jobs) {
    const tr = document.createElement("tr");
    tr.append(cell("td", job.kind, "strong"));
    tr.append(cell("td", job.state, job.state === "failed" ? "neg" : job.state === "done" ? "pos" : ""));
    tr.append(cell("td", shortStamp(job.started_at)));
    tr.append(cell("td", job.run_id || NBSP_DASH));

    const actions = document.createElement("td");
    if (job.run_id) {
      const view = document.createElement("a");
      view.href = `/api/runs/${encodeURIComponent(job.run_id)}/report`;
      view.target = "_blank";
      view.rel = "noreferrer";
      view.textContent = "report";
      actions.append(view);
    } else if (job.error) {
      actions.append(cell("span", job.error, "muted small"));
    }
    tr.append(actions);
    body.append(tr);
  }
}

// ------------------------------------------------------------------ honesty checks

/** Colour a verdict by what it says, using the same words the CLI colours on. */
function verdictClass(text) {
  if (text.includes("LEAK WARNING")) return "bad";
  if (text.includes("does not")) return "warn";
  return "ok";
}

async function loadChecks() {
  const { runs } = await api(`/api/runs?market=${encodeURIComponent(state.market)}&command=backtest&limit=1`)
    .catch(() => ({ runs: [] }));
  if (!runs.length) {
    $("checks-empty").hidden = false;
    $("checks-content").hidden = true;
    return;
  }

  const detail = await api(`/api/runs/${encodeURIComponent(runs[0].run_id)}`);
  $("checks-empty").hidden = true;
  $("checks-content").hidden = false;

  const kpis = $("checks-kpis");
  kpis.textContent = "";
  const metrics = detail.metrics || {};
  kpi(kpis, "Sharpe (net of costs)", num(metrics.sharpe), {
    className: metrics.sharpe > 0 ? "pos" : "neg",
  });
  kpi(kpis, "Annual return", pct(metrics.ann_return), {
    className: metrics.ann_return > 0 ? "pos" : "neg",
  });
  kpi(kpis, "Max drawdown", pct(metrics.max_drawdown), { className: "neg" });
  kpi(kpis, "Turnover / week", pct(metrics.turnover));
  kpi(kpis, "Weeks tested", String(metrics.n_periods ?? NBSP_DASH));

  const deflated = detail.deflated || {};
  if (deflated.probability !== undefined) {
    kpi(kpis, "P(true Sharpe > 0)", num(deflated.probability), {
      className: deflated.probability >= 0.95 ? "pos" : "neg",
      sub: `after ${deflated.n_trials} trial(s)`,
    });
  }

  const list = $("checks-verdicts");
  list.textContent = "";
  const verdicts = detail.verdicts || [];
  if (!verdicts.length) {
    list.append(cell("li", "This run had no ablation. Re-run the backtest with Ablation ticked to get the null-model comparison and the leak check.", "warn"));
  }
  for (const verdict of verdicts) list.append(cell("li", verdict, verdictClass(verdict)));
  if (deflated.verdict) list.append(cell("li", `Deflated Sharpe: ${deflated.verdict}`, deflated.probability >= 0.95 ? "ok" : "warn"));

  renderRowsTable($("checks-ablation"), detail.ablation);

  $("checks-cost-heading").hidden = !(detail.cost_rows || []).length;
  renderRowsTable($("checks-cost"), detail.cost_rows);

  $("checks-ablation-heading").hidden = !(detail.ablation || []).length;

  const pbo = detail.pbo || {};
  const hasPbo = pbo.pbo !== undefined;
  $("checks-pbo-heading").hidden = !hasPbo;
  const pboNode = $("checks-pbo");
  pboNode.textContent = "";
  if (hasPbo) {
    const kpiBox = document.createElement("div");
    kpiBox.className = "kpis";
    kpi(kpiBox, "PBO", num(pbo.pbo), {
      className: pbo.pbo > 0.5 ? "neg" : "pos",
      sub: `${pbo.n_paths} combinatorial path(s)`,
    });
    pboNode.append(kpiBox);
    pboNode.append(cell("p", pbo.verdict || "", "muted small"));
    const paths = document.createElement("div");
    paths.className = "scroll";
    renderRowsTable(paths, detail.pbo_paths);
    pboNode.append(paths);
  }

  const link = $("checks-run-link");
  link.textContent = "";
  link.append(cell("span", `From run ${detail.run_id} — `));
  const report = document.createElement("a");
  report.href = `/api/runs/${encodeURIComponent(detail.run_id)}/report`;
  report.target = "_blank";
  report.rel = "noreferrer";
  report.textContent = "open the full tearsheet with charts";
  link.append(report);
}

// -------------------------------------------------------------------- run history

async function loadHistory() {
  const { runs } = await api(`/api/runs?market=${encodeURIComponent(state.market)}`)
    .catch(() => ({ runs: [] }));
  const body = $("history-body");
  body.textContent = "";
  if (!runs.length) {
    const tr = document.createElement("tr");
    const td = cell("td", "No runs yet for this market.", "muted");
    td.colSpan = 7;
    tr.append(td);
    body.append(tr);
    return;
  }

  for (const run of runs) {
    const tr = document.createElement("tr");
    tr.append(cell("td", run.run_id, "strong"));
    tr.append(cell("td", run.command + (run.pinned_model ? " (pinned)" : "")));
    tr.append(cell("td", num(run.sharpe), "num"));
    tr.append(cell("td", pct(run.max_drawdown), "num"));
    tr.append(cell("td", (run.config_hash || "").slice(0, 8)));
    tr.append(cell("td", shortStamp(run.started_at)));

    const actions = document.createElement("td");
    if (run.report) {
      const view = document.createElement("a");
      view.href = `/api/runs/${encodeURIComponent(run.run_id)}/report`;
      view.target = "_blank";
      view.rel = "noreferrer";
      view.textContent = "report";
      actions.append(view);
    }
    if (run.has_model) {
      const copy = document.createElement("button");
      copy.className = "link";
      copy.style.marginLeft = run.report ? "10px" : "0";
      copy.textContent = "copy --use-model";
      copy.title = "Copy the command that reproduces this run's book exactly";
      copy.addEventListener("click", async () => {
        const command = `swingbot signal --market ${run.market} --use-model ${run.run_id}`;
        try {
          await navigator.clipboard.writeText(command);
          toast("command copied");
        } catch {
          toast(command);
        }
      });
      actions.append(copy);
    }
    tr.append(actions);
    body.append(tr);
  }
}

// -------------------------------------------------------------------------- health

async function loadHealth() {
  let health;
  try {
    health = await api(`/api/health?market=${encodeURIComponent(state.market)}`);
  } catch (error) {
    $("health-line").textContent = "unreachable";
    toast(error.message);
    return;
  }

  $("health-line").textContent = health.data.ready
    ? `${health.data.n_tickers} tickers · ${health.data.n_weeks} weeks · ${health.n_runs} runs`
    : "no data yet — run the demo";

  const node = $("health-content");
  node.textContent = "";

  if (!health.data.ready) {
    const banner = cell("div", `Data not ready: ${health.data.error}`, "banner err");
    node.append(banner);
  }
  if (health.data.bias_warning) {
    node.append(cell("div", health.data.bias_warning, "banner warn"));
  }

  // Where the prices came from, above everything else in this section. Every other number
  // on this dashboard is downstream of it: a Sharpe computed on generated prices is a
  // measurement of the generator, and nothing else here would tell you that.
  const prov = health.data.provenance;
  if (prov && prov.n_tickers) {
    node.append(cell("div", prov.verdict, `banner ${prov.clean ? "ok" : "err"}`));
  }

  const facts = document.createElement("dl");
  facts.className = "facts";
  const add = (label, value) => {
    facts.append(cell("dt", label));
    facts.append(cell("dd", value));
  };
  add("swingbot", health.version);
  add("Market", `${health.display_name} (${health.market})`);
  add("Config hash", health.config_hash);
  add("Calendar", health.data.calendar || NBSP_DASH);
  add("Latest week", health.data.latest_decision
    ? `decide ${health.data.latest_decision} → enter ${health.data.latest_entry} → exit ${health.data.latest_exit}`
    : NBSP_DASH);
  add("Bars", health.data.ready ? `${health.data.n_bars.toLocaleString()} across ${health.data.n_tickers} tickers` : NBSP_DASH);
  if (prov && prov.n_tickers) {
    const sources = Object.entries(prov.sources).map(([k, v]) => `${k} ${v}`).join(" · ");
    add("Price sources", sources || "unknown");
  }
  add("Round-trip cost", `long ${num(health.costs.round_trip_long_bps, 1)} bps · short ${num(health.costs.round_trip_short_bps, 1)} bps (as ${health.costs.short_instrument})`);
  add("Forecast blend", `${pct(health.blend.price, 0)} price / ${pct(health.blend.news, 0)} news`);
  add("News backend", health.nlp_backend);
  add("Distinct configs tried", `${health.n_trial_configs} — this is the n in the deflated Sharpe`);
  if (health.news_cache?.entries !== undefined) {
    add("News cache", `${health.news_cache.entries} entries, ${health.news_cache.failed} failed, ${num(health.news_cache.total_spend_usd, 2)} USD spent`);
  }
  node.append(facts);

  // What shorting needs in capital. A weekly short on NSE is a single-stock future, and
  // futures trade in indivisible exchange-set lots — so a per-name weight cap puts a hard
  // floor under the account size at which the short sleeve can exist at all. Learning
  // that here beats learning it from a rejected order on a Monday morning.
  const shorting = health.shorting;
  if (shorting && Object.keys(shorting).length) {
    node.append(cell("h3", "Shorting: what it needs in capital"));
    if (!shorting.available) {
      node.append(cell("div", shorting.reason || "unavailable", "banner warn"));
    } else {
      const ok = shorting.n_tradeable >= 7;
      node.append(cell("div", shorting.verdict, `banner ${ok ? "ok" : "err"}`));
      const s = document.createElement("dl");
      s.className = "facts";
      const put = (k, v) => { s.append(cell("dt", k)); s.append(cell("dd", v)); };
      put("Shortable names", `${shorting.n_tradeable} of ${shorting.n_shortable} fit the ${pct(shorting.max_weight, 0)} cap at ${autoNum(shorting.equity)} of equity`);
      put("Smallest workable account", autoNum(shorting.minimum_equity_for_any));
      if (shorting.n_no_futures) {
        put("No futures contract", `${shorting.n_no_futures} universe name(s) — long-only regardless of the model`);
      }
      put("Lot snapshot", shorting.snapshot_date || NBSP_DASH);
      node.append(s);

      const blocked = (shorting.blocked || []).slice(0, 10);
      if (blocked.length) {
        node.append(cell("p", "Cheapest lots, and the equity each one needs:", "muted small"));
        const bwrap = document.createElement("div");
        bwrap.className = "scroll";
        renderRowsTable(bwrap, blocked);
        node.append(bwrap);
      }
    }
  }

  node.append(cell("h3", "Packages"));
  const wrap = document.createElement("div");
  wrap.className = "scroll";
  renderRowsTable(wrap, health.packages);
  node.append(wrap);
}

// -------------------------------------------------------------------------- wiring

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tab")) other.classList.remove("active");
    for (const panel of document.querySelectorAll(".panel")) panel.classList.remove("active");
    tab.classList.add("active");
    $(`panel-${tab.dataset.panel}`).classList.add("active");
  });
}

for (const button of document.querySelectorAll("button[data-job]")) {
  button.addEventListener("click", () => startJob(button.dataset.job));
}

$("market").addEventListener("change", (event) => {
  state.market = event.target.value;
  refreshAll();
});

function refreshAll() {
  loadWeek();
  loadChecks();
  loadHistory();
  loadHealth();
  loadJobs();
}

refreshAll();
