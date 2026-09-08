/*
 * The popup: this week's orders as a checklist you work down while placing them.
 *
 * Three things here are not cosmetic.
 *
 * **The list is in execution order, not file order.** `orders.csv` sorts by side then
 * ticker, which puts every buy first purely because "buy" sorts before "sell" — close to
 * the worst order for anyone without margin, because the buys demand cash the sells have
 * not released yet. The sequence comes from the server, which computed it at signal time:
 * closes, then reductions, then new shorts, then new longs, largest first within each.
 *
 * **The book's live state is on screen.** Placing 8 of 14 orders leaves an exposure the
 * model never chose — possibly net long when it wanted neutral. The running panel makes a
 * partial execution a known state rather than something discovered next week.
 *
 * **The fill price is captured, not just a tick.** Every backtest in this system charges a
 * modelled cost and sweeps it to show the strategy does not depend on the exact number.
 * That is an honest assumption, and still an assumption. A fill price makes it measurable,
 * which is the one thing a spreadsheet of intentions can never do.
 *
 * Design goal throughout: a row cannot be misread. Side is coloured, quantity is tabular,
 * the instrument is spelled out, and a futures short — the genuinely dangerous row,
 * because placing it as a plain sell means selling stock you do not own — gets both a
 * per-row marker and a banner.
 */

const {
  loadSettings, saveSettings, fetchLatestOrders, fetchSlippage,
  pushPlaced, pushFill, loadPlaced, savePlaced, orderId,
} = self.swingbot;

const el = (id) => document.getElementById(id);

/** Current view: the payload, the tick set, and any fill prices entered. */
let state = { settings: null, payload: null, placed: new Set(), fills: {}, source: "dashboard" };

// ---------------------------------------------------------------------------- helpers

function showStatus(message, kind = "error") {
  const node = el("status");
  node.textContent = message;
  node.className = `status${kind === "info" ? " info" : ""}`;
  node.hidden = false;
}

const hideStatus = () => { el("status").hidden = true; };

let toastTimer = null;
function toast(message) {
  const node = el("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 1500);
}

async function copy(text, label) {
  try {
    await navigator.clipboard.writeText(String(text));
    toast(`${label} copied`);
  } catch {
    // Clipboard permission can be refused; showing the value still gets the job done.
    toast(`copy failed — value is ${text}`);
  }
}

const symbolFor = (currency) => (currency === "INR" ? "₹" : currency === "USD" ? "$" : "");

function money(value, currency) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${symbolFor(currency)}${Math.round(value).toLocaleString()}`;
}

/**
 * Compact money for the running panel, where four figures share 420 pixels.
 *
 * Lakh and crore for INR, k and m otherwise — because ₹12.5L is what an Indian user
 * reads at a glance and ₹1,250,000 is not.
 */
function compact(value, currency) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";

  const scale =
    currency === "INR"
      ? abs >= 1e7 ? [1e7, "cr"] : abs >= 1e5 ? [1e5, "L"] : abs >= 1e3 ? [1e3, "k"] : [1, ""]
      : abs >= 1e6 ? [1e6, "m"] : abs >= 1e3 ? [1e3, "k"] : [1, ""];

  const [divisor, suffix] = scale;
  const scaled = abs / divisor;
  const shown = divisor > 1 && scaled < 10 ? scaled.toFixed(1) : Math.round(scaled);
  return `${sign}${symbolFor(currency)}${shown}${suffix}`;
}

const qty = (value) =>
  Number.isInteger(value) ? value.toLocaleString() : Number(value).toLocaleString();

/**
 * Human label for one step of the sequence.
 *
 * Unnumbered on purpose. These are four fixed stages but most weeks only use two of
 * them, and a list that opened at "3 · New shorts" with no 1 or 2 above it read like
 * something was missing. The row numbers carry the order; these carry the reason.
 */
const STEP_LABELS = {
  close: "Close out first — frees the most capital and takes risk off",
  reduce: "Then trim — less of what you already hold",
  open_short: "Then new shorts — no cash needed, and borrow can disappear",
  open_long: "New longs last — this is the part that spends cash",
};

// ------------------------------------------------------------------------------ badge

/*
 * Recomputed here as well as in the background worker so a tick updates the badge
 * immediately, rather than at the next alarm up to fifteen minutes later.
 */
async function refreshBadge() {
  const remaining = state.payload
    ? state.payload.orders.filter((o) => !state.placed.has(orderId(o))).length
    : 0;
  try {
    await chrome.action.setBadgeText({ text: remaining > 0 ? String(remaining) : "" });
    await chrome.action.setBadgeBackgroundColor({ color: "#1f7a4d" });
  } catch {
    /* Badge APIs are unavailable in some contexts; not worth failing the popup over. */
  }
}

// -------------------------------------------------------------------- order sequencing

/**
 * The orders in the order they should be worked.
 *
 * Uses the server's sequence when there is one. The fallback is not file order but a
 * local approximation of the same rule, because file order is actively harmful here and
 * a run written before sequencing existed should not silently get the bad ordering back.
 */
function sequencedOrders(payload) {
  const orders = payload.orders || [];
  const sequence = payload.sequence || [];

  if (sequence.length) {
    const rank = new Map(sequence.map((row) => [row.client_order_id, row]));
    const decorated = orders.map((order) => {
      const row = rank.get(order.client_order_id);
      return { order, step: row?.step || "open_long", position: row?.position ?? 999 };
    });
    decorated.sort((a, b) => a.position - b.position);
    return decorated;
  }

  const priority = { close: 0, reduce: 1, open_short: 2, open_long: 3 };
  const stepOf = (order) => {
    const derivative = ["futures", "equity_short"].includes(order.instrument);
    if (derivative) return order.side === "sell" ? "open_short" : "close";
    return order.side === "sell" ? "reduce" : "open_long";
  };
  return orders
    .map((order) => ({ order, step: stepOf(order), position: 0 }))
    .sort((a, b) =>
      priority[a.step] - priority[b.step] ||
      (b.order.est_value || 0) - (a.order.est_value || 0));
}

// ------------------------------------------------------------------- running exposure

/**
 * What the book looks like right now, given only the rows already ticked.
 *
 * Signed by side: a filled buy adds long exposure, a filled sell of a short instrument
 * adds short exposure. Cash counts a long purchase as spent and an equity sale as
 * released; a short sale releases nothing, because assuming it does is exactly how a
 * buying-power rejection happens.
 */
function runningExposure(decorated, placed, equity) {
  let long = 0;
  let short = 0;
  let cash = 0;
  let remainingLongs = 0;

  for (const { order, step } of decorated) {
    const value = Number(order.est_value) || 0;
    const done = placed.has(orderId(order));
    if (!done) {
      if (step === "open_long") remainingLongs += value;
      continue;
    }
    if (step === "open_long") { long += value; cash -= value; }
    else if (step === "open_short") { short += value; }
    else if (step === "reduce" || step === "close") { long -= value; cash += value; }
  }

  const net = long - short;
  return {
    long, short, net, cash, remainingLongs,
    netFraction: equity ? net / equity : null,
    grossFraction: equity ? (long + short) / equity : null,
  };
}

function renderRunning(decorated, placed, payload) {
  const done = decorated.filter(({ order }) => placed.has(orderId(order))).length;
  const panel = el("running");
  if (!done) { panel.hidden = true; return; }
  panel.hidden = false;

  const equity = Number(payload.equity) || 0;
  const x = runningExposure(decorated, placed, equity);
  const currency = payload.currency;

  el("run-long").textContent = compact(x.long, currency);
  el("run-short").textContent = compact(x.short, currency);
  const netNode = el("run-net");
  netNode.textContent = compact(x.net, currency);
  netNode.className = `v ${x.net > 0 ? "pos" : x.net < 0 ? "neg" : ""}`;
  el("run-cash").textContent = compact(x.cash, currency);

  // The point of the panel: say what stopping here would leave you holding.
  const note = el("run-note");
  const remaining = decorated.length - done;
  if (!remaining) {
    note.className = "running-note";
    note.textContent = "All orders placed. The book now matches the target.";
    return;
  }

  const parts = [`${remaining} order(s) still to place.`];
  if (x.netFraction !== null && Math.abs(x.netFraction) > 0.15) {
    note.className = "running-note alert";
    parts.push(
      `Stopping now leaves you ${x.netFraction > 0 ? "net long" : "net short"} ` +
      `${Math.abs(x.netFraction * 100).toFixed(0)}% of equity — the target book was close to balanced.`,
    );
  } else {
    note.className = "running-note";
  }
  if (x.remainingLongs > 0) {
    parts.push(`${compact(x.remainingLongs, currency)} of buying still to fund.`);
  }
  note.textContent = parts.join(" ");
}

// ---------------------------------------------------------------------------- rendering

function render() {
  const { payload, placed, settings } = state;
  el("market").value = settings.market;
  el("open-dashboard").href = settings.baseUrl;

  const body = el("orders-body");
  body.textContent = "";

  if (!payload) {
    for (const id of ["meta", "orders-table", "futures-warning", "lot-warning", "notes", "running", "quality-panel"]) {
      el(id).hidden = true;
    }
    return;
  }

  const decorated = sequencedOrders(payload);
  const total = decorated.length;
  const done = decorated.filter(({ order }) => placed.has(orderId(order))).length;

  el("meta").hidden = false;
  el("meta-entry").textContent = payload.entry_session
    ? `Place at the open on ${payload.entry_session}`
    : "This week's orders";
  const progress = el("meta-progress");
  progress.textContent = total ? `${done} / ${total} placed` : "no orders";
  progress.classList.toggle("complete", total > 0 && done === total);
  el("meta-run").textContent =
    state.source === "pasted"
      ? "pasted from orders.csv — not verified against a run"
      : `${payload.run_id}${payload.equity ? ` · equity ${money(payload.equity, payload.currency)}` : ""}`;

  renderRunning(decorated, placed, payload);

  // Notes are where the portfolio explains itself — a capacity truncation, a risk limit,
  // a position too small to be tradeable. Above the list, because they change what you
  // should do about it.
  const notes = [...(payload.notes || []), ...(payload.caveats || [])];
  const notesNode = el("notes");
  if (notes.length) {
    const list = document.createElement("ul");
    for (const note of notes) {
      const item = document.createElement("li");
      item.textContent = note;
      list.append(item);
    }
    notesNode.textContent = "";
    notesNode.append(list);
    notesNode.hidden = false;
  } else {
    notesNode.hidden = true;
  }

  el("futures-warning").hidden = !payload.orders.some((o) =>
    String(o.instrument || "").includes("futures"));

  // Unrounded derivative quantities are the one thing here that gets outright rejected.
  const unrounded = payload.orders.filter((o) => String(o.tag || "").includes("lot-unknown"));
  el("lot-warning").hidden = !unrounded.length;
  if (unrounded.length) {
    el("lot-warning-text").textContent =
      `${unrounded.length} order(s) are for an instrument that trades in exchange-set lots, ` +
      "and no lot size is on file — so these quantities are not placeable as shown. " +
      "Check the lot with your broker and round DOWN, never up.";
  }

  if (!total) {
    el("orders-table").hidden = true;
    const empty = el("empty");
    empty.hidden = false;
    empty.textContent =
      "No trades this week — the target book matches what you already hold. That is a normal outcome; the no-trade band skips changes too small to be worth the cost.";
    return;
  }

  el("empty").hidden = true;
  el("orders-table").hidden = false;

  let lastStep = null;
  for (const { order, step } of decorated) {
    // A header row per step, so the four stages of the rebalance are visible rather than
    // implied by row order.
    if (step !== lastStep) {
      const header = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 7;
      cell.className = "step-label";
      cell.textContent = STEP_LABELS[step] || step;
      header.append(cell);
      body.append(header);
      lastStep = step;
    }

    body.append(orderRow(order, payload));
  }

  renderQuality();
}

function orderRow(order, payload) {
  const id = orderId(order);
  const isDone = state.placed.has(id);
  const row = document.createElement("tr");
  if (isDone) row.classList.add("done");

  const doneCell = document.createElement("td");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = isDone;
  box.title = "Mark as placed";
  box.addEventListener("change", () => togglePlaced(id, box.checked));
  doneCell.append(box);

  const seqCell = document.createElement("td");
  seqCell.className = "seq";
  const position = (payload.sequence || []).find((r) => r.client_order_id === id)?.position;
  seqCell.textContent = position ?? "";

  const tickerCell = document.createElement("td");
  const name = document.createElement("span");
  name.className = "ticker";
  name.textContent = order.ticker;
  tickerCell.append(name);
  const instrument = document.createElement("span");
  const isFutures = String(order.instrument || "").includes("futures");
  instrument.className = `instrument${isFutures ? " futures" : ""}`;
  instrument.textContent = isFutures
    ? "futures · F&O segment"
    : order.instrument === "equity_short"
      ? "short sale · borrow required"
      : order.instrument || "equity";
  tickerCell.append(instrument);

  const sideCell = document.createElement("td");
  sideCell.className = order.side === "buy" ? "side-buy" : "side-sell";
  sideCell.textContent = String(order.side).toUpperCase();

  const qtyCell = document.createElement("td");
  qtyCell.className = "num";
  qtyCell.textContent = qty(order.quantity);

  // Value, and under it the fill-price box. Placed next to the value because that is
  // where the eye already is when reading back a confirmation.
  const valueCell = document.createElement("td");
  valueCell.className = "num";
  valueCell.append(document.createTextNode(money(order.est_value, payload.currency)));
  valueCell.append(fillInput(order, payload));

  const copyCell = document.createElement("td");
  copyCell.append(copyButton(order));

  row.append(doneCell, seqCell, tickerCell, sideCell, qtyCell, valueCell, copyCell);
  return row;
}

/** One button, two values, alternating: a ticket asks for the symbol then the quantity. */
function copyButton(order) {
  const button = document.createElement("button");
  button.className = "copy-btn";
  button.textContent = "copy";
  button.title = `Copy "${order.ticker}", then click again for the quantity`;
  let next = "ticker";
  button.addEventListener("click", async () => {
    if (next === "ticker") {
      await copy(order.ticker, "ticker");
      button.textContent = "qty";
      next = "quantity";
    } else {
      await copy(order.quantity, "quantity");
      button.textContent = "copy";
      next = "ticker";
    }
  });
  return button;
}

/**
 * The fill-price box, and the slippage it implies.
 *
 * Deliberately optional and out of the way: nobody should be blocked from ticking a row
 * because they have not read the confirmation yet. But when it is filled in, the number
 * shown next to it is the same one the backtest's cost model was predicting, so the
 * comparison is visible at the moment it is cheapest to make.
 */
function fillInput(order, payload) {
  const id = orderId(order);
  const wrap = document.createElement("span");

  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.min = "0";
  input.className = "fill-input";
  input.placeholder = "fill @";
  input.title = "Price you actually got, before commission and taxes";
  const recorded = state.fills[id]?.price;
  if (recorded) input.value = recorded;

  const slip = document.createElement("span");
  slip.className = "slip";

  const showSlippage = (price) => {
    const reference = Number(order.est_price) || 0;
    if (!price || !reference) { slip.textContent = ""; return; }

    // Signed so positive is always a cost, whichever side the trade was.
    const direction = order.side === "buy" ? 1 : -1;
    const bps = direction * ((price - reference) / reference) * 10000;

    // The same threshold the server applies before averaging a fill in. Twenty percent
    // from the reference on a weekly rebalance is a mistyped digit, not a trade, and
    // showing "+89,999 bps" as though it were a measurement is worse than saying so.
    if (Math.abs(bps) > 2000) {
      slip.className = "slip pos";
      slip.textContent = `that is ${(bps / 100).toFixed(0)}% from ${reference.toFixed(2)} — typo?`;
      return;
    }

    slip.className = `slip ${bps > 0 ? "pos" : "neg"}`;
    const expected = order.expected_slip_bps;
    slip.textContent = expected
      ? `${bps >= 0 ? "+" : ""}${bps.toFixed(0)} bps (model said ${Number(expected).toFixed(0)})`
      : `${bps >= 0 ? "+" : ""}${bps.toFixed(0)} bps`;
  };
  showSlippage(Number(input.value));

  input.addEventListener("change", async () => {
    const price = Number(input.value);
    showSlippage(price);
    if (!price || price <= 0) return;
    state.fills[id] = { ...(state.fills[id] || {}), price };
    // A fill is a fact nothing else can reconstruct, so a failure to save it is said out
    // loud rather than logged and forgotten.
    if (state.source !== "dashboard") { toast("fill noted locally only"); return; }
    try {
      await pushFill({
        baseUrl: state.settings.baseUrl,
        runId: state.payload.run_id,
        fill: { client_order_id: id, placed: true, price },
      });
      state.placed.add(id);
      await savePlaced(state.payload.run_id, state.placed);
      toast("fill recorded");
      render();
      await refreshBadge();
    } catch (error) {
      showStatus(`could not save that fill: ${error.message}`);
    }
  });

  wrap.append(input, slip);
  return wrap;
}

// ------------------------------------------------------------------ execution quality

async function renderQuality() {
  const panel = el("quality-panel");
  if (state.source !== "dashboard" || !state.payload?.run_id) { panel.hidden = true; return; }

  let report;
  try {
    report = await fetchSlippage({
      baseUrl: state.settings.baseUrl, runId: state.payload.run_id,
    });
  } catch { panel.hidden = true; return; }

  if (!report || !report.n_filled) { panel.hidden = true; return; }
  panel.hidden = false;

  const badge = el("quality-badge");
  badge.textContent = `${report.weighted_slippage_bps >= 0 ? "+" : ""}${report.weighted_slippage_bps.toFixed(0)} bps`;
  el("quality-verdict").textContent = report.verdict;

  const rows = el("quality-rows");
  rows.textContent = "";
  for (const fill of (report.fills || []).slice(0, 8)) {
    const line = document.createElement("div");
    const left = document.createElement("span");
    left.textContent = `${fill.ticker} ${fill.side}`;
    const right = document.createElement("span");
    right.className = fill.slippage_bps > 0 ? "bad" : "";
    right.textContent = `${fill.slippage_bps >= 0 ? "+" : ""}${fill.slippage_bps.toFixed(0)} bps`;
    line.append(left, right);
    rows.append(line);
  }
}

// ------------------------------------------------------------------------------ actions

async function togglePlaced(id, checked) {
  if (checked) state.placed.add(id);
  else state.placed.delete(id);

  const runId = state.payload?.run_id;
  if (runId) {
    await savePlaced(runId, state.placed);
    if (state.source === "dashboard") {
      await pushPlaced({ baseUrl: state.settings.baseUrl, runId, placed: [...state.placed] });
    }
  }
  render();
  await refreshBadge();
}

async function load() {
  hideStatus();
  state.settings = await loadSettings();
  el("market").value = state.settings.market;

  try {
    const payload = await fetchLatestOrders(state.settings);
    if (!payload) {
      state.payload = null;
      state.placed = new Set();
      render();
      const empty = el("empty");
      empty.hidden = false;
      empty.innerHTML =
        "No signal run yet for this market. Run <code>swingbot signal</code> — or press the button in the dashboard — and reopen this popup.";
      await refreshBadge();
      return;
    }

    state.source = "dashboard";
    state.payload = payload;
    state.fills = payload.execution || {};

    // The server's tick set and the local one are unioned rather than one overwriting
    // the other. A tick made here while the server was down, and one made in the
    // dashboard, are both real; neither should be silently dropped.
    const local = await loadPlaced(payload.run_id);
    state.placed = new Set([...local, ...(payload.placed || [])]);
    if (state.placed.size !== local.size) await savePlaced(payload.run_id, state.placed);

    el("empty").hidden = true;
    render();
    await refreshBadge();
  } catch (error) {
    state.payload = null;
    render();
    showStatus(error.message);
    const empty = el("empty");
    empty.hidden = false;
    empty.innerHTML =
      "Start the dashboard with <code>swingbot serve</code>, or paste <code>orders.csv</code> below.";
    el("paste-panel").open = true;
  }
}

/*
 * The paste fallback. A minimal CSV reader is enough: `orders.csv` is written by
 * `orders_to_frame`, so the columns are known and no field is ever quoted or contains a
 * comma. Anything more would be pretending to handle input this file never produces.
 */
function parseOrdersCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) throw new Error("that does not look like orders.csv — no data rows");

  const header = lines[0].split(",").map((h) => h.trim());
  const missing = ["ticker", "side", "quantity"].filter((c) => !header.includes(c));
  if (missing.length) {
    throw new Error(`orders.csv is missing the ${missing.join(", ")} column(s)`);
  }

  const numeric = new Set(["quantity", "est_price", "est_value", "limit_price", "expected_slip_bps"]);
  const orders = lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    header.forEach((column, i) => {
      const raw = (cells[i] ?? "").trim();
      row[column] = numeric.has(column) ? (raw === "" ? null : Number(raw)) : raw;
    });
    return row;
  });

  return {
    run_id: `pasted-${new Date().toISOString().slice(0, 10)}`,
    entry_session: "", currency: "", equity: null,
    notes: [], caveats: ["Pasted from a file — the dashboard did not verify this against a run."],
    orders, placed: [], sequence: [], execution: {},
  };
}

async function loadPasted() {
  try {
    const payload = parseOrdersCsv(el("paste-input").value);
    state.source = "pasted";
    state.payload = payload;
    state.fills = {};
    state.placed = await loadPlaced(payload.run_id);
    hideStatus();
    el("empty").hidden = true;
    render();
    await refreshBadge();
  } catch (error) {
    showStatus(error.message);
  }
}

// -------------------------------------------------------------------------------- wiring

el("refresh").addEventListener("click", load);
el("settings").addEventListener("click", () => chrome.runtime.openOptionsPage());
el("paste-load").addEventListener("click", loadPasted);
el("market").addEventListener("change", async (event) => {
  await saveSettings({ market: event.target.value });
  await load();
});
el("open-dashboard").addEventListener("click", (event) => {
  event.preventDefault();
  chrome.tabs.create({ url: state.settings.baseUrl });
});

load();
