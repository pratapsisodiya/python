/*
 * The popup: this week's orders as a checklist you work down while placing them.
 *
 * The whole design goal is that a row cannot be misread. Side is coloured, quantity is
 * tabular, the instrument is spelled out under the ticker, and a futures short — the one
 * genuinely dangerous row, because placing it as a plain sell means selling stock you do
 * not own — gets both a per-row marker and a banner at the top.
 *
 * State: the tick list is authoritative in `chrome.storage.local` and mirrored to the
 * dashboard. Local-first, because a tick is a note about something the user did in the
 * real world and losing it to a network blip would be worse than the mirror going stale.
 */

const { loadSettings, saveSettings, fetchLatestOrders, pushPlaced, loadPlaced, savePlaced, orderId } =
  self.swingbot;

const el = (id) => document.getElementById(id);

/** Current view: the payload being rendered and the set of placed order ids. */
let state = { settings: null, payload: null, placed: new Set(), source: "dashboard" };

// ---------------------------------------------------------------------------- helpers

function showStatus(message, kind = "error") {
  const node = el("status");
  node.textContent = message;
  node.className = `status${kind === "info" ? " info" : ""}`;
  node.hidden = false;
}

function hideStatus() {
  el("status").hidden = true;
}

let toastTimer = null;
function toast(message) {
  const node = el("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 1400);
}

async function copy(text, label) {
  try {
    await navigator.clipboard.writeText(String(text));
    toast(`${label} copied`);
  } catch {
    // Clipboard permission can be refused; a selectable prompt still gets the job done.
    toast(`copy failed — value is ${text}`);
  }
}

const money = (value, currency) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const symbol = currency === "INR" ? "₹" : currency === "USD" ? "$" : "";
  return `${symbol}${Math.round(value).toLocaleString()}`;
};

const qty = (value) => (Number.isInteger(value) ? value.toLocaleString() : Number(value).toLocaleString());

// ----------------------------------------------------------------------- badge syncing

/*
 * The badge shows orders still to place. Recomputed here as well as in the background
 * worker, so ticking a row updates it immediately instead of at the next alarm.
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

// ---------------------------------------------------------------------------- rendering

function render() {
  const { payload, placed, settings } = state;
  el("market").value = settings.market;
  el("open-dashboard").href = settings.baseUrl;

  const body = el("orders-body");
  body.textContent = "";

  if (!payload) {
    el("meta").hidden = true;
    el("orders-table").hidden = true;
    el("futures-warning").hidden = true;
    el("notes").hidden = true;
    return;
  }

  const total = payload.orders.length;
  const done = payload.orders.filter((o) => placed.has(orderId(o))).length;

  el("meta").hidden = false;
  el("meta-entry").textContent = payload.entry_session
    ? `Place at the open on ${payload.entry_session}`
    : "This week's orders";
  const progress = el("meta-progress");
  progress.textContent = total ? `${done} / ${total} placed` : "no orders";
  progress.classList.toggle("complete", total > 0 && done === total);
  el("meta-run").textContent =
    state.source === "pasted"
      ? "pasted from orders.csv"
      : `${payload.run_id}${payload.equity ? ` · equity ${money(payload.equity, payload.currency)}` : ""}`;

  // Notes are where the portfolio explains itself — a Kelly fallback, a capacity
  // truncation, a sector cap. Surfacing them here means a strangely small book is
  // explained at the moment the user is about to act on it.
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

  const hasFutures = payload.orders.some((o) => (o.instrument || "").includes("futures"));
  el("futures-warning").hidden = !hasFutures;

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

  for (const order of payload.orders) {
    const id = orderId(order);
    const isDone = placed.has(id);
    const row = document.createElement("tr");
    if (isDone) row.classList.add("done");

    const doneCell = document.createElement("td");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = isDone;
    box.title = "Mark as placed";
    box.addEventListener("change", () => togglePlaced(id, box.checked));
    doneCell.append(box);

    const tickerCell = document.createElement("td");
    const name = document.createElement("span");
    name.className = "ticker";
    name.textContent = order.ticker;
    tickerCell.append(name);
    const instrument = document.createElement("span");
    instrument.className = `instrument${(order.instrument || "").includes("futures") ? " futures" : ""}`;
    instrument.textContent =
      order.instrument === "futures"
        ? "futures · F&O segment"
        : order.instrument === "equity_short"
          ? "short sale · borrow required"
          : order.instrument || "equity";
    tickerCell.append(instrument);

    const sideCell = document.createElement("td");
    sideCell.className = order.side === "buy" ? "side-buy" : "side-sell";
    sideCell.textContent = order.side.toUpperCase();

    const qtyCell = document.createElement("td");
    qtyCell.className = "num";
    qtyCell.textContent = qty(order.quantity);

    const valueCell = document.createElement("td");
    valueCell.className = "num";
    valueCell.textContent = money(order.est_value, payload.currency);

    const copyCell = document.createElement("td");
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "copy";
    copyBtn.title = `Copy "${order.ticker}", then click again for the quantity`;
    // Two values, one button, alternating: on a broker ticket you type the symbol, then
    // the quantity, in that order. Two separate buttons in a 24px column would be worse.
    let next = "ticker";
    copyBtn.addEventListener("click", async () => {
      if (next === "ticker") {
        await copy(order.ticker, "ticker");
        copyBtn.textContent = "qty";
        next = "quantity";
      } else {
        await copy(order.quantity, "quantity");
        copyBtn.textContent = "copy";
        next = "ticker";
      }
    });
    copyCell.append(copyBtn);

    row.append(doneCell, tickerCell, sideCell, qtyCell, valueCell, copyCell);
    body.append(row);
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

    // The server's copy and the local copy are unioned rather than one overwriting the
    // other. A tick made here while the server was down, and a tick made in the
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
 * The paste fallback. A minimal CSV reader is enough here: `orders.csv` is written by
 * `orders_to_frame`, so the columns are known and no field is ever quoted or contains a
 * comma. Anything more would be pretending to handle input this file never produces.
 */
function parseOrdersCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) throw new Error("that does not look like orders.csv — no data rows");

  const header = lines[0].split(",").map((h) => h.trim());
  const required = ["ticker", "side", "quantity"];
  const missing = required.filter((column) => !header.includes(column));
  if (missing.length) {
    throw new Error(`orders.csv is missing the ${missing.join(", ")} column(s)`);
  }

  const numeric = new Set(["quantity", "est_price", "est_value", "limit_price"]);
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
    entry_session: "",
    currency: "",
    equity: null,
    notes: [],
    caveats: ["Pasted from a file — the dashboard did not verify this against a run."],
    orders,
    placed: [],
  };
}

async function loadPasted() {
  const text = el("paste-input").value;
  try {
    const payload = parseOrdersCsv(text);
    state.source = "pasted";
    state.payload = payload;
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
