/*
 * Talking to the local dashboard.
 *
 * Everything this extension knows comes from a swingbot server running on the user's own
 * machine. There is no remote endpoint, no account, and no credential anywhere in this
 * codebase — which is why the manifest's host permissions are limited to loopback.
 *
 * Shared by the popup, the options page and the background worker, loaded as a classic
 * script in all three, so it attaches to `self` rather than using ES module exports (a
 * Firefox MV3 event page cannot be a module).
 *
 * Wrapped in an IIFE, and this is load-bearing rather than style. Classic scripts in one
 * page share a single global scope: with these functions declared at the top level,
 * `popup.js` doing `const { loadSettings } = self.swingbot` was a redeclaration of the
 * same binding, and the popup died with "Identifier 'loadSettings' has already been
 * declared" before rendering a single row. Only `self.swingbot` escapes now.
 */
(() => {
"use strict";

const DEFAULT_SETTINGS = {
  baseUrl: "http://127.0.0.1:8765",
  market: "us",
};

/** Read the user's settings, falling back to the loopback default. */
async function loadSettings() {
  const stored = await chrome.storage.local.get(["baseUrl", "market"]);
  const baseUrl = (stored.baseUrl || DEFAULT_SETTINGS.baseUrl).replace(/\/+$/, "");
  return { baseUrl, market: stored.market || DEFAULT_SETTINGS.market };
}

async function saveSettings(settings) {
  await chrome.storage.local.set(settings);
}

/*
 * A short timeout on purpose. The most common failure by far is "the dashboard isn't
 * running", and the popup should say so in a second rather than showing a spinner that
 * looks like the server is thinking.
 */
const REQUEST_TIMEOUT_MS = 4000;

async function request(path, { baseUrl, method = "GET", body = null } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method,
      signal: controller.signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`${response.status} ${response.statusText}${detail ? `: ${detail.slice(0, 200)}` : ""}`);
    }
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`no response from ${baseUrl} — is \`swingbot serve\` running?`);
    }
    // A refused connection surfaces as an opaque TypeError, so name the likely cause
    // rather than showing "Failed to fetch" and leaving the user to guess.
    if (error instanceof TypeError) {
      throw new Error(`could not reach ${baseUrl} — is \`swingbot serve\` running?`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/** This week's orders for a market, or null when no signal run exists yet. */
async function fetchLatestOrders({ baseUrl, market }) {
  const payload = await request(`/api/orders/latest?market=${encodeURIComponent(market)}`, { baseUrl });
  return payload && payload.run_id ? payload : null;
}

/**
 * Tell the dashboard which orders have been placed.
 *
 * Best-effort by design: the extension's own `chrome.storage` copy is the one the popup
 * renders from, so a server that is briefly down must not lose a tick the user just made.
 */
async function pushPlaced({ baseUrl, runId, placed }) {
  try {
    return await request(`/api/orders/${encodeURIComponent(runId)}/placed`, {
      baseUrl,
      method: "POST",
      body: { placed },
    });
  } catch (error) {
    console.warn("swingbot: could not sync the checklist to the dashboard", error);
    return null;
  }
}

/**
 * Record what an order actually filled at.
 *
 * Unlike a tick, this one is worth telling the user about when it fails: a fill price is
 * a fact about the past that nothing else can reconstruct, and the whole point of
 * capturing it is that the dashboard can then measure realised slippage against what the
 * backtest assumed. So the error is returned rather than swallowed.
 */
async function pushFill({ baseUrl, runId, fill }) {
  return request(`/api/orders/${encodeURIComponent(runId)}/fills`, {
    baseUrl,
    method: "POST",
    body: { fills: [fill] },
  });
}

/** Realised slippage for a run, once fills have been entered. */
async function fetchSlippage({ baseUrl, runId }) {
  return request(`/api/orders/${encodeURIComponent(runId)}/slippage`, { baseUrl });
}

/*
 * The checklist lives under one key per run id, so last week's ticks never bleed into this
 * week's list, and an old run reopened later still shows what was done at the time.
 */
const placedKey = (runId) => `placed:${runId}`;

async function loadPlaced(runId) {
  const key = placedKey(runId);
  const stored = await chrome.storage.local.get([key]);
  return new Set(stored[key] || []);
}

async function savePlaced(runId, placed) {
  await chrome.storage.local.set({ [placedKey(runId)]: [...placed] });
}

/** Stable identity for one order row. */
function orderId(order) {
  return order.client_order_id || `${order.ticker}-${order.side}-${order.quantity}`;
}

self.swingbot = {
  DEFAULT_SETTINGS,
  loadSettings,
  saveSettings,
  request,
  fetchLatestOrders,
  pushPlaced,
  pushFill,
  fetchSlippage,
  loadPlaced,
  savePlaced,
  orderId,
};
})();
