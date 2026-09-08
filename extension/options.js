/*
 * Settings: where the local dashboard is, and which market to show.
 *
 * The address is validated against loopback before it is saved. Not as security theatre —
 * the manifest's host permissions already make a remote address unreachable — but because
 * a typo'd hostname would otherwise fail as an opaque network error, and the honest
 * message is "this extension only talks to your own machine".
 */

const { loadSettings, saveSettings, request, DEFAULT_SETTINGS } = self.swingbot;

const el = (id) => document.getElementById(id);

const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);

function validate(raw) {
  const value = (raw || "").trim() || DEFAULT_SETTINGS.baseUrl;
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error("that is not a valid address — try http://127.0.0.1:8765");
  }
  if (!LOOPBACK.has(url.hostname)) {
    throw new Error(
      `${url.hostname} is not your own machine. This extension can only reach 127.0.0.1 or localhost.`,
    );
  }
  return `${url.origin}`;
}

function show(message, ok = true) {
  const node = el("result");
  node.textContent = message;
  node.style.color = ok ? "var(--accent)" : "var(--sell)";
}

async function init() {
  const settings = await loadSettings();
  el("baseUrl").value = settings.baseUrl;
  el("market").value = settings.market;
}

el("save").addEventListener("click", async () => {
  try {
    const baseUrl = validate(el("baseUrl").value);
    await saveSettings({ baseUrl, market: el("market").value });
    el("baseUrl").value = baseUrl;
    show("saved");
  } catch (error) {
    show(error.message, false);
  }
});

el("test").addEventListener("click", async () => {
  try {
    const baseUrl = validate(el("baseUrl").value);
    show("checking…");
    const health = await request("/api/health", { baseUrl });
    show(`connected — ${health.market || "?"}, ${health.n_runs ?? 0} run(s)`);
  } catch (error) {
    show(error.message, false);
  }
});

init();
