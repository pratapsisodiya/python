# swingbot order helper (browser extension)

A small browser extension that shows you this week's swingbot orders in a checklist while
you place them at your broker.

It reads the orders from the swingbot dashboard running on **your own machine**. It has no
account, no server, and no access to anything outside `127.0.0.1`. It never signs in to
your broker and never places a trade.

---

## What it does

- Lists this week's orders: ticker, side, quantity, instrument, estimated value.
- One **copy** button per row that gives you the ticker, then the quantity, in the order a
  broker's order ticket asks for them.
- A tick box per row, so you can see what you have already placed. The list survives
  closing the popup, and it syncs both ways with the dashboard.
- A count on the toolbar icon of orders still to place, so a new weekly signal is visible
  without opening anything.
- A loud banner when a row is a **futures** short (India), because that must go on the F&O
  segment rather than equity delivery.
- The portfolio's own notes — a capacity cap that truncated orders, a risk limit that bound,
  a Kelly fallback — shown right above the list, so an unexpectedly small book explains
  itself before you act on it.
- A paste box for `orders.csv`, for when the dashboard is not running.

## What it deliberately does not do

**It does not fill in your broker's order ticket for you.** That would need permission to
run code on live brokerage pages, it would break every time a broker changed its markup,
and a bug in that code would place a wrong trade with real money. Copying two fields per
row is a few seconds slower, works at every broker without modification, and cannot
misfire.

This is the same reasoning that keeps swingbot's core broker-independent, applied one layer
out. If you want automation, automate the *decision* (a scheduled `swingbot run-weekly`,
see the main README) — not the click that sends money to an exchange.

---

## Install

You are loading it from source, unpacked. It is not on any extension store.

### Chrome, Edge, Brave

1. Start the dashboard: `swingbot serve` (see the main README, "Step 5: the web dashboard").
2. Open `chrome://extensions`.
3. Turn on **Developer mode** (top right).
4. Click **Load unpacked** and choose this `extension/` folder.
5. Pin "swingbot order helper" to the toolbar and click it.

### Firefox

Firefox needs a slightly different manifest (it uses an event page where Chrome uses a
service worker), so swap the manifest first:

```bash
cd extension
cp manifest.json manifest.chrome.json      # keep the Chrome one
cp manifest.firefox.json manifest.json
```

1. Open `about:debugging#/runtime/this-firefox`.
2. Click **Load Temporary Add-on** and choose `extension/manifest.json`.

Firefox drops temporary add-ons when it restarts, so you will need to load it again next
time. To go back to Chrome, restore `manifest.chrome.json` over `manifest.json`.

---

## Settings

Click the gear in the popup, or the extension's "Extension options".

| Setting | Default | Notes |
| --- | --- | --- |
| Dashboard address | `http://127.0.0.1:8765` | Must be loopback. Change the port here if you ran `swingbot serve --port`. |
| Market | `us` | Which market's orders to show. Also switchable from the popup. |

**Test connection** tells you whether the dashboard is reachable before you go looking for
orders.

---

## Using it on a Friday

1. Run the weekly signal — `swingbot signal --market us`, or the button in the dashboard.
2. Click the extension. The badge shows how many orders are waiting.
3. Open your broker in another tab.
4. For each row: click **copy** (ticker), paste into the symbol field, click **copy** again
   (quantity), paste into the quantity field, set the side to match, and place it.
5. Tick the row. The badge goes down.
6. When the badge is empty, you are done. The dashboard records the same checklist in the
   run folder, so next week you can see what you actually placed.

If a row says **futures · F&O segment**, place it on the derivatives segment. If it says
**short sale · borrow required**, your broker needs to have the stock available to borrow;
if it does not, skip the row and note it.

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| "could not reach http://127.0.0.1:8765" | `swingbot serve` is not running, or is on a different port. |
| "No signal run yet for this market" | You have not run `swingbot signal` for that market. |
| Ticks disappeared | They are stored per run id. A new signal run starts a fresh checklist — that is intentional, last week's ticks are not this week's. |
| Copy does nothing | The browser refused clipboard access; the popup shows the value in the toast so you can read it off. |
| Badge shows a count after you finished | Ticks made in the dashboard while the popup was closed sync on the next poll (up to 15 minutes). Click refresh in the popup to force it. |
