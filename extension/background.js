/*
 * Background worker: keeps the toolbar badge honest.
 *
 * The badge shows how many of this week's orders are still unplaced, so a new signal run
 * is visible without opening the popup. That is the whole job. It deliberately does not
 * notify, nag, or act — an extension that pops a "place your trades" alert would be
 * pressuring someone into a market order, which is the opposite of what a weekly strategy
 * wants.
 *
 * Chrome runs this as an MV3 service worker (terminated when idle, woken by the alarm);
 * Firefox runs the same file as an event page. Nothing here holds state between wakes, so
 * both behave identically.
 */

importScripts("client.js");

const { loadSettings, fetchLatestOrders, loadPlaced, orderId } = self.swingbot;

const ALARM = "swingbot-poll";

/*
 * Fifteen minutes. The underlying thing changes at most once a week, so polling exists
 * only to notice a run the user just kicked off from the dashboard. Anything faster would
 * spend battery to watch a number that is almost always unchanged.
 */
const POLL_MINUTES = 15;

async function setBadge(count) {
  try {
    await chrome.action.setBadgeText({ text: count > 0 ? String(count) : "" });
    await chrome.action.setBadgeBackgroundColor({ color: "#1f7a4d" });
  } catch (error) {
    console.warn("swingbot: badge update failed", error);
  }
}

async function refresh() {
  let settings;
  try {
    settings = await loadSettings();
    const payload = await fetchLatestOrders(settings);
    if (!payload) {
      await setBadge(0);
      return;
    }
    const placed = await loadPlaced(payload.run_id);
    const remaining = payload.orders.filter((order) => !placed.has(orderId(order))).length;
    await setBadge(remaining);
  } catch {
    // The dashboard being off is the normal state most of the week, not an error worth
    // showing. Clear the badge so a stale count never implies there is work to do.
    await setBadge(0);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create(ALARM, { periodInMinutes: POLL_MINUTES, delayInMinutes: 1 });
  refresh();
});

chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create(ALARM, { periodInMinutes: POLL_MINUTES, delayInMinutes: 1 });
  refresh();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM) refresh();
});

/*
 * There is no message listener here on purpose. The popup sets the badge itself the moment
 * a row is ticked, because a round trip through this worker would show a stale count for
 * as long as Chrome takes to wake it.
 */
