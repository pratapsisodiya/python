"""NSE instrument reference data: ISINs, lot sizes, order-freeze limits.

The Indian market imposes three constraints that do not exist in the US and that a
strategy written for cash equity will quietly violate:

**A single-stock future trades in an exchange-defined lot.** Not shares — lots. And the
lot differs enormously by underlying: RELIANCE is 500, IOC is 4,875. Any single
market-wide number is wrong for almost every name, and rounding a small lot up to a large
one multiplies the intended exposure. So the lots come from the exchange's own instrument
list, per symbol.

**One lot has a minimum notional, and it can exceed the whole position budget.** RELIANCE
at 1,295 in a 500-lot is 647,500 of exposure. Under a 12 percent per-name cap that
position is only reachable on an account north of five crore. Below that, the name is not
"expensive to short" — it is *impossible* to short, and no amount of sizing logic changes
that. :func:`capital_adequacy` is the function that says so out loud, because the
alternative is finding out from a rejected order.

**NSE refuses an order above the freeze quantity.** A single ticket for more than the
per-symbol freeze limit is rejected outright and has to be split. It is carried here so
the order file can flag it rather than have the broker do it.

The snapshot is committed to the repository on purpose. Lot sizes are revised by the
exchange, so a backtest that silently picked up today's lots would not reproduce; a dated
file that the user refreshes deliberately does. It is a *current* snapshot, not
point-in-time history — the same honest limitation the universe membership file carries,
and stated the same way.
"""

from __future__ import annotations

import collections
import gzip
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

#: Upstox publish the exchange's instrument list as a public, unauthenticated file. Used
#: rather than NSE's own endpoint because nseindia.com requires a browser cookie
#: handshake and refuses plain clients with a 403.
NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

DEFAULT_SNAPSHOT = Path("config/universe/nse_instruments.csv")

COLUMNS = (
    "ticker",
    "name",
    "isin",
    "instrument_key",
    "lot_size",
    "freeze_quantity",
    "snapshot_date",
)


@dataclass(frozen=True, slots=True)
class Instrument:
    ticker: str
    name: str
    isin: str
    instrument_key: str
    #: F&O lot for the nearest expiry, or 0 when the name has no futures contract.
    lot_size: int
    #: Largest quantity NSE will accept on one ticket, or 0 when unknown.
    freeze_quantity: int

    @property
    def is_shortable_via_futures(self) -> bool:
        return self.lot_size > 0

    def one_lot_notional(self, price: float) -> float:
        return self.lot_size * price


class NSEInstruments:
    """Reference data for NSE symbols, read from a committed snapshot."""

    name = "nse_instruments"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self._by_ticker = {
            str(row.ticker): Instrument(
                ticker=str(row.ticker),
                name=str(row.name_),
                isin=str(row.isin),
                instrument_key=str(row.instrument_key),
                lot_size=int(row.lot_size or 0),
                freeze_quantity=int(row.freeze_quantity or 0),
            )
            for row in frame.rename(columns={"name": "name_"}).itertuples()
        }

    # ----------------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: Path | str = DEFAULT_SNAPSHOT) -> NSEInstruments:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No NSE instrument snapshot at {path}. Run `swingbot fetch instruments "
                "--market india` to download one."
            )
        frame = pd.read_csv(path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        return cls(frame)

    @classmethod
    def load_if_present(cls, path: Path | str = DEFAULT_SNAPSHOT) -> NSEInstruments | None:
        """As :meth:`load`, but returns None instead of raising.

        Used by the pipeline, which must keep working for anyone who has not refreshed the
        snapshot — the consequence is unrounded futures quantities and a loud flag, not a
        crash.
        """
        try:
            return cls.load(path)
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            log.warning("NSE instrument snapshot unavailable (%s)", exc)
            return None

    # ------------------------------------------------------------------------ public

    def get(self, ticker: str) -> Instrument | None:
        return self._by_ticker.get(ticker)

    def instrument_key(self, ticker: str) -> str | None:
        found = self._by_ticker.get(ticker)
        return found.instrument_key if found else None

    def instrument_keys(self, tickers: list[str]) -> dict[str, str]:
        return {t: k for t in tickers if (k := self.instrument_key(t))}

    def lot_sizes(self) -> dict[str, int]:
        """Per-symbol F&O lot, for names that have a futures contract."""
        return {
            t: i.lot_size for t, i in self._by_ticker.items() if i.lot_size > 1
        }

    def freeze_quantities(self) -> dict[str, int]:
        return {
            t: i.freeze_quantity for t, i in self._by_ticker.items() if i.freeze_quantity > 0
        }

    @property
    def snapshot_date(self) -> str:
        column = self._frame.get("snapshot_date")
        return str(column.iloc[0]) if column is not None and len(column) else ""

    def __len__(self) -> int:
        return len(self._by_ticker)


# --------------------------------------------------------------------------------------
# Refreshing the snapshot
# --------------------------------------------------------------------------------------


def fetch_instruments(url: str = NSE_INSTRUMENTS_URL, *, timeout: float = 60.0) -> list[dict]:
    """Download and decompress the exchange instrument list."""
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return json.loads(gzip.decompress(response.content))


def build_snapshot(rows: list[dict], *, tickers: list[str] | None = None) -> pd.DataFrame:
    """Reduce the exchange dump to the columns this system uses.

    The lot comes from the *nearest-expiry* future for each underlying. Later expiries can
    carry a revised lot, and the one that governs an order placed this week is the front
    month's.
    """
    equities = {
        row["trading_symbol"]: row
        for row in rows
        if row.get("segment") == "NSE_EQ" and row.get("instrument_type") == "EQ"
    }

    futures: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if (
            row.get("segment") == "NSE_FO"
            and row.get("instrument_type") == "FUT"
            and row.get("asset_symbol")
        ):
            futures[row["asset_symbol"]].append(row)

    lots = {
        symbol: int(min(contracts, key=lambda r: r.get("expiry") or 0).get("lot_size") or 0)
        for symbol, contracts in futures.items()
    }

    wanted = tickers if tickers is not None else sorted(equities)
    today = datetime.now(UTC).date().isoformat()

    records = []
    for ticker in wanted:
        equity = equities.get(ticker)
        if equity is None:
            continue
        records.append({
            "ticker": ticker,
            "name": equity.get("name", ticker),
            "isin": equity.get("isin", ""),
            "instrument_key": equity.get("instrument_key", ""),
            "lot_size": lots.get(ticker, 0),
            "freeze_quantity": int(equity.get("freeze_quantity") or 0),
            "snapshot_date": today,
        })

    return pd.DataFrame(records, columns=list(COLUMNS))


def refresh_snapshot(
    path: Path | str = DEFAULT_SNAPSHOT,
    *,
    tickers: list[str] | None = None,
    url: str = NSE_INSTRUMENTS_URL,
) -> tuple[Path, int, int]:
    """Download the instrument list and write the snapshot.

    Returns the path, how many symbols were written, and how many of those carry an F&O
    lot. The second number matters: a name with no futures contract cannot be shorted
    weekly on NSE at all, whatever the model thinks of it.
    """
    frame = build_snapshot(fetch_instruments(url), tickers=tickers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    with_lots = int((frame["lot_size"] > 1).sum())
    log.info("wrote %d NSE instrument(s) to %s, %d with an F&O lot", len(frame), path, with_lots)
    return path, len(frame), with_lots


# --------------------------------------------------------------------------------------
# Capital adequacy — the constraint nobody expects
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class LotConstraint:
    ticker: str
    lot_size: int
    price: float
    one_lot_notional: float
    #: Account equity at which one lot first fits inside the per-name weight cap.
    minimum_equity: float
    tradeable: bool


@dataclass(slots=True)
class CapitalAdequacy:
    """Whether the shortable half of the book is reachable at a given account size."""

    equity: float
    max_weight: float
    constraints: list[LotConstraint]
    snapshot_date: str = ""

    @property
    def tradeable(self) -> list[LotConstraint]:
        return [c for c in self.constraints if c.tradeable]

    @property
    def blocked(self) -> list[LotConstraint]:
        return [c for c in self.constraints if not c.tradeable]

    @property
    def minimum_equity_for_any(self) -> float:
        """Smallest account at which at least one name becomes shortable."""
        return min((c.minimum_equity for c in self.constraints), default=0.0)

    def verdict(self) -> str:
        if not self.constraints:
            return "no lot sizes on file, so nothing can be checked"
        n_ok = len(self.tradeable)
        if n_ok == 0:
            return (
                f"NOT ONE shortable name fits a {self.max_weight:.0%} position cap at "
                f"{self.equity:,.0f} of equity. One lot of the cheapest F&O name is "
                f"{min(c.one_lot_notional for c in self.constraints):,.0f}, so the short "
                f"sleeve needs about {self.minimum_equity_for_any:,.0f} to exist at all. "
                "Below that, run the strategy long-only "
                "(--set market_profile.short_instrument=none) rather than generating "
                "shorts you cannot place."
            )
        if n_ok < 7:
            return (
                f"only {n_ok} of {len(self.constraints)} shortable names fit the "
                f"{self.max_weight:.0%} cap at {self.equity:,.0f} of equity. The short "
                "sleeve will be concentrated in whatever is cheapest rather than in what "
                "the model dislikes most, which is a different strategy than the one that "
                "was backtested."
            )
        return (
            f"{n_ok} of {len(self.constraints)} shortable names fit the "
            f"{self.max_weight:.0%} cap at {self.equity:,.0f} of equity"
        )

    def to_dict(self) -> dict:
        return {
            "equity": self.equity,
            "max_weight": self.max_weight,
            "snapshot_date": self.snapshot_date,
            "n_shortable": len(self.constraints),
            "n_tradeable": len(self.tradeable),
            "minimum_equity_for_any": self.minimum_equity_for_any,
            "verdict": self.verdict(),
            "blocked": [
                {
                    "ticker": c.ticker,
                    "lot_size": c.lot_size,
                    "one_lot_notional": c.one_lot_notional,
                    "minimum_equity": c.minimum_equity,
                }
                for c in sorted(self.blocked, key=lambda c: c.minimum_equity)
            ],
        }


def capital_adequacy(
    instruments: NSEInstruments,
    prices: dict[str, float],
    *,
    equity: float,
    max_weight: float,
    tickers: list[str] | None = None,
) -> CapitalAdequacy:
    """Which shortable names one lot of actually fits inside the position cap.

    The arithmetic is trivial and the conclusion is not: a per-name cap and an indivisible
    lot together set a hard floor on account size that no sizing rule can work around. A
    system that sized a 3 percent short in RELIANCE and emitted the order would be
    describing a trade that cannot be placed.
    """
    names = tickers if tickers is not None else sorted(instruments.lot_sizes())
    constraints = []

    for ticker in names:
        instrument = instruments.get(ticker)
        price = prices.get(ticker)
        if instrument is None or not instrument.is_shortable_via_futures or not price:
            continue
        notional = instrument.one_lot_notional(price)
        constraints.append(
            LotConstraint(
                ticker=ticker,
                lot_size=instrument.lot_size,
                price=float(price),
                one_lot_notional=notional,
                minimum_equity=notional / max_weight if max_weight > 0 else float("inf"),
                tradeable=bool(equity > 0 and notional <= equity * max_weight),
            )
        )

    return CapitalAdequacy(
        equity=equity,
        max_weight=max_weight,
        constraints=constraints,
        snapshot_date=instruments.snapshot_date,
    )
