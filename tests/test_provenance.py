"""Where every price came from, and whether anyone can tell.

The provider chain's job is to fall through until something answers. That is the right
design for a system meant to keep working when a vendor is down, and it has one dangerous
consequence: the last thing in a live chain used to be a random-walk generator, which
answers for *anything*. "The fetch succeeded" therefore said nothing about whether the
prices were real.

Both ways it actually went wrong are pinned here.

Three renamed NSE symbols — ZOMATO, TATAMOTORS, LTIM — were silently filled in by the
generator and sat inside a 129-name "real" dataset. And `swingbot demo` writes its
generated CSVs into the same directory a user's own exports go, where they shadow
everything: a completed real fetch of 128 NSE names was quietly ignored in favour of
leftover demo files, and the Sharpe on screen was measuring the generator.

Neither produced an error. Neither showed up in a report. That is the failure mode this
codebase treats as worse than a crash, so provenance is now recorded, survives the parquet
cache, and reaches the caveat list that every report prints.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from swingbot.data.chain import ProviderChain
from swingbot.data.prices_csv import GENERATED_MARKER, CSVProvider, write_csv_fixture
from swingbot.data.synthetic import SyntheticProvider
from swingbot.io.store import ParquetStore
from swingbot.types import TICKER

START = dt.date(2022, 1, 3)
END = dt.date(2023, 12, 29)


class OnlyKnowsSome:
    """A real provider that covers part of the universe, like every real provider."""

    name = "partial"

    def __init__(self, known: list[str]) -> None:
        self.known = list(known)
        self._inner = SyntheticProvider(seed=1, close_hour_utc=21)

    def available(self) -> bool:
        return True

    def daily_bars(self, tickers, start, end):
        wanted = [t for t in tickers if t in self.known]
        if not wanted:
            return pd.DataFrame()
        return self._inner.daily_bars(wanted, start, end)


# --------------------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------------------


def test_each_ticker_is_attributed_to_the_provider_that_answered(tmp_path):
    chain = ProviderChain(
        [OnlyKnowsSome(["AAA", "BBB"]), SyntheticProvider(seed=2, close_hour_utc=21)],
        store=ParquetStore(tmp_path),
    )
    chain.daily_bars(["AAA", "BBB", "CCC"], START, END)

    assert chain.attribution == {"AAA": "partial", "BBB": "partial", "CCC": "synthetic"}


def test_only_generated_tickers_are_reported_as_fabricated(tmp_path):
    """The question that matters: which of these prices did nobody observe."""
    chain = ProviderChain(
        [OnlyKnowsSome(["AAA", "BBB"]), SyntheticProvider(seed=2, close_hour_utc=21)],
        store=ParquetStore(tmp_path),
    )
    chain.daily_bars(["AAA", "BBB", "CCC"], START, END)

    assert chain.fabricated_tickers() == ["CCC"]


def test_a_chain_with_no_generator_leaves_a_name_with_no_data(tmp_path):
    """The fix for the NSE case: absent beats invented.

    LTIM has no history at any real source. With no generator in the chain it simply has
    no bars, the liquidity screen drops it, and nothing about it is fiction. That is the
    correct outcome and it is why the India profile ships without `synthetic`.
    """
    chain = ProviderChain([OnlyKnowsSome(["AAA", "BBB"])], store=ParquetStore(tmp_path))
    bars = chain.daily_bars(["AAA", "BBB", "LTIM"], START, END)

    assert sorted(bars[TICKER].unique()) == ["AAA", "BBB"]
    assert chain.fabricated_tickers() == []
    assert "LTIM" not in chain.attribution


# --------------------------------------------------------------------------------------
# Surviving the cache
# --------------------------------------------------------------------------------------


def test_provenance_survives_the_parquet_cache(tmp_path):
    """The defect that made the caveat unable to fire.

    Attribution was recorded at fetch time and dropped on the way into parquet. On the
    next run every ticker came back labelled "cache", so a dataset holding three invented
    names alongside 126 real ones was indistinguishable from a wholly real one.
    """
    store = ParquetStore(tmp_path)
    first = ProviderChain(
        [OnlyKnowsSome(["AAA"]), SyntheticProvider(seed=3, close_hour_utc=21)],
        store=store,
    )
    first.daily_bars(["AAA", "CCC"], START, END)
    assert first.fabricated_tickers() == ["CCC"]

    # A fresh chain over the same store: nothing is fetched, everything is a cache hit.
    second = ProviderChain([], store=store)
    second.daily_bars(["AAA", "CCC"], START, END)

    assert second.attribution == {"AAA": "partial", "CCC": "synthetic"}
    assert second.fabricated_tickers() == ["CCC"], "the cache laundered generated prices"


def test_the_sidecar_is_merged_rather_than_replaced(tmp_path):
    """A partial refresh must not make the rest of the cache look clean."""
    store = ParquetStore(tmp_path)
    ProviderChain(
        [OnlyKnowsSome(["AAA"]), SyntheticProvider(seed=4, close_hour_utc=21)], store=store
    ).daily_bars(["AAA", "CCC"], START, END)

    # Refetch one ticker only.
    ProviderChain([OnlyKnowsSome(["AAA"])], store=store).daily_bars(
        ["AAA"], START, END, refresh=True
    )

    stored = json.loads((tmp_path / "raw" / "attribution.json").read_text())["providers"]
    assert stored == {"AAA": "partial", "CCC": "synthetic"}


def test_cache_is_never_recorded_as_a_source(tmp_path):
    """It is where an answer was kept, not who gave it."""
    store = ParquetStore(tmp_path)
    ProviderChain([OnlyKnowsSome(["AAA"])], store=store).daily_bars(["AAA"], START, END)
    ProviderChain([], store=store).daily_bars(["AAA"], START, END)

    stored = json.loads((tmp_path / "raw" / "attribution.json").read_text())["providers"]
    assert "cache" not in stored.values()


# --------------------------------------------------------------------------------------
# Generated CSVs, which look exactly like real ones
# --------------------------------------------------------------------------------------


def test_a_marked_csv_directory_renames_the_provider(tmp_path):
    """A generated CSV is byte-identical to a real export, so the directory is marked.

    Without this, `swingbot demo` poisons the same directory a user drops real exports
    into and the chain reports "csv supplied 129 tickers" either way — which is exactly
    how a completed real NSE fetch came to be ignored in favour of stale demo files.
    """
    directory = tmp_path / "csv"
    bars = SyntheticProvider(seed=5, close_hour_utc=21).daily_bars(["AAA"], START, END)
    write_csv_fixture(bars, directory)

    provider = CSVProvider(directory)
    assert provider.name == "csv", "an unmarked directory is assumed to be real data"
    assert provider.is_generated is False

    (directory / GENERATED_MARKER).write_text('{"generated_by": "swingbot demo"}')
    assert provider.name == "synthetic-csv"
    assert provider.is_generated is True


def test_generated_csvs_are_reported_as_fabricated(tmp_path):
    """And therefore reach the caveat list, like any other generated price."""
    directory = tmp_path / "csv"
    bars = SyntheticProvider(seed=6, close_hour_utc=21).daily_bars(["AAA", "BBB"], START, END)
    write_csv_fixture(bars, directory)
    (directory / GENERATED_MARKER).write_text('{"generated_by": "swingbot demo"}')

    chain = ProviderChain([CSVProvider(directory)], store=ParquetStore(tmp_path / "store"))
    chain.daily_bars(["AAA", "BBB"], START, END)

    assert chain.fabricated_tickers() == ["AAA", "BBB"]


def test_demo_marks_the_directory_it_writes(tmp_path):
    """End to end, through the command a beginner runs first."""
    from swingbot.config import load_config
    from swingbot.pipeline import generate_demo_data

    cfg = load_config(
        "us", set_values=[f"run.data_dir={tmp_path}", "universe.max_names=6"]
    )
    directory = generate_demo_data(cfg, years=3, n_names=6)

    marker = directory / GENERATED_MARKER
    assert marker.exists(), "the demo must declare that its prices are generated"

    payload = json.loads(marker.read_text())
    assert payload["generated_by"] == "swingbot demo"
    assert "SYNTHETIC" in payload["note"]
    assert CSVProvider(directory).name == "synthetic-csv"


# --------------------------------------------------------------------------------------
# The caveat that reaches the report
# --------------------------------------------------------------------------------------


def test_a_demo_dataset_says_so_in_the_report(tmp_path):
    """Because the number people remember is the Sharpe, not the README.

    A demo run posts a very high Sharpe — the generator plants the signal deliberately —
    so the report has to say the figure measures the generator.
    """
    from swingbot.config import load_config
    from swingbot.pipeline import build_market_data, generate_demo_data

    cfg = load_config(
        "us", set_values=[f"run.data_dir={tmp_path}", "universe.max_names=25"]
    )
    generate_demo_data(cfg, years=4, n_names=25)
    data = build_market_data(cfg, with_labels=False)

    demo_caveats = [c for c in data.caveats if "DEMO DATA" in c]
    assert demo_caveats, f"a fully generated dataset must say so; got {data.caveats}"
    assert "measurement of the generator" in demo_caveats[0]


def test_a_partly_generated_dataset_names_the_invented_symbols(tmp_path):
    """Mixed real and fake is the more dangerous case, so the names are listed."""
    from swingbot.config import load_config
    from swingbot.pipeline import build_market_data

    cfg = load_config(
        "us",
        set_values=[
            f"run.data_dir={tmp_path}",
            "data.providers=[csv,synthetic]",
            "data.start=2019-01-02",
            "universe.max_names=24",
            "features.min_names_per_week=10",
        ],
    )

    # Real files for all but the last four names of the universe, so the generator has to
    # fill exactly those — the shape of the NSE case, where a handful of renamed symbols
    # were invented alongside a mostly genuine dataset.
    from swingbot.data import load_universe

    tickers = load_universe(cfg).all_tickers()
    supplied, invented = tickers[:-4], tickers[-4:]

    directory = tmp_path / "us" / "csv"
    write_csv_fixture(
        SyntheticProvider(seed=7, close_hour_utc=21).daily_bars(
            supplied, dt.date(2019, 1, 2), END
        ),
        directory,
    )

    data = build_market_data(cfg, with_labels=False)

    mixed = [c for c in data.caveats if "SYNTHETIC prices" in c]
    assert mixed, f"a partly generated dataset must say which names; got {data.caveats}"
    assert "mixed in with real ones" in mixed[0]
    assert f"{len(invented)} of {len(tickers)}" in mixed[0]
    assert any(ticker in mixed[0] for ticker in invented), (
        "the caveat has to name the invented symbols; a count alone leaves the user "
        "unable to find them"
    )


# --------------------------------------------------------------------------------------
# `doctor`, which is where a user asks whether their setup is sound
# --------------------------------------------------------------------------------------


def test_doctor_says_where_the_prices_came_from(tmp_path):
    """Coverage without provenance is a volume statistic, not a truth statement.

    `doctor` could report 128 tickers and 272,000 bars while every one of them came out of
    a random-walk generator. It is a fact about the dataset rather than about a particular
    week, so it belongs here beside the calendar and not only in a run's caveat list.
    """
    from swingbot.config import load_config
    from swingbot.pipeline import generate_demo_data
    from swingbot.service import doctor_report

    cfg = load_config("us", set_values=[f"run.data_dir={tmp_path}", "universe.max_names=8"])
    generate_demo_data(cfg, years=3, n_names=8)

    prov = doctor_report(cfg).provenance

    assert prov["sources"] == {"synthetic-csv": prov["n_tickers"]}
    assert prov["n_real"] == 0
    assert prov["clean"] is False
    assert "DEMO DATA" in prov["verdict"]


def test_doctor_says_so_plainly_when_every_price_is_real(tmp_path):
    """The other half of the check: a clean dataset must be recognisable as clean.

    A verdict that only ever warns is one nobody reads. This is the case the India profile
    now produces, and it has to be distinguishable from "we never asked".
    """
    from swingbot.config import load_config
    from swingbot.data import load_universe
    from swingbot.service import doctor_report

    cfg = load_config(
        "us",
        set_values=[
            f"run.data_dir={tmp_path}",
            "data.providers=[csv]",
            "data.start=2019-01-02",
            "universe.max_names=12",
            "features.min_names_per_week=6",
        ],
    )
    # Unmarked CSVs — i.e. what a real export looks like.
    write_csv_fixture(
        SyntheticProvider(seed=8, close_hour_utc=21).daily_bars(
            load_universe(cfg).all_tickers(), dt.date(2019, 1, 2), END
        ),
        tmp_path / "us" / "csv",
    )

    prov = doctor_report(cfg).provenance

    assert prov["clean"] is True
    assert prov["n_fabricated"] == 0
    assert prov["fabricated"] == []
    assert prov["verdict"].startswith(f"all {prov['n_tickers']} names")


@pytest.mark.parametrize("market", ["india"])
def test_the_india_profile_ships_without_a_generator(market):
    """The configuration change that stops the contamination at source."""
    from swingbot.config import load_config

    providers = load_config(market).data.providers
    assert "synthetic" not in providers, (
        "a generator at the end of a live chain invents prices for any symbol a real "
        "source cannot supply, which is how three renamed NSE tickers got fabricated"
    )
    assert "upstox" in providers
