"""Feature transformer protocol and registry.

Two design rules carry the whole feature layer, and both exist to make leakage
impossible rather than merely unlikely.

**Causality per row.** ``transform`` must produce a value for ``(ticker, session)`` that
depends only on rows at or before that session. No cross-date statistics, no
forward-fill across the boundary, no scaler fitted on the full panel. The truncation test
in ``tests/test_lookahead_regression.py`` verifies this by rebuilding features from a
physically truncated panel and demanding bitwise-identical output.

**Auto-discovery.** Transformers register themselves, and the test suite parametrises
over the registry. That means a new leaky feature cannot be added without a test
noticing, which is the only version of this guarantee that survives contact with a
codebase that keeps growing.

Any state a transformer needs (winsorisation bounds, category maps) is fitted in ``fit``
on training rows only, never inside ``transform``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import pandas as pd

from ..types import SESSION, TICKER


@runtime_checkable
class FeatureTransformer(Protocol):
    """Produces one block of features from a bar panel."""

    #: Short identifier, used in logs and in test parametrisation.
    name: str
    #: Columns this transformer adds. Declared so the pipeline can detect collisions.
    outputs: tuple[str, ...]
    #: Sessions of history required before the first non-null output.
    warmup_sessions: int

    def fit(self, panel: pd.DataFrame, train_mask: pd.Series | None = None) -> FeatureTransformer:
        """Fit any internal state on training rows only. Most blocks are stateless."""
        ...

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return ``ticker, session`` plus this block's output columns."""
        ...


class BaseTransformer:
    """Convenience base: stateless ``fit``, output validation, index discipline."""

    name: str = "base"
    outputs: tuple[str, ...] = ()
    warmup_sessions: int = 0

    def fit(self, panel: pd.DataFrame, train_mask: pd.Series | None = None):  # noqa: ARG002
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def _frame(self, panel: pd.DataFrame, columns: dict[str, pd.Series]) -> pd.DataFrame:
        """Assemble the output frame with the key columns first."""
        out = pd.DataFrame(
            {TICKER: panel[TICKER].to_numpy(), SESSION: panel[SESSION].to_numpy()},
            index=panel.index,
        )
        for name, series in columns.items():
            out[name] = pd.to_numeric(series, errors="coerce").to_numpy()
        missing = set(self.outputs) - set(out.columns)
        if missing:
            raise ValueError(
                f"{self.name} declared outputs {sorted(missing)} but did not produce them"
            )
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} outputs={len(self.outputs)}>"


class FeatureRegistry:
    """Registry of transformer factories, so tests can enumerate every block."""

    def __init__(self) -> None:
        self._factories: dict[str, type] = {}

    def register(self, cls: type) -> type:
        name = getattr(cls, "name", None)
        if not name:
            raise ValueError(f"{cls!r} must define a class-level `name`")
        if name in self._factories:
            raise ValueError(f"Duplicate feature transformer name {name!r}")
        self._factories[name] = cls
        return cls

    def build(self, name: str, **kwargs) -> BaseTransformer:
        if name not in self._factories:
            raise KeyError(f"Unknown transformer {name!r}. Known: {sorted(self._factories)}")
        return self._factories[name](**kwargs)

    def build_all(self, names: list[str] | None = None, **kwargs) -> list[BaseTransformer]:
        selected = names or list(self._factories)
        return [self.build(n, **kwargs) for n in selected]

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __iter__(self) -> Iterator[tuple[str, type]]:
        return iter(sorted(self._factories.items()))

    def __len__(self) -> int:
        return len(self._factories)


#: The global registry. Import and decorate to add a block.
REGISTRY = FeatureRegistry()


def register(cls: type) -> type:
    """Class decorator that adds a transformer to the global registry."""
    return REGISTRY.register(cls)


# --------------------------------------------------------------------------------------
# Grouped rolling helpers
# --------------------------------------------------------------------------------------


def grouped_shift(panel: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Per-ticker shift. Always used instead of a bare shift, which crosses tickers."""
    return panel.groupby(TICKER, sort=False)[column].shift(periods)


def grouped_rolling(
    panel: pd.DataFrame,
    column: str,
    window: int,
    how: str = "mean",
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Per-ticker rolling statistic over trailing rows only.

    ``min_periods`` defaults to the full window, so a feature is null until it has
    enough history rather than being computed from three observations and treated as
    equally reliable.
    """
    min_periods = window if min_periods is None else min_periods
    grouped = panel.groupby(TICKER, sort=False)[column]
    rolled = grouped.transform(
        lambda s: getattr(s.rolling(window, min_periods=min_periods), how)()
    )
    return rolled


def grouped_ewm(panel: pd.DataFrame, column: str, span: int) -> pd.Series:
    grouped = panel.groupby(TICKER, sort=False)[column]
    return grouped.transform(lambda s: s.ewm(span=span, adjust=False, min_periods=span).mean())


def ensure_sorted(panel: pd.DataFrame) -> pd.DataFrame:
    """Sort by ticker then session.

    Every rolling computation assumes this ordering, and an unsorted panel produces
    silently wrong numbers rather than an error, so it is enforced at the entry point.
    """
    return panel.sort_values([TICKER, SESSION]).reset_index(drop=True)
