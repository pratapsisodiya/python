"""Charts for the tearsheet, rendered to inline SVG.

SVG embedded directly in the HTML rather than PNG files alongside it, so the report is a
single self-contained file that survives being emailed, moved or archived. A tearsheet
that breaks when its image folder is left behind is not a record of anything.

Matplotlib with the Agg backend, so this works on a headless machine and in cron.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Muted palette that stays legible printed, projected, and in a dark-mode mail client.
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
POSITIVE = "#2f7d5d"
NEGATIVE = "#b04a3f"
ACCENT = "#3d6a99"
FILL = "#c9d6e3"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(MUTED)


def _to_svg(fig) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    svg = buffer.getvalue()
    # Drop the XML preamble so the fragment embeds cleanly in HTML.
    return svg[svg.index("<svg") :] if "<svg" in svg else svg


def equity_chart(
    curves: dict[str, pd.Series], *, width: float = 9.0, height: float = 3.4
) -> str:
    """Equity curves on a log scale.

    Log scale because a linear axis makes a decade of compounding look like all the
    action happened at the end, and hides an early drawdown that would have ended the
    strategy in practice.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    colours = [ACCENT, MUTED, POSITIVE, NEGATIVE, "#8a6d3b", "#6b5b95"]

    for i, (name, series) in enumerate(curves.items()):
        clean = series.dropna()
        if clean.empty:
            continue
        index = pd.to_datetime(clean.index)
        primary = i == 0
        ax.plot(
            index, clean.to_numpy(),
            linewidth=1.9 if primary else 1.1,
            color=colours[i % len(colours)],
            alpha=1.0 if primary else 0.75,
            label=name,
            zorder=3 if primary else 2,
        )

    ax.set_yscale("log")
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel("growth of 1", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncols=3, loc="upper left")
    _style(ax)
    return _to_svg(fig)


def drawdown_chart(drawdown: pd.Series, *, width: float = 9.0, height: float = 2.0) -> str:
    fig, ax = plt.subplots(figsize=(width, height))
    clean = drawdown.dropna()
    if not clean.empty:
        index = pd.to_datetime(clean.index)
        values = clean.to_numpy() * 100.0
        ax.fill_between(index, values, 0, color=NEGATIVE, alpha=0.22, zorder=2)
        ax.plot(index, values, color=NEGATIVE, linewidth=1.2, zorder=3)
    ax.set_ylabel("drawdown %", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax)
    return _to_svg(fig)


def rolling_sharpe_chart(
    returns: pd.Series, *, window: int = 52, width: float = 9.0, height: float = 2.0
) -> str:
    """Rolling one-year Sharpe.

    The chart that shows whether an edge was persistent or came from one good year. A
    strategy whose entire Sharpe is a 2020 spike is a different proposition from one that
    worked steadily, and the headline number cannot distinguish them.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    clean = returns.dropna()
    if len(clean) > window:
        import numpy as np

        rolling = (
            clean.rolling(window).mean() / clean.rolling(window).std()
        ) * np.sqrt(52.0)
        rolling = rolling.dropna()
        index = pd.to_datetime(rolling.index)
        ax.plot(index, rolling.to_numpy(), color=ACCENT, linewidth=1.3, zorder=3)
        ax.fill_between(
            index, rolling.to_numpy(), 0,
            where=rolling.to_numpy() >= 0, color=POSITIVE, alpha=0.15,
        )
        ax.fill_between(
            index, rolling.to_numpy(), 0,
            where=rolling.to_numpy() < 0, color=NEGATIVE, alpha=0.15,
        )
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_ylabel(f"rolling {window}w Sharpe", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax)
    return _to_svg(fig)


def decile_chart(deciles: pd.Series, *, width: float = 4.4, height: float = 2.6) -> str:
    """Mean realised return by forecast decile.

    The single most informative diagnostic in the report. A real signal produces a
    monotone ladder; a spread driven by one extreme bucket with noise between is usually
    a handful of outliers rather than an effect.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    if not deciles.empty:
        values = deciles.to_numpy() * 100.0
        colours = [NEGATIVE if v < 0 else POSITIVE for v in values]
        ax.bar(range(len(values)), values, color=colours, alpha=0.85, zorder=3)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(i + 1) for i in range(len(values))], fontsize=7)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("forecast decile (1 = lowest)", color=MUTED, fontsize=8)
    ax.set_ylabel("mean fwd return %", color=MUTED, fontsize=8)
    _style(ax)
    return _to_svg(fig)


def ic_chart(ic: pd.Series, *, width: float = 4.4, height: float = 2.6) -> str:
    fig, ax = plt.subplots(figsize=(width, height))
    clean = ic.dropna()
    if not clean.empty:
        index = pd.to_datetime(clean.index)
        ax.bar(
            index, clean.to_numpy(), width=5.0,
            color=[POSITIVE if v >= 0 else NEGATIVE for v in clean.to_numpy()],
            alpha=0.6, zorder=2,
        )
        rolling = clean.rolling(26, min_periods=10).mean()
        ax.plot(index, rolling.to_numpy(), color=INK, linewidth=1.4, zorder=3)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("weekly IC", color=MUTED, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax)
    return _to_svg(fig)


def cost_chart(breakdown: dict[str, float], *, width: float = 4.4, height: float = 2.6) -> str:
    fig, ax = plt.subplots(figsize=(width, height))
    items = [(k, v) for k, v in breakdown.items() if k != "total" and v > 0]
    if items:
        items.sort(key=lambda kv: -kv[1])
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        ax.barh(range(len(values)), values, color=ACCENT, alpha=0.8, zorder=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
    ax.set_xlabel("cumulative cost", color=MUTED, fontsize=8)
    _style(ax)
    return _to_svg(fig)
