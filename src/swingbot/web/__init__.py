"""The local dashboard.

A small FastAPI app that runs on the user's own machine, so the weekly output can be
*looked at* rather than read out of terminal scrollback. It calls
:mod:`swingbot.service` — the same functions the CLI calls — so nothing here decides a
position or computes a number.

Three properties this package is required to keep, each of them enforced by a test in
``tests/test_architecture.py`` rather than by good intentions:

**It is read-only with respect to a broker.** No credential, no venue, no order
submission. The dashboard runs the research pipeline and shows you what it produced; the
trade is placed by you, at your broker. ``test_the_web_layer_submits_no_orders`` fails the
build if a route ever grows an order-submission path.

**Nothing in the core imports it.** The dependency arrow points one way: the web layer
knows about the pipeline, the pipeline does not know a server exists. That is what keeps
the research path reproducible from a bare `python -c`, and
``test_nothing_outside_the_web_layer_imports_it`` keeps it that way.

**FastAPI is optional.** ``import swingbot.web`` works with no web dependencies
installed, because the imports happen inside :func:`create_app`. Someone who only wants
the CLI should not be made to install a web stack, and the clean-import test asserts it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from ..config import Config

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "create_app", "serve"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_MISSING = (
    "The dashboard needs FastAPI and uvicorn, which are optional.\n\n"
    "    pip install 'swingbot[web]'\n\n"
    "Everything else in swingbot works without them."
)


def _require(module: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised by hand, not in CI
        raise RuntimeError(_MISSING) from exc


def create_app(
    cfg: Config,
    *,
    runs_dir: Path | str | None = None,
    profile: str | None = None,
    set_values: list[str] | None = None,
) -> FastAPI:
    """Build the dashboard application for one market's config.

    ``profile`` and ``set_values`` should be whatever the caller resolved ``cfg`` from, so
    that a request for a different market re-resolves through the same layers instead of
    falling back to the shipped defaults.
    """
    _require("fastapi")
    from .app import build_app

    return build_app(cfg, runs_dir=runs_dir, profile=profile, set_values=set_values)


def serve(
    cfg: Config,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    runs_dir: Path | str | None = None,
    profile: str | None = None,
    set_values: list[str] | None = None,
    log_level: str = "warning",
) -> None:
    """Run the dashboard until interrupted.

    ``log_level`` defaults to ``warning`` so uvicorn's per-request access log does not
    bury the pipeline's own log lines, which are the ones a user actually wants to read
    while a backtest is running.
    """
    uvicorn = _require("uvicorn")
    app = create_app(cfg, runs_dir=runs_dir, profile=profile, set_values=set_values)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def is_loopback(host: str) -> bool:
    """Whether a host binds only to this machine.

    Used to decide whether to warn. The dashboard has no authentication and exposes a
    button that runs code, so binding it to a LAN address is a decision someone should
    make deliberately rather than discover.
    """
    return host in {"127.0.0.1", "localhost", "::1", "[::1]"}
