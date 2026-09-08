"""Architectural boundaries, enforced by scanning the AST rather than by convention.

The claim this system makes is that the research core is broker-independent. That is only
worth anything if something checks it, because the boundary is easy to cross by accident
and impossible to find afterwards: a backtest that imports a broker SDK has, somewhere, a
path where live account state can reach a historical decision.

So the import graph is parsed and asserted. Adding a broker means one class inside
``swingbot.execution``; anything else that reaches for one fails the build.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import swingbot

PACKAGE_ROOT = Path(swingbot.__file__).parent

#: Names that mean a module is talking to a venue. Deliberately broad: the point is to
#: catch the accident, and a legitimate adapter lives in the one exempt package.
BROKER_MARKERS = frozenset({
    "alpaca", "alpaca_trade_api", "ib_insync", "ibapi", "ibkr", "interactivebrokers",
    "kiteconnect", "kite", "zerodha", "upstox", "fyers", "angelone", "smartapi",
    "dhanhq", "aliceblue", "shoonya", "finvasia", "iifl", "motilaloswal",
    "robin_stocks", "robinhood", "tda", "tdameritrade", "schwab", "etrade",
    "oanda", "ccxt", "binance", "coinbase", "kraken", "tradier", "webull",
    "polygon_trade", "broker", "brokerage",
})

#: The only package allowed to import a broker SDK.
EXEMPT_PREFIX = "swingbot/execution/"


def _python_files() -> list[Path]:
    return sorted(
        p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def _imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, node.lineno))
    return out


def _relative(path: Path) -> str:
    return f"swingbot/{path.relative_to(PACKAGE_ROOT).as_posix()}"


def test_core_imports_no_broker_sdk():
    """The load-bearing test for the whole broker-independence claim."""
    violations = []
    for path in _python_files():
        rel = _relative(path)
        if rel.startswith(EXEMPT_PREFIX):
            continue
        for module, line in _imports(path):
            root = module.split(".")[0].lower()
            if root in BROKER_MARKERS:
                violations.append(f"{rel}:{line} imports {module!r}")

    assert not violations, (
        "broker SDK imported outside swingbot.execution:\n  "
        + "\n  ".join(violations)
        + "\n\nThe research core must not know a broker exists. Put venue code behind "
        "the ExecutionAdapter protocol in swingbot/execution/."
    )


def test_execution_core_is_itself_broker_free():
    """Even inside the execution package, the shipped adapters touch no venue.

    The protocol and the file and paper adapters are pure. A real broker adapter would be
    a new module here, and this test names the ones that are expected to stay clean.
    """
    for name in ("protocols.py", "orders.py", "csv_out.py"):
        path = PACKAGE_ROOT / "execution" / name
        for module, line in _imports(path):
            root = module.split(".")[0].lower()
            assert root not in BROKER_MARKERS, f"{name}:{line} imports {module!r}"


def test_research_layers_do_not_import_execution():
    """Features, models and validation must not reach into the execution layer.

    A one-directional dependency is what keeps the research path reproducible: if a
    feature could read a live position, a backtest would depend on the state of an
    account that did not exist at the time.
    """
    protected = ("features/", "model/", "validation/", "data/", "news/")
    violations = []
    for path in _python_files():
        rel = _relative(path)
        if not any(rel.startswith(f"swingbot/{p}") for p in protected):
            continue
        for module, line in _imports(path):
            if "execution" in module:
                violations.append(f"{rel}:{line} imports {module!r}")
    assert not violations, "research layers importing execution:\n  " + "\n  ".join(violations)


def test_no_network_calls_in_the_feature_or_model_path():
    """Feature and model code must be pure computation.

    A network call inside feature construction would make a backtest depend on whatever
    the internet returned at the moment it ran, which is the opposite of reproducible.
    """
    network = {"httpx", "requests", "urllib", "urllib3", "socket", "aiohttp"}
    violations = []
    for path in _python_files():
        rel = _relative(path)
        if not any(
            rel.startswith(f"swingbot/{p}") for p in ("features/", "model/", "validation/")
        ):
            continue
        for module, line in _imports(path):
            if module.split(".")[0].lower() in network:
                violations.append(f"{rel}:{line} imports {module!r}")
    assert not violations, "network access in the research path:\n  " + "\n  ".join(violations)


# --------------------------------------------------------------------------------------
# The web boundary
#
# The dashboard is a second front end onto the same pipeline, and it introduces three ways
# to undermine the guarantees above. It could reach for the execution seam, which would put
# a venue back inside the system with a credential attached. The core could start importing
# it, which would make the research path depend on a web framework and turn a reproducible
# computation into a service call. Or its imports could creep to module level, quietly
# making a web stack mandatory for a CLI-only install.
#
# All three are cheap to check and impossible to spot in review six months from now.
# --------------------------------------------------------------------------------------

WEB_PREFIX = "swingbot/web/"

#: `cli.py` is the one legitimate importer: `swingbot serve` has to be able to start it.
WEB_IMPORTERS = ("swingbot/cli.py",)


def test_nothing_outside_the_web_layer_imports_it():
    """The dependency arrow points one way: web knows the pipeline, not the reverse.

    This is what keeps a backtest reproducible from a bare interpreter. The moment a
    feature, a model or the engine imports the dashboard, the research path acquires a
    dependency on a web framework and on whatever request-scoped state that framework
    happens to hold.
    """
    violations = []
    for path in _python_files():
        rel = _relative(path)
        if rel.startswith(WEB_PREFIX) or rel in WEB_IMPORTERS:
            continue
        for module, line in _imports(path):
            if module == "swingbot.web" or module.startswith("swingbot.web."):
                violations.append(f"{rel}:{line} imports {module!r}")
            # Relative forms of the same thing: `from .web import ...` inside the package.
            if module.startswith("web.") or module == "web":
                violations.append(f"{rel}:{line} imports {module!r}")

    assert not violations, (
        "the core imports the web layer:\n  "
        + "\n  ".join(violations)
        + "\n\nOnly swingbot/cli.py may import swingbot.web, so that `swingbot serve` can "
        "start it. Everything else must work with no web dependencies installed."
    )


def test_the_web_layer_never_touches_the_execution_seam():
    """The dashboard is read-only with respect to a broker, and this is the proof.

    The user chose a dashboard that shows what the pipeline decided and records what they
    placed by hand. That choice is a security boundary, not a feature gap: with no
    order-submission path there is no reason for the app to hold a broker credential, and
    therefore no credential for a bug or a hostile page in the same browser to reach.

    What is asserted is the *import*, not a method name. An earlier version of this test
    looked for calls to ``.submit(`` and immediately flagged ``ThreadPoolExecutor.submit``
    in the job runner — the name is too common to carry a guarantee. The import boundary
    is unambiguous: ``swingbot.execution`` is the only package that can construct an
    adapter or hold an :class:`Order`, so a web module that cannot import it cannot place
    a trade however it is written.

    The dashboard does not need it. It reads ``orders.csv`` and ``book.json`` off disk,
    and when it needs a signal produced it calls ``service.run_signal``, which owns the
    adapter. Orders are still written to files — by the service, one layer down, exactly
    as the CLI does it.
    """
    violations = []
    for path in _python_files():
        rel = _relative(path)
        if not rel.startswith(WEB_PREFIX):
            continue
        for module, line in _imports(path):
            # Absolute (`swingbot.execution...`) and relative (`..execution`) forms both.
            if "execution" in module.split("."):
                violations.append(f"{rel}:{line} imports {module!r}")

    assert not violations, (
        "the web layer reached for the execution seam:\n  "
        + "\n  ".join(violations)
        + "\n\nThe dashboard is read-only with respect to a broker: it shows what the "
        "pipeline decided and records what the user placed themselves. It reads order "
        "files off disk and calls service.run_signal to produce new ones; it must not "
        "construct an adapter of its own. Keeping it that way is why this app needs no "
        "credential."
    )


def test_the_web_layer_stays_optional():
    """`import swingbot.web` must work with no web dependencies installed.

    Enforced structurally rather than by an environment: the module-level imports of
    ``swingbot/web/__init__.py`` may not include fastapi or uvicorn, so the check holds
    even in a CI job that happens to have them. The heavy imports live inside
    ``create_app`` and ``serve``.
    """
    init = PACKAGE_ROOT / "web" / "__init__.py"
    tree = ast.parse(init.read_text(), filename=str(init))

    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name.split(".")[0].lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".")[0].lower())
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` imports never execute, so they are not a real dependency.
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import | ast.ImportFrom):
                    top_level.append("__typing_only__")

    web_stack = {"fastapi", "uvicorn", "starlette"}
    offenders = sorted(web_stack.intersection(top_level))
    assert not offenders, (
        f"swingbot/web/__init__.py imports {offenders} at module level, so "
        "`import swingbot.web` now requires the web extra. Move the import inside "
        "create_app() or serve()."
    )


def test_execution_adapter_protocol_is_satisfied():
    """The shipped adapters really implement the protocol they claim to."""
    from swingbot.execution import (
        CSVExecutionAdapter,
        ExecutionAdapter,
        PaperExecutionAdapter,
    )

    for cls in (CSVExecutionAdapter, PaperExecutionAdapter):
        for method in ("current_positions", "account_equity", "submit", "close"):
            assert hasattr(cls, method), f"{cls.__name__} is missing {method}"
        assert isinstance(cls, type)
    assert ExecutionAdapter is not None


def test_orders_carry_no_broker_vocabulary():
    """``Order`` must stay a plain instruction any venue can be told to interpret."""
    from dataclasses import fields

    from swingbot.types import Order

    names = {f.name for f in fields(Order)}
    assert names == {
        "ticker", "side", "quantity", "order_type", "limit_price", "instrument",
        "client_order_id", "tag",
    }, f"Order has drifted toward a venue-specific shape: {sorted(names)}"


def test_secrets_are_never_read_from_yaml():
    """Credentials come from the environment only, so a shared repo cannot leak one."""
    import yaml

    for path in sorted(Path("config").rglob("*.yaml")):
        text = path.read_text()
        loaded = yaml.safe_load(text) or {}

        def walk(node, trail="", *, source=path):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{trail}.{key}", source=source)
            elif isinstance(node, str) and len(node) > 20:
                lowered = trail.lower()
                if any(m in lowered for m in ("token", "secret", "password", "api_key")):
                    # Only the *name* of an environment variable may appear here.
                    assert lowered.endswith("_env"), (
                        f"{source}{trail} looks like a literal credential"
                    )

        walk(loaded)


@pytest.mark.parametrize("module", [
    "swingbot.config", "swingbot.pit", "swingbot.calendars", "swingbot.types",
    "swingbot.data", "swingbot.features", "swingbot.model", "swingbot.validation",
    "swingbot.portfolio", "swingbot.backtest", "swingbot.news", "swingbot.execution",
    "swingbot.report", "swingbot.notify", "swingbot.pipeline", "swingbot.service",
    "swingbot.web", "swingbot.cli",
])
def test_every_package_imports_cleanly(module):
    """No import-time side effects, and no missing optional dependency breaks a package."""
    __import__(module)
