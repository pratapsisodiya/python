"""swingbot — a broker-independent weekly swing-trading research and signal system.

The package is organised so that nothing above ``swingbot.execution`` knows a broker
exists. The core computes target positions; adapters translate them. See
``tests/test_architecture.py``, which fails the build if that boundary is crossed.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
