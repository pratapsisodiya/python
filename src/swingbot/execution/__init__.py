"""Execution: the single broker seam, plus file and paper adapters."""

from .csv_out import CSVExecutionAdapter, PaperExecutionAdapter, build_adapter
from .orders import (
    build_orders,
    execution_sequence,
    orders_to_frame,
    positions_after,
    summarise_orders,
)
from .protocols import ExecutionAdapter

__all__ = [
    "CSVExecutionAdapter",
    "ExecutionAdapter",
    "PaperExecutionAdapter",
    "build_adapter",
    "build_orders",
    "execution_sequence",
    "orders_to_frame",
    "positions_after",
    "summarise_orders",
]
