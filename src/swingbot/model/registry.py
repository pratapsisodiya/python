"""Model persistence.

A saved model is useless without the exact feature schema it was trained against. If the
feature set changes and a stale model is loaded, the columns silently misalign and the
predictions become noise that looks like signal. So the schema, the training window and
the config hash are saved alongside the estimator, and loading validates them.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ..io.store import read_json, write_json


class SchemaMismatchError(RuntimeError):
    """Raised when a saved model's features do not match the current panel."""


@dataclass(slots=True)
class ModelCard:
    """Everything needed to know whether a saved model is safe to use."""

    name: str
    market: str
    config_hash: str
    feature_names: list[str]
    price_columns: list[str] = field(default_factory=list)
    news_columns: list[str] = field(default_factory=list)
    train_start: str = ""
    train_end: str = ""
    n_train_rows: int = 0
    trained_at: str = ""
    backend: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def save_model(
    model,
    directory: Path | str,
    *,
    market: str,
    config_hash: str,
    train_start: date | None = None,
    train_end: date | None = None,
    n_train_rows: int = 0,
    price_columns: list[str] | None = None,
    news_columns: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    card = ModelCard(
        name=getattr(model, "name", type(model).__name__),
        market=market,
        config_hash=config_hash,
        feature_names=list(getattr(model, "feature_names", [])),
        price_columns=list(price_columns or []),
        news_columns=list(news_columns or []),
        train_start=str(train_start or ""),
        train_end=str(train_end or ""),
        n_train_rows=int(n_train_rows),
        trained_at=datetime.now(UTC).isoformat(),
        backend=str(getattr(model, "backend", "")),
        extra=extra or {},
    )

    (directory / "model.pkl").write_bytes(pickle.dumps(model))
    write_json(directory / "model_card.json", card.to_dict())
    return directory


def load_model(directory: Path | str, *, expected_features: list[str] | None = None):
    """Load a model, refusing to return one whose schema no longer matches."""
    directory = Path(directory)
    payload = directory / "model.pkl"
    if not payload.exists():
        raise FileNotFoundError(f"No saved model at {directory}")

    model = pickle.loads(payload.read_bytes())
    card_data = read_json(directory / "model_card.json", {})
    card = ModelCard(**card_data) if card_data else None

    if expected_features is not None and card is not None and card.feature_names:
        missing = set(card.feature_names) - set(expected_features)
        if missing:
            raise SchemaMismatchError(
                f"Saved model expects {len(missing)} feature(s) the current panel does "
                f"not provide, for example {sorted(missing)[:5]}. Retrain before signalling."
            )
    return model, card
