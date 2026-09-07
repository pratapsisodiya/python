"""Forecasting models and the two-block blender."""

from .baselines import (
    MomentumOnlyModel,
    RandomSignModel,
    ShuffledLabelModel,
    SubsetModel,
    ZeroModel,
)
from .blend import BlendedModel
from .gbdt import HAS_LIGHTGBM, GBDTModel
from .protocols import ReturnModel
from .registry import ModelCard, SchemaMismatchError, load_model, save_model
from .ridge import RidgeModel

__all__ = [
    "HAS_LIGHTGBM",
    "BlendedModel",
    "GBDTModel",
    "ModelCard",
    "MomentumOnlyModel",
    "RandomSignModel",
    "ReturnModel",
    "RidgeModel",
    "SchemaMismatchError",
    "ShuffledLabelModel",
    "SubsetModel",
    "ZeroModel",
    "load_model",
    "save_model",
]


def build_model(kind: str, cfg=None, *, seed: int = 7):
    """Factory used by the CLI and the ablation runner."""
    kind = kind.lower()
    if kind == "gbdt":
        params = cfg.model.gbdt.model_dump() if cfg is not None else {}
        return GBDTModel(seed=seed, **params)
    if kind == "ridge":
        params = cfg.model.ridge.model_dump() if cfg is not None else {}
        return RidgeModel(**params)
    if kind == "momentum":
        return MomentumOnlyModel()
    if kind == "zero":
        return ZeroModel()
    if kind == "random":
        return RandomSignModel(seed=seed)
    raise ValueError(f"Unknown model kind {kind!r}")
