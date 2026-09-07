"""Feature engineering: price, technical, chart structure, regime, news, labels."""

from .base import REGISTRY, BaseTransformer, FeatureRegistry, FeatureTransformer, register
from .cross_sectional import (
    cross_sectional_demean,
    cross_sectional_rank,
    cross_sectional_zscore,
    drop_thin_dates,
    sector_neutralize,
)
from .labels import (
    build_labels,
    combine_weights,
    forward_return_matrix,
    recency_weights,
    uniqueness_weights,
)
from .news_agg import (
    EVENT_GROUPS,
    NEWS_FEATURE_NAMES,
    build_news_features,
    news_coverage_report,
)
from .patterns import PatternFeatures
from .pipeline import (
    PRICE_BLOCKS,
    FeaturePipeline,
    feature_columns,
    news_feature_columns,
    price_feature_columns,
)
from .price import PriceFeatures, ReturnMoments
from .regime import RegimeFeatures
from .technical import TechnicalFeatures

__all__ = [
    "EVENT_GROUPS",
    "NEWS_FEATURE_NAMES",
    "PRICE_BLOCKS",
    "REGISTRY",
    "BaseTransformer",
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureTransformer",
    "PatternFeatures",
    "PriceFeatures",
    "RegimeFeatures",
    "ReturnMoments",
    "TechnicalFeatures",
    "build_labels",
    "build_news_features",
    "combine_weights",
    "cross_sectional_demean",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    "drop_thin_dates",
    "feature_columns",
    "forward_return_matrix",
    "news_coverage_report",
    "news_feature_columns",
    "price_feature_columns",
    "recency_weights",
    "register",
    "sector_neutralize",
    "uniqueness_weights",
]
