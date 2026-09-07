"""Configuration: layered YAML, environment overrides, and a reproducibility hash.

Layering order, later wins::

    config/base.yaml
    config/markets/<market>.yaml
    config/profiles/<profile>.yaml
    environment          SWINGBOT__PORTFOLIO__TARGET_VOL_ANNUAL=0.10
    --set                portfolio.target_vol_annual=0.10

Secrets are read from the environment only and never appear in a config file or in the
config hash. Every run snapshots its fully resolved config, so reproducing a result is a
file diff rather than an act of memory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import ShortInstrument

ENV_PREFIX = "SWINGBOT__"
DEFAULT_CONFIG_DIR = Path("config")


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class RunConfig(_Base):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    seed: int = 7
    log_level: str = "INFO"


class DataConfig(_Base):
    providers: list[str] = Field(default_factory=lambda: ["csv", "synthetic"])
    cache_enabled: bool = True
    start: date | None = None
    end: date | None = None
    min_history_sessions: int = 260
    min_price: float = 5.0
    min_adv_notional: float = 1.0e7


class UniverseConfig(_Base):
    file: Path | None = None
    max_names: int = 500
    assume_survivorship_biased: bool = True


class CalendarConfig(_Base):
    decision_weekday: int = 4
    hold_sessions: int = 5

    @field_validator("decision_weekday")
    @classmethod
    def _weekday_range(cls, v: int) -> int:
        if not 0 <= v <= 4:
            raise ValueError("decision_weekday must be 0 (Mon) through 4 (Fri)")
        return v

    @field_validator("hold_sessions")
    @classmethod
    def _hold_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("hold_sessions must be at least 1")
        return v


class NewsFeatureConfig(_Base):
    enabled: bool = True
    lookback_days: int = 10
    half_life_hours: float = 36.0
    peer_weight: float = 0.3
    speculative_discount: float = 0.5


class FeatureConfig(_Base):
    winsorize_quantile: float = 0.02
    min_names_per_week: int = 20
    sector_neutralize: bool = False
    news: NewsFeatureConfig = Field(default_factory=NewsFeatureConfig)


class LabelConfig(_Base):
    kind: str = "excess_log_return"
    vol_scale: bool = False


class CVConfig(_Base):
    scheme: str = "purged_walk_forward"
    train_weeks: int = 156
    test_weeks: int = 26
    embargo_weeks: int = 2
    expanding: bool = True
    min_train_weeks: int = 104


class BlendConfig(_Base):
    price_weight: float = 0.90
    news_weight: float = 0.10

    @property
    def normalised(self) -> tuple[float, float]:
        total = self.price_weight + self.news_weight
        if total <= 0:
            return 1.0, 0.0
        return self.price_weight / total, self.news_weight / total


class GBDTConfig(_Base):
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 80
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 5.0


class RidgeConfig(_Base):
    alpha: float = 10.0


class ModelConfig(_Base):
    price_model: str = "gbdt"
    news_model: str = "ridge"
    blend: BlendConfig = Field(default_factory=BlendConfig)
    gbdt: GBDTConfig = Field(default_factory=GBDTConfig)
    ridge: RidgeConfig = Field(default_factory=RidgeConfig)
    recency_half_life_weeks: float = 104.0


class PortfolioConfig(_Base):
    n_long: int = 8
    n_short: int = 7
    gross_leverage: float = 1.0
    max_net_exposure: float = 0.30
    target_vol_annual: float = 0.12
    sizing: str = "inverse_vol"
    kelly_fraction: float = 0.25
    max_weight: float = 0.12
    max_sector_weight: float = 0.35
    no_trade_band: float = 0.005
    min_position_notional: float = 0.0
    max_participation_adv: float = 0.05


class RiskConfig(_Base):
    enabled: bool = True
    drawdown_scale_at: float = 0.08
    drawdown_flat_at: float = 0.15
    cooldown_weeks: int = 4


class BacktestConfig(_Base):
    initial_equity: float = 1_000_000.0
    cost_scale: float = 1.0
    start: date | None = None
    end: date | None = None


class ExecutionConfig(_Base):
    adapter: str = "csv"
    output_dir: Path = Path("runs")


class TelegramConfig(_Base):
    enabled: bool = False
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"


class EmailConfig(_Base):
    enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    use_tls: bool = True
    sender: str | None = None
    recipients: list[str] = Field(default_factory=list)
    username_env: str = "SMTP_USERNAME"
    password_env: str = "SMTP_PASSWORD"


class NotifyConfig(_Base):
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)


class OpenAICompatConfig(_Base):
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"


class AnthropicConfig(_Base):
    model: str = "claude-opus-5"
    api_key_env: str = "ANTHROPIC_API_KEY"


class FinbertConfig(_Base):
    model: str = "ProsusAI/finbert"


class NLPConfig(_Base):
    """The news-analysis layer.

    Deliberately provider-agnostic. ``lexicon`` is the default because it needs no API
    key and runs offline, so the system is useful the moment it is cloned.
    """

    backend: str = "lexicon"
    batch_size: int = 20
    max_articles_per_run: int = 2000
    max_spend_usd: float = 5.0
    timeout_seconds: float = 60.0
    max_retries: int = 3
    prompt_version: str = "v1"
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    finbert: FinbertConfig = Field(default_factory=FinbertConfig)


class RSSConfig(_Base):
    max_items_per_symbol: int = 50
    request_delay_seconds: float = 0.5
    hl: str = "en-US"
    gl: str = "US"
    ceid: str = "US:en"


class NewsConfig(_Base):
    provider: str = "jsonl"
    jsonl_path: Path | None = None
    rss: RSSConfig = Field(default_factory=RSSConfig)


class CostConfig(_Base):
    """Cost model parameters, all in basis points of notional unless noted."""

    commission_bps: float = 0.5
    min_commission: float = 0.0
    regulatory_sell_bps: float = 0.0
    min_half_spread_bps: float = 2.0
    spread_from_range_factor: float = 0.25
    impact_k: float = 0.5
    borrow_bps_annual: float = 50.0
    # India cash-segment statutory charges.
    stt_buy_bps: float = 0.0
    stt_sell_bps: float = 0.0
    stamp_duty_bps: float = 0.0
    exchange_bps: float = 0.0
    gst_rate: float = 0.0
    # Futures leg, used when short_instrument is FUTURES.
    futures_commission_bps: float = 0.0
    futures_stt_sell_bps: float = 0.0
    futures_exchange_bps: float = 0.0
    futures_stamp_duty_bps: float = 0.0
    futures_roll_bps: float = 0.0


class MarketProfile(_Base):
    name: str = "us"
    display_name: str = "US Equities"
    currency: str = "USD"
    timezone: str = "America/New_York"
    close_local_time: str = "16:00"
    symbol_suffix: str = ""
    benchmark: str = "SPY"
    allow_short: bool = True
    short_instrument: ShortInstrument = ShortInstrument.CASH_EQUITY

    @field_validator("short_instrument", mode="before")
    @classmethod
    def _none_means_no_shorting(cls, v):
        """Treat a null short instrument as "no shorting" rather than an error.

        ``--set market_profile.short_instrument=none`` is the natural way to ask for a
        long-only book, but the CLI coerces the string "none" to Python ``None`` along
        with "null" and the empty string, which would otherwise fail enum validation with
        a message that gives no hint about what to type instead. A null short instrument
        genuinely means there is no instrument to short with, so mapping it to NONE is
        the honest reading, not a workaround.
        """
        return ShortInstrument.NONE if v is None else v


class Config(_Base):
    market: str = "us"
    run: RunConfig = Field(default_factory=RunConfig)
    market_profile: MarketProfile = Field(default_factory=MarketProfile)
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    costs: CostConfig = Field(default_factory=CostConfig)

    # ---------------------------------------------------------------- derived helpers

    @property
    def market_dir(self) -> Path:
        """Per-market data root, so India and US never share a cache."""
        return self.run.data_dir / self.market_profile.name

    @property
    def shorts_allowed(self) -> bool:
        return (
            self.market_profile.allow_short
            and self.market_profile.short_instrument is not ShortInstrument.NONE
            and self.portfolio.n_short > 0
        )

    def config_hash(self) -> str:
        """Stable hash of the resolved config, used to key runs and the trial ledger."""
        payload = self.model_dump(mode="json")
        # Paths and log level do not change results.
        payload.get("run", {}).pop("log_level", None)
        payload.get("run", {}).pop("data_dir", None)
        payload.get("run", {}).pop("runs_dir", None)
        payload.get("execution", {}).pop("output_dir", None)
        payload.pop("notify", None)
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        loaded = yaml.safe_load(fh)
    return loaded or {}


def _coerce_scalar(text: str) -> Any:
    """Turn a CLI or environment string into the obvious Python value."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    if lowered.startswith("[") and lowered.endswith("]"):
        inner = text.strip()[1:-1]
        return [_coerce_scalar(part) for part in inner.split(",") if part.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        dotted = key[len(ENV_PREFIX) :].lower().replace("__", ".")
        _set_dotted(out, dotted, _coerce_scalar(raw))
    return out


def load_config(
    market: str = "us",
    profile: str | None = None,
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    overrides: dict[str, Any] | None = None,
    set_values: list[str] | None = None,
) -> Config:
    """Resolve the layered configuration for a market.

    ``set_values`` are ``section.key=value`` strings from ``--set`` on the CLI.
    """
    config_dir = Path(config_dir)
    market_key = _resolve_market_key(market)

    merged = _read_yaml(config_dir / "base.yaml")
    merged = _deep_merge(merged, _read_yaml(config_dir / "markets" / f"{market_key}.yaml"))
    if profile:
        profile_path = config_dir / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"No such profile: {profile_path}")
        merged = _deep_merge(merged, _read_yaml(profile_path))

    merged = _deep_merge(merged, _env_overrides())

    if overrides:
        merged = _deep_merge(merged, overrides)

    for item in set_values or []:
        if "=" not in item:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        _set_dotted(merged, dotted.strip(), _coerce_scalar(raw))

    merged["market"] = market_key
    return Config.model_validate(merged)


#: Accepted spellings for each market, so ``--market in`` and ``--market india`` agree.
_MARKET_ALIASES = {
    "us": "us",
    "usa": "us",
    "united_states": "us",
    "nyse": "us",
    "nasdaq": "us",
    "in": "india",
    "ind": "india",
    "india": "india",
    "nse": "india",
}


def _resolve_market_key(market: str) -> str:
    key = market.strip().lower()
    if key not in _MARKET_ALIASES:
        known = ", ".join(sorted(set(_MARKET_ALIASES.values())))
        raise ValueError(f"Unknown market {market!r}. Known markets: {known}")
    return _MARKET_ALIASES[key]


def available_markets() -> list[str]:
    return sorted(set(_MARKET_ALIASES.values()))


def get_secret(env_var: str) -> str | None:
    """Read a secret from the environment. Secrets never come from YAML."""
    value = os.environ.get(env_var)
    return value.strip() if value and value.strip() else None
