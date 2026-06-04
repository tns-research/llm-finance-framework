"""
Configuration Classes for LLM Finance Framework

Type-safe, validated configuration classes that replace the global variable spaghetti.
This provides a clean, testable configuration management system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class ExperimentType(Enum):
    """Enumeration of available experiment types"""

    BASELINE = "baseline"
    MEMORY_ONLY = "memory_only"
    MEMORY_FEELING = "memory_feeling"
    DATES_ONLY = "dates_only"
    DATES_MEMORY = "dates_memory"
    DATES_FULL = "dates_full"


@dataclass
class DataSettings:
    """Data source and processing configuration"""

    symbol: str = "^GSPC"
    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"
    raw_data_path: Optional[str] = None
    processed_data_path: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate data settings and return error messages"""
        errors = []
        if not self.symbol:
            errors.append("Symbol cannot be empty")

        # Validate date formats
        try:
            start_dt = pd.to_datetime(self.start_date)
        except (ValueError, TypeError):
            errors.append(f"Invalid start_date format: {self.start_date}")
            start_dt = None

        try:
            end_dt = pd.to_datetime(self.end_date)
        except (ValueError, TypeError):
            errors.append(f"Invalid end_date format: {self.end_date}")
            end_dt = None

        # Only check date order if both dates are valid
        if start_dt is not None and end_dt is not None:
            if start_dt >= end_dt:
                errors.append("Start date must be before end date")

        return errors


@dataclass
class ModelSettings:
    """LLM model configuration"""

    use_dummy_model: bool = True
    models: List[Dict[str, str]] = field(default_factory=list)
    api_key: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate model settings"""
        errors = []
        if not self.use_dummy_model:
            if not self.models:
                errors.append("Real models require at least one model configuration")
            for i, model in enumerate(self.models):
                if not model.get("tag"):
                    errors.append(f"Model {i} missing 'tag' field")
                if not model.get("router_model"):
                    errors.append(f"Model {i} missing 'router_model' field")
        return errors


@dataclass
class MemoryFeatures:
    """Memory and journaling feature flags"""

    strategic_journal: bool = True
    feeling_log: bool = True
    chain_of_thought: bool = False
    full_trading_history: bool = True


@dataclass
class TechnicalFeatures:
    """Technical analysis feature flags"""

    indicators: bool = True
    historical_series: bool = True
    aggregated_stats: bool = True


@dataclass
class ReportingFeatures:
    """Reporting and visualization features"""

    comprehensive_reports: bool = True
    plots: bool = True
    statistical_validation: bool = True


@dataclass
class FeatureFlags:
    """Organized feature flags by domain"""

    memory: MemoryFeatures = field(default_factory=MemoryFeatures)
    technical: TechnicalFeatures = field(default_factory=TechnicalFeatures)
    reporting: ReportingFeatures = field(default_factory=ReportingFeatures)


@dataclass
class ExperimentConfig:
    """Individual experiment configuration"""

    name: str
    description: str
    show_dates: bool = False
    features: FeatureFlags = field(default_factory=FeatureFlags)

    @classmethod
    def from_dict(cls, name: str, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from legacy dictionary format"""
        features = FeatureFlags(
            memory=MemoryFeatures(
                strategic_journal=config_dict.get("ENABLE_STRATEGIC_JOURNAL", False),
                feeling_log=config_dict.get("ENABLE_FEELING_LOG", False),
                chain_of_thought=config_dict.get("ENABLE_CHAIN_OF_THOUGHT", False),
                full_trading_history=config_dict.get(
                    "ENABLE_FULL_TRADING_HISTORY", True
                ),
            ),
            technical=TechnicalFeatures(
                indicators=config_dict.get("ENABLE_TECHNICAL_INDICATORS", True),
                historical_series=config_dict.get("ENABLE_TECHNICAL_INDICATORS", True),
                aggregated_stats=config_dict.get("ENABLE_TECHNICAL_INDICATORS", True),
            ),
            reporting=ReportingFeatures(
                comprehensive_reports=True,  # Always enabled for now
                plots=True,
                statistical_validation=True,
            ),
        )

        return cls(
            name=name,
            description=config_dict["description"],
            show_dates=config_dict.get("SHOW_DATE_TO_LLM", False),
            features=features,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format for compatibility"""
        return {
            "description": self.description,
            "SHOW_DATE_TO_LLM": self.show_dates,
            "ENABLE_STRATEGIC_JOURNAL": self.features.memory.strategic_journal,
            "ENABLE_FEELING_LOG": self.features.memory.feeling_log,
            "ENABLE_FULL_TRADING_HISTORY": self.features.memory.full_trading_history,
            "ENABLE_TECHNICAL_INDICATORS": self.features.technical.indicators,
        }


@dataclass
class TraderPersonality:
    """
    Configuration for a trader personality profile.

    Defines behavioral characteristics that influence how the LLM approaches
    trading decisions, allowing systematic testing of different trading styles.

    Attributes:
        name: Human-readable name (e.g., "Cautious Conservative")
        description: Core personality description used in system prompts
        risk_tolerance: Risk preference level ("low", "medium", "high")
        decision_style: Decision-making approach ("defensive", "offensive", "systematic")
        bias_description: Explanation of behavioral biases and tendencies
        rule_modifiers: Optional personality-specific rule adjustments
    """

    name: str
    description: str
    risk_tolerance: str = "medium"
    decision_style: str = "balanced"
    bias_description: str = ""
    rule_modifiers: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate personality configuration"""
        errors = []
        if not self.name:
            errors.append("Personality name cannot be empty")
        if not self.description:
            errors.append("Personality description cannot be empty")
        if self.risk_tolerance not in ["low", "medium", "high"]:
            errors.append(f"Invalid risk_tolerance: {self.risk_tolerance}")
        if self.decision_style not in [
            "defensive",
            "offensive",
            "systematic",
            "balanced",
            "reactive",
            "contrarian",
        ]:
            errors.append(f"Invalid decision_style: {self.decision_style}")
        return errors


@dataclass
class PersonalitySettings:
    """
    Container for personality-related configuration settings.

    Manages the active personality and available personality profiles.
    """

    active_personality: str = "cautious"
    personalities: Dict[str, TraderPersonality] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate personality settings"""
        errors = []
        if not self.active_personality:
            errors.append("Active personality cannot be empty")
        if self.active_personality not in self.personalities:
            available = list(self.personalities.keys())
            errors.append(
                f"Active personality '{self.active_personality}' not found in available personalities: {available}"
            )

        # Validate each personality
        for name, personality in self.personalities.items():
            personality_errors = personality.validate()
            for error in personality_errors:
                errors.append(f"Personality '{name}': {error}")

        return errors


@dataclass
class GlobalConfig:
    """Root configuration object"""

    # Core settings
    use_dummy_model: bool = True
    test_mode: bool = True
    test_limit: int = 15

    # Debug settings
    debug_show_full_prompt: bool = True
    start_row: Optional[int] = None
    openrouter_api_base: str = "https://openrouter.ai/api/v1/chat/completions"

    # Technical indicator windows
    ma20_window: int = 20
    ret_5d_window: int = 5
    vol20_window: int = 20

    # Nested configurations
    data: DataSettings = field(default_factory=DataSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    active_experiment: str = "memory_feeling"
    experiments: Dict[str, ExperimentConfig] = field(default_factory=dict)
    personality: PersonalitySettings = field(default_factory=PersonalitySettings)

    def __post_init__(self):
        """Initialize default experiments and personalities after creation"""
        self._load_default_experiments()
        self._load_default_personalities()

    def _load_default_experiments(self):
        """Load default experiment configurations"""
        legacy_configs = {
            "baseline": {
                "description": "Minimal context: no dates, no memory, no feeling",
                "SHOW_DATE_TO_LLM": False,
                "ENABLE_STRATEGIC_JOURNAL": False,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_CHAIN_OF_THOUGHT": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "memory_only": {
                "description": "Memory/journal only: no dates, no feeling",
                "SHOW_DATE_TO_LLM": False,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_CHAIN_OF_THOUGHT": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "memory_feeling": {
                "description": "Memory + feeling: no dates",
                "SHOW_DATE_TO_LLM": False,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": True,
                "ENABLE_CHAIN_OF_THOUGHT": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "dates_only": {
                "description": "Dates only: no memory, no feeling",
                "SHOW_DATE_TO_LLM": True,
                "ENABLE_STRATEGIC_JOURNAL": False,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_CHAIN_OF_THOUGHT": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "dates_memory": {
                "description": "Dates + memory: no feeling",
                "SHOW_DATE_TO_LLM": True,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_CHAIN_OF_THOUGHT": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "dates_full": {
                "description": "Full context: dates + memory + feeling",
                "SHOW_DATE_TO_LLM": True,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": True,
                "ENABLE_CHAIN_OF_THOUGHT": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
        }

        for name, config in legacy_configs.items():
            self.experiments[name] = ExperimentConfig.from_dict(name, config)

    def _load_default_personalities(self):
        """
        Load default trader personality configurations.

        These personalities provide systematic testing of different behavioral
        profiles that influence LLM trading decisions.
        """
        self.personality.personalities = {
            "cautious": TraderPersonality(
                name="Cautious Conservative",
                description="cautious but rational financial asset trader",
                risk_tolerance="low",
                decision_style="defensive",
                bias_description="Prioritizes capital preservation, requires strong conviction for directional trades. Prefers HOLD when signals are mixed and takes quick action to exit losing positions.",
            ),
            "aggressive": TraderPersonality(
                name="Aggressive Growth",
                description="bold and opportunistic financial asset trader",
                risk_tolerance="high",
                decision_style="offensive",
                bias_description="Seeks alpha through active positioning, tolerates higher volatility for potential returns. More willing to take directional risk based on market momentum.",
            ),
            "balanced": TraderPersonality(
                name="Balanced Professional",
                description="disciplined and analytical financial asset portfolio manager",
                risk_tolerance="medium",
                decision_style="systematic",
                bias_description="Balances risk and reward systematically, follows structured decision criteria. Makes decisions based on comprehensive analysis rather than instinct.",
            ),
            "momentum": TraderPersonality(
                name="Momentum Trader",
                description="trend-following momentum trader focused on financial assets",
                risk_tolerance="high",
                decision_style="reactive",
                bias_description="Capitalizes on market trends, quick to cut losses and ride winners. Emphasizes timing and market direction over fundamental valuation.",
            ),
            "contrarian": TraderPersonality(
                name="Contrarian Value",
                description="contrarian value investor trading financial assets",
                risk_tolerance="medium",
                decision_style="contrarian",
                bias_description="Fades market sentiment, buys fear and sells greed when fundamentals suggest. Goes against prevailing market psychology when indicators show extremes.",
            ),
        }

        # Ensure active personality exists
        if self.personality.active_personality not in self.personality.personalities:
            # Fallback to default - but this should never happen with our defaults
            self.personality.active_personality = "cautious"

    def validate(self) -> List[str]:
        """Validate entire configuration"""
        errors = []
        errors.extend(self.data.validate())
        errors.extend(self.models.validate())
        errors.extend(self.personality.validate())

        if self.active_experiment not in self.experiments:
            errors.append(
                f"Active experiment '{self.active_experiment}' not found in available experiments: {list(self.experiments.keys())}"
            )

        if self.test_limit <= 0:
            errors.append("Test limit must be positive")

        return errors
