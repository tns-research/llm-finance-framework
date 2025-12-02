"""
Configuration Classes for LLM Finance Framework

Type-safe, validated configuration classes that replace the global variable spaghetti.
This provides a clean, testable configuration management system.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


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
                if not model.get('tag'):
                    errors.append(f"Model {i} missing 'tag' field")
                if not model.get('router_model'):
                    errors.append(f"Model {i} missing 'router_model' field")
        return errors


@dataclass
class MemoryFeatures:
    """Memory and journaling feature flags"""
    strategic_journal: bool = True
    feeling_log: bool = True
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
    def from_dict(cls, name: str, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create ExperimentConfig from legacy dictionary format"""
        features = FeatureFlags(
            memory=MemoryFeatures(
                strategic_journal=config_dict.get("ENABLE_STRATEGIC_JOURNAL", False),
                feeling_log=config_dict.get("ENABLE_FEELING_LOG", False),
                full_trading_history=config_dict.get("ENABLE_FULL_TRADING_HISTORY", True),
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
            )
        )

        return cls(
            name=name,
            description=config_dict["description"],
            show_dates=config_dict.get("SHOW_DATE_TO_LLM", False),
            features=features
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
class GlobalConfig:
    """Root configuration object"""
    # Core settings
    use_dummy_model: bool = True
    test_mode: bool = True
    test_limit: int = 15

    # Nested configurations
    data: DataSettings = field(default_factory=DataSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    active_experiment: str = "memory_feeling"
    experiments: Dict[str, ExperimentConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default experiments after creation"""
        self._load_default_experiments()

    def _load_default_experiments(self):
        """Load default experiment configurations"""
        legacy_configs = {
            "baseline": {
                "description": "Minimal context: no dates, no memory, no feeling",
                "SHOW_DATE_TO_LLM": False,
                "ENABLE_STRATEGIC_JOURNAL": False,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "memory_only": {
                "description": "Memory/journal only: no dates, no feeling",
                "SHOW_DATE_TO_LLM": False,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "memory_feeling": {
                "description": "Memory + feeling: no dates",
                "SHOW_DATE_TO_LLM": False,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": True,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "dates_only": {
                "description": "Dates only: no memory, no feeling",
                "SHOW_DATE_TO_LLM": True,
                "ENABLE_STRATEGIC_JOURNAL": False,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "dates_memory": {
                "description": "Dates + memory: no feeling",
                "SHOW_DATE_TO_LLM": True,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": False,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
            "dates_full": {
                "description": "Full context: dates + memory + feeling",
                "SHOW_DATE_TO_LLM": True,
                "ENABLE_STRATEGIC_JOURNAL": True,
                "ENABLE_FEELING_LOG": True,
                "ENABLE_FULL_TRADING_HISTORY": True,
                "ENABLE_TECHNICAL_INDICATORS": True,
            },
        }

        for name, config in legacy_configs.items():
            self.experiments[name] = ExperimentConfig.from_dict(name, config)

    def validate(self) -> List[str]:
        """Validate entire configuration"""
        errors = []
        errors.extend(self.data.validate())
        errors.extend(self.models.validate())

        if self.active_experiment not in self.experiments:
            errors.append(f"Active experiment '{self.active_experiment}' not found in available experiments: {list(self.experiments.keys())}")

        if self.test_limit <= 0:
            errors.append("Test limit must be positive")

        return errors
