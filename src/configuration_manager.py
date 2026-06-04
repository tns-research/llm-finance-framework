"""
Configuration Manager for LLM Finance Framework

Centralized configuration management with validation and type safety.
Replaces the global variable spaghetti with a clean, testable API.
"""

import logging
import runpy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config_classes import (
    ExperimentConfig,
    FeatureFlags,
    GlobalConfig,
    MemoryFeatures,
    ReportingFeatures,
    TechnicalFeatures,
    TraderPersonality,
)


class ConfigurationManager:
    """
    Centralized configuration management with validation and type safety.

    This class replaces the global variable spaghetti with a clean, testable API.
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        config_values: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(__name__)
        self._config_file = config_file or self._get_default_config_path()
        self._config_values = dict(config_values) if config_values is not None else None
        self._config = self._load_config()
        self._validate_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        base_dir = Path(__file__).parent.parent
        return str(base_dir / "src" / "config.py")

    def _load_config(self) -> GlobalConfig:
        """Load configuration from a value mapping, file, or create defaults."""
        try:
            config = GlobalConfig()
            self._apply_config_values(config, self._load_config_values())
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config, using defaults: {e}")
            return GlobalConfig()

    def _load_config_values(self) -> Dict[str, Any]:
        """Load the flat configuration values from the configured source."""
        if self._config_values is not None:
            return self._config_values

        config_path = self._config_file
        if config_path != self._get_default_config_path():
            try:
                return runpy.run_path(config_path)
            except Exception as e:
                raise ImportError(
                    f"Could not load custom config from {config_path}: {e}"
                ) from e

        try:
            import src.config as legacy_config

            return dict(vars(legacy_config))
        except ImportError as e:
            raise ImportError("Could not import default config from src.config") from e

    def _apply_config_values(self, config: GlobalConfig, values: Mapping[str, Any]):
        """Apply flat config values to the typed configuration object."""
        config.use_dummy_model = values.get("USE_DUMMY_MODEL", config.use_dummy_model)
        config.test_mode = values.get("TEST_MODE", config.test_mode)
        config.test_limit = values.get("TEST_LIMIT", config.test_limit)
        config.debug_show_full_prompt = values.get(
            "DEBUG_SHOW_FULL_PROMPT", config.debug_show_full_prompt
        )
        config.start_row = values.get("START_ROW", config.start_row)
        config.openrouter_api_base = values.get(
            "OPENROUTER_API_BASE", config.openrouter_api_base
        )
        config.ma20_window = values.get("MA20_WINDOW", config.ma20_window)
        config.ret_5d_window = values.get("RET_5D_WINDOW", config.ret_5d_window)
        config.vol20_window = values.get("VOL20_WINDOW", config.vol20_window)
        config.active_experiment = values.get(
            "ACTIVE_EXPERIMENT", config.active_experiment
        )

        config.data.symbol = values.get("SYMBOL", config.data.symbol)
        config.data.start_date = values.get("DATA_START", config.data.start_date)
        config.data.end_date = values.get("DATA_END", config.data.end_date)

        config.models.models = values.get("LLM_MODELS", config.models.models)
        config.models.use_dummy_model = config.use_dummy_model

        active_exp = config.active_experiment
        if active_exp in config.experiments:
            exp_config = config.experiments[active_exp]
            exp_config.features.technical.indicators = values.get(
                "ENABLE_TECHNICAL_INDICATORS", exp_config.features.technical.indicators
            )
            exp_config.features.memory.strategic_journal = values.get(
                "ENABLE_STRATEGIC_JOURNAL", exp_config.features.memory.strategic_journal
            )
            exp_config.features.memory.feeling_log = values.get(
                "ENABLE_FEELING_LOG", exp_config.features.memory.feeling_log
            )
            exp_config.features.memory.chain_of_thought = values.get(
                "ENABLE_CHAIN_OF_THOUGHT", exp_config.features.memory.chain_of_thought
            )
            exp_config.features.memory.full_trading_history = values.get(
                "ENABLE_FULL_TRADING_HISTORY",
                exp_config.features.memory.full_trading_history,
            )
            exp_config.show_dates = values.get(
                "SHOW_DATE_TO_LLM", exp_config.show_dates
            )

        config.personality.active_personality = values.get(
            "ACTIVE_PERSONALITY", config.personality.active_personality
        )

        legacy_personalities = values.get("TRADER_PERSONALITIES", {})
        for name, personality_dict in legacy_personalities.items():
            personality = TraderPersonality(
                name=personality_dict["name"],
                description=personality_dict["description"],
                risk_tolerance=personality_dict.get("risk_tolerance", "medium"),
                decision_style=personality_dict.get("decision_style", "balanced"),
                bias_description=personality_dict.get("bias_description", ""),
                rule_modifiers=personality_dict.get("rule_modifiers", {}),
            )
            config.personality.personalities[name] = personality

    def _validate_config(self):
        """Validate configuration and log errors"""
        errors = self._config.validate()
        if errors:
            error_msg = f"Invalid configuration: {'; '.join(errors)}"
            for error in errors:
                self.logger.error(f"Configuration error: {error}")
            raise ValueError(error_msg)

    def get_current_experiment(self) -> ExperimentConfig:
        """Get the currently active experiment configuration"""
        if self._config.active_experiment not in self._config.experiments:
            available = list(self._config.experiments.keys())
            raise ValueError(
                f"Active experiment '{self._config.active_experiment}' not found in available experiments: {available}"
            )
        return self._config.experiments[self._config.active_experiment]

    def get_feature_flags(self) -> Dict[str, bool]:
        """Get flattened feature flags for backward compatibility"""
        experiment = self.get_current_experiment()
        features = experiment.features

        return {
            # Memory features
            "ENABLE_STRATEGIC_JOURNAL": features.memory.strategic_journal,
            "ENABLE_FEELING_LOG": features.memory.feeling_log,
            "ENABLE_CHAIN_OF_THOUGHT": features.memory.chain_of_thought,
            "ENABLE_FULL_TRADING_HISTORY": features.memory.full_trading_history,
            # Technical features
            "ENABLE_TECHNICAL_INDICATORS": features.technical.indicators,
            # Reporting features
            "ENABLE_COMPREHENSIVE_REPORTS": features.reporting.comprehensive_reports,
            "ENABLE_PLOTS": features.reporting.plots,
            "ENABLE_STATISTICAL_VALIDATION": features.reporting.statistical_validation,
            # Other settings
            "SHOW_DATE_TO_LLM": experiment.show_dates,
        }

    def get_data_settings(self) -> Dict[str, Any]:
        """Get data-related settings"""
        return {
            "SYMBOL": self._config.data.symbol,
            "DATA_START": self._config.data.start_date,
            "DATA_END": self._config.data.end_date,
        }

    def get_symbol_info(self) -> tuple[str, str]:
        """
        Get current symbol code and user-friendly name.

        Returns:
            tuple: (symbol_code, user_friendly_name)
        """
        try:
            from .config import get_current_symbol_info

            return get_current_symbol_info()
        except ImportError as e:
            self.logger.warning(f"Could not import get_current_symbol_info: {e}")
            return "SPY", "SPY ETF (S&P 500 tracker)"

    def get_model_settings(self) -> Dict[str, Any]:
        """Get model-related settings"""
        return {
            "USE_DUMMY_MODEL": self._config.use_dummy_model,
            "TEST_MODE": self._config.test_mode,
            "TEST_LIMIT": self._config.test_limit,
            "LLM_MODELS": self._config.models.models,
        }

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with validation"""
        # Apply updates to config object
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                # Try nested update
                self._update_nested_config(key, value)

        # Validate updated config
        self._validate_config()
        self.logger.info("Configuration updated successfully")

    def _update_nested_config(self, key: str, value: Any):
        """Update nested configuration attributes"""
        parts = key.split(".")
        obj = self._config

        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise ValueError(f"Invalid config path: {key}")

        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
        else:
            raise ValueError(f"Invalid config attribute: {key}")

    def get_experiment_suffix(self) -> str:
        """Get experiment suffix for file naming (backward compatibility)"""
        experiment = self.get_current_experiment()

        if experiment.name in self._config.experiments:
            return f"_{experiment.name}"
        else:
            # Build suffix from flags
            flags = self.get_feature_flags()
            parts = []
            if flags["SHOW_DATE_TO_LLM"]:
                parts.append("dates")
            if flags["ENABLE_STRATEGIC_JOURNAL"]:
                parts.append("mem")
            if flags["ENABLE_FEELING_LOG"]:
                parts.append("feel")
            if not parts:
                parts.append("minimal")
            return "_" + "_".join(parts)

    def get_current_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        experiment = self.get_current_experiment()
        flags = self.get_feature_flags()

        return {
            "experiment": experiment.name,
            "description": experiment.description,
            "show_dates": flags["SHOW_DATE_TO_LLM"],
            "strategic_journal": flags["ENABLE_STRATEGIC_JOURNAL"],
            "feeling_log": flags["ENABLE_FEELING_LOG"],
            "chain_of_thought": flags["ENABLE_CHAIN_OF_THOUGHT"],
        }

    def list_experiments(self) -> Dict[str, str]:
        """List available experiments with descriptions"""
        return {name: exp.description for name, exp in self._config.experiments.items()}

    def create_experiment(
        self, name: str, description: str, **kwargs
    ) -> ExperimentConfig:
        """Create a new experiment configuration"""
        experiment = ExperimentConfig(
            name=name,
            description=description,
            show_dates=kwargs.get("show_dates", False),
            features=FeatureFlags(
                memory=MemoryFeatures(
                    strategic_journal=kwargs.get("strategic_journal", False),
                    feeling_log=kwargs.get("feeling_log", False),
                    full_trading_history=kwargs.get("full_trading_history", True),
                ),
                technical=TechnicalFeatures(
                    indicators=kwargs.get("technical_indicators", True),
                ),
                reporting=ReportingFeatures(),  # Use defaults
            ),
        )

        self._config.experiments[name] = experiment
        self.logger.info(f"Created new experiment: {name}")
        return experiment

    def set_active_experiment(self, experiment_name: str):
        """Set the active experiment"""
        if experiment_name not in self._config.experiments:
            available = list(self._config.experiments.keys())
            raise ValueError(
                f"Experiment '{experiment_name}' not found. Available: {available}"
            )

        self._config.active_experiment = experiment_name
        self.logger.info(f"Active experiment set to: {experiment_name}")

    def get_active_personality(self) -> TraderPersonality:
        """
        Get the currently active trader personality.

        Returns:
            TraderPersonality: The active personality configuration

        Raises:
            ValueError: If the active personality is not found
        """
        personality_name = self._config.personality.active_personality
        if personality_name not in self._config.personality.personalities:
            available = list(self._config.personality.personalities.keys())
            raise ValueError(
                f"Active personality '{personality_name}' not found. Available: {available}"
            )
        return self._config.personality.personalities[personality_name]

    def set_active_personality(self, personality_name: str):
        """
        Set the active trader personality.

        Args:
            personality_name: Name of the personality to activate

        Raises:
            ValueError: If personality doesn't exist
        """
        if personality_name not in self._config.personality.personalities:
            available = list(self._config.personality.personalities.keys())
            raise ValueError(
                f"Personality '{personality_name}' not found. Available: {available}"
            )

        self._config.personality.active_personality = personality_name
        self.logger.info(f"Active personality set to: {personality_name}")
