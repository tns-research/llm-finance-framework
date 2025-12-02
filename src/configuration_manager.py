"""
Configuration Manager for LLM Finance Framework

Centralized configuration management with validation and type safety.
Replaces the global variable spaghetti with a clean, testable API.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .config_classes import (
    GlobalConfig, ExperimentConfig, FeatureFlags,
    MemoryFeatures, TechnicalFeatures, ReportingFeatures
)


class ConfigurationManager:
    """
    Centralized configuration management with validation and type safety.

    This class replaces the global variable spaghetti with a clean, testable API.
    """

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(__name__)
        self._config_file = config_file or self._get_default_config_path()
        self._config = self._load_config()
        self._validate_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        base_dir = Path(__file__).parent.parent
        return str(base_dir / "src" / "config.py")

    def _load_config(self) -> GlobalConfig:
        """Load configuration from file or create defaults"""
        try:
            # For now, create default config
            # Later: load from YAML/TOML file
            config = GlobalConfig()
            self._apply_legacy_overrides(config)
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config, using defaults: {e}")
            return GlobalConfig()

    def _apply_legacy_overrides(self, config: GlobalConfig):
        """Apply legacy config.py overrides for backward compatibility"""
        # Import legacy config to get current values
        try:
            # Dynamic import to avoid circular imports
            import sys
            import importlib.util

            config_path = self._config_file
            spec = importlib.util.spec_from_file_location("legacy_config", config_path)
            legacy_config = importlib.util.module_from_spec(spec)
            sys.modules["legacy_config"] = legacy_config
            spec.loader.exec_module(legacy_config)

            # Apply basic settings
            config.use_dummy_model = getattr(legacy_config, 'USE_DUMMY_MODEL', True)
            config.test_mode = getattr(legacy_config, 'TEST_MODE', True)
            config.test_limit = getattr(legacy_config, 'TEST_LIMIT', 15)
            config.active_experiment = getattr(legacy_config, 'ACTIVE_EXPERIMENT', 'memory_feeling')

            # Apply data settings
            config.data.symbol = getattr(legacy_config, 'SYMBOL', '^GSPC')
            config.data.start_date = getattr(legacy_config, 'DATA_START', '2015-01-01')
            config.data.end_date = getattr(legacy_config, 'DATA_END', '2023-12-31')

            # Apply model settings
            config.models.models = getattr(legacy_config, 'LLM_MODELS', [])
            config.models.use_dummy_model = config.use_dummy_model

            # CRITICAL FIX: Apply feature flags that are set dynamically in legacy config
            # These are set based on the active experiment at the bottom of config.py
            active_exp = config.active_experiment
            if active_exp in config.experiments:
                exp_config = config.experiments[active_exp]

                # Read the dynamically set values from legacy config
                exp_config.features.technical.indicators = getattr(legacy_config, 'ENABLE_TECHNICAL_INDICATORS', True)
                exp_config.features.memory.strategic_journal = getattr(legacy_config, 'ENABLE_STRATEGIC_JOURNAL', True)
                exp_config.features.memory.feeling_log = getattr(legacy_config, 'ENABLE_FEELING_LOG', True)
                exp_config.show_dates = getattr(legacy_config, 'SHOW_DATE_TO_LLM', False)

        except Exception as e:
            self.logger.warning(f"Could not import legacy config from {self._config_file}, using defaults: {e}")

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
            raise ValueError(f"Active experiment '{self._config.active_experiment}' not found in available experiments: {available}")
        return self._config.experiments[self._config.active_experiment]

    def get_feature_flags(self) -> Dict[str, bool]:
        """Get flattened feature flags for backward compatibility"""
        experiment = self.get_current_experiment()
        features = experiment.features

        return {
            # Memory features
            'ENABLE_STRATEGIC_JOURNAL': features.memory.strategic_journal,
            'ENABLE_FEELING_LOG': features.memory.feeling_log,
            'ENABLE_FULL_TRADING_HISTORY': features.memory.full_trading_history,

            # Technical features
            'ENABLE_TECHNICAL_INDICATORS': features.technical.indicators,

            # Reporting features
            'ENABLE_COMPREHENSIVE_REPORTS': features.reporting.comprehensive_reports,
            'ENABLE_PLOTS': features.reporting.plots,
            'ENABLE_STATISTICAL_VALIDATION': features.reporting.statistical_validation,

            # Other settings
            'SHOW_DATE_TO_LLM': experiment.show_dates,
        }

    def get_data_settings(self) -> Dict[str, Any]:
        """Get data-related settings"""
        return {
            'SYMBOL': self._config.data.symbol,
            'DATA_START': self._config.data.start_date,
            'DATA_END': self._config.data.end_date,
        }

    def get_model_settings(self) -> Dict[str, Any]:
        """Get model-related settings"""
        return {
            'USE_DUMMY_MODEL': self._config.use_dummy_model,
            'TEST_MODE': self._config.test_mode,
            'TEST_LIMIT': self._config.test_limit,
            'LLM_MODELS': self._config.models.models,
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
        parts = key.split('.')
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
            if flags['SHOW_DATE_TO_LLM']:
                parts.append("dates")
            if flags['ENABLE_STRATEGIC_JOURNAL']:
                parts.append("mem")
            if flags['ENABLE_FEELING_LOG']:
                parts.append("feel")
            if not parts:
                parts.append("minimal")
            return "_" + "_".join(parts)

    def get_current_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        experiment = self.get_current_experiment()
        flags = self.get_feature_flags()

        return {
            'experiment': experiment.name,
            'description': experiment.description,
            'show_dates': flags['SHOW_DATE_TO_LLM'],
            'strategic_journal': flags['ENABLE_STRATEGIC_JOURNAL'],
            'feeling_log': flags['ENABLE_FEELING_LOG'],
        }

    def list_experiments(self) -> Dict[str, str]:
        """List available experiments with descriptions"""
        return {
            name: exp.description
            for name, exp in self._config.experiments.items()
        }

    def create_experiment(self, name: str, description: str, **kwargs) -> ExperimentConfig:
        """Create a new experiment configuration"""
        experiment = ExperimentConfig(
            name=name,
            description=description,
            show_dates=kwargs.get('show_dates', False),
            features=FeatureFlags(
                memory=MemoryFeatures(
                    strategic_journal=kwargs.get('strategic_journal', False),
                    feeling_log=kwargs.get('feeling_log', False),
                    full_trading_history=kwargs.get('full_trading_history', True),
                ),
                technical=TechnicalFeatures(
                    indicators=kwargs.get('technical_indicators', True),
                ),
                reporting=ReportingFeatures(),  # Use defaults
            )
        )

        self._config.experiments[name] = experiment
        self.logger.info(f"Created new experiment: {name}")
        return experiment

    def set_active_experiment(self, experiment_name: str):
        """Set the active experiment"""
        if experiment_name not in self._config.experiments:
            available = list(self._config.experiments.keys())
            raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available}")

        self._config.active_experiment = experiment_name
        self.logger.info(f"Active experiment set to: {experiment_name}")

    def get_config_as_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary for serialization"""
        return {
            'use_dummy_model': self._config.use_dummy_model,
            'test_mode': self._config.test_mode,
            'test_limit': self._config.test_limit,
            'active_experiment': self._config.active_experiment,
            'data': {
                'symbol': self._config.data.symbol,
                'start_date': self._config.data.start_date,
                'end_date': self._config.data.end_date,
            },
            'models': {
                'use_dummy_model': self._config.models.use_dummy_model,
                'models': self._config.models.models,
            },
            'experiments': {
                name: exp.to_dict() for name, exp in self._config.experiments.items()
            }
        }
