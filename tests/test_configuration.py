"""
Unit tests for the configuration management system.

Tests cover all components: Configuration classes, ConfigurationManager,
backward compatibility, and prompt building.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.config_classes import (
    GlobalConfig, ExperimentConfig, FeatureFlags,
    MemoryFeatures, TechnicalFeatures, DataSettings,
    ExperimentType
)
from src.configuration_manager import ConfigurationManager
from src.prompt_builder import PromptBuilder


class TestConfigurationClasses:
    """Test configuration data classes"""

    def test_global_config_creation(self):
        """Test GlobalConfig creation with defaults"""
        config = GlobalConfig()
        assert config.use_dummy_model == True
        assert config.test_mode == True
        assert config.active_experiment == "memory_feeling"
        assert len(config.experiments) == 6  # Default experiments loaded

    def test_global_config_with_custom_values(self):
        """Test GlobalConfig with custom values"""
        config = GlobalConfig(
            use_dummy_model=False,
            test_limit=50,
            active_experiment="baseline"
        )
        assert config.use_dummy_model == False
        assert config.test_limit == 50
        assert config.active_experiment == "baseline"

    def test_experiment_config_from_dict(self):
        """Test creating ExperimentConfig from legacy dict"""
        legacy_dict = {
            "description": "Test experiment",
            "SHOW_DATE_TO_LLM": True,
            "ENABLE_STRATEGIC_JOURNAL": True,
            "ENABLE_FEELING_LOG": False,
            "ENABLE_TECHNICAL_INDICATORS": True,
        }

        config = ExperimentConfig.from_dict("test_exp", legacy_dict)
        assert config.name == "test_exp"
        assert config.description == "Test experiment"
        assert config.show_dates == True
        assert config.features.memory.strategic_journal == True
        assert config.features.memory.feeling_log == False
        assert config.features.technical.indicators == True

    def test_experiment_config_to_dict(self):
        """Test converting ExperimentConfig back to dict"""
        config = ExperimentConfig(
            name="test",
            description="Test config",
            show_dates=True,
            features=FeatureFlags(
                memory=MemoryFeatures(strategic_journal=True, feeling_log=False),
                technical=TechnicalFeatures(indicators=True)
            )
        )

        data = config.to_dict()
        assert data["description"] == "Test config"
        assert data["SHOW_DATE_TO_LLM"] == True
        assert data["ENABLE_STRATEGIC_JOURNAL"] == True
        assert data["ENABLE_FEELING_LOG"] == False

    def test_data_settings_validation_valid(self):
        """Test DataSettings validation with valid data"""
        settings = DataSettings(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        errors = settings.validate()
        assert len(errors) == 0

    def test_data_settings_validation_invalid_symbol(self):
        """Test DataSettings validation with invalid symbol"""
        settings = DataSettings(symbol="", start_date="2020-01-01", end_date="2023-12-31")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Symbol cannot be empty" in errors[0]

    def test_data_settings_validation_invalid_dates(self):
        """Test DataSettings validation with invalid dates"""
        settings = DataSettings(
            symbol="^GSPC",
            start_date="invalid",
            end_date="2023-12-31"
        )
        errors = settings.validate()
        assert len(errors) >= 1
        assert "Invalid start_date" in errors[0]

    def test_data_settings_validation_date_order(self):
        """Test DataSettings validation with reversed date order"""
        settings = DataSettings(
            symbol="^GSPC",
            start_date="2023-12-31",
            end_date="2020-01-01"
        )
        errors = settings.validate()
        assert len(errors) == 1
        assert "must be before end date" in errors[0]

    def test_feature_flags_structure(self):
        """Test FeatureFlags structure"""
        flags = FeatureFlags()
        assert hasattr(flags.memory, 'strategic_journal')
        assert hasattr(flags.technical, 'indicators')
        assert hasattr(flags.reporting, 'comprehensive_reports')

    def test_memory_features_defaults(self):
        """Test MemoryFeatures default values"""
        features = MemoryFeatures()
        assert features.strategic_journal == True
        assert features.feeling_log == True
        assert features.full_trading_history == True

    def test_technical_features_defaults(self):
        """Test TechnicalFeatures default values"""
        features = TechnicalFeatures()
        assert features.indicators == True
        assert features.historical_series == True
        assert features.aggregated_stats == True


class TestConfigurationManager:
    """Test ConfigurationManager functionality"""

    def test_creation(self):
        """Test ConfigurationManager initialization"""
        manager = ConfigurationManager()
        assert manager._config is not None
        assert isinstance(manager._config, GlobalConfig)

    def test_get_current_experiment(self):
        """Test getting current experiment"""
        manager = ConfigurationManager()
        experiment = manager.get_current_experiment()
        assert experiment.name == "memory_feeling"
        assert "Memory + feeling" in experiment.description

    def test_get_current_experiment_invalid(self):
        """Test getting invalid experiment raises error"""
        manager = ConfigurationManager()
        manager._config.active_experiment = "nonexistent"

        with pytest.raises(ValueError, match="not found"):
            manager.get_current_experiment()

    def test_get_feature_flags(self):
        """Test getting flattened feature flags"""
        manager = ConfigurationManager()
        flags = manager.get_feature_flags()
        assert isinstance(flags, dict)
        assert 'ENABLE_STRATEGIC_JOURNAL' in flags
        assert 'ENABLE_TECHNICAL_INDICATORS' in flags
        assert 'SHOW_DATE_TO_LLM' in flags
        assert 'ENABLE_COMPREHENSIVE_REPORTS' in flags

    def test_get_data_settings(self):
        """Test getting data settings"""
        manager = ConfigurationManager()
        settings = manager.get_data_settings()
        assert 'SYMBOL' in settings
        assert 'DATA_START' in settings
        assert 'DATA_END' in settings

    def test_get_model_settings(self):
        """Test getting model settings"""
        manager = ConfigurationManager()
        settings = manager.get_model_settings()
        assert 'USE_DUMMY_MODEL' in settings
        assert 'TEST_MODE' in settings
        assert 'TEST_LIMIT' in settings
        assert 'LLM_MODELS' in settings

    def test_get_experiment_suffix_memory_feeling(self):
        """Test experiment suffix for memory_feeling"""
        manager = ConfigurationManager()
        suffix = manager.get_experiment_suffix()
        assert suffix == "_memory_feeling"

    def test_get_experiment_suffix_baseline(self):
        """Test experiment suffix for baseline"""
        manager = ConfigurationManager()
        manager.set_active_experiment("baseline")
        suffix = manager.get_experiment_suffix()
        assert suffix == "_baseline"

    def test_set_active_experiment_valid(self):
        """Test setting valid active experiment"""
        manager = ConfigurationManager()
        manager.set_active_experiment("baseline")
        assert manager._config.active_experiment == "baseline"

    def test_set_active_experiment_invalid(self):
        """Test setting invalid active experiment raises error"""
        manager = ConfigurationManager()

        with pytest.raises(ValueError, match="not found"):
            manager.set_active_experiment("nonexistent")

    def test_update_config_basic(self):
        """Test updating basic config settings"""
        manager = ConfigurationManager()

        # Update basic setting
        manager.update_config({'test_mode': False})
        assert manager._config.test_mode == False

        # Update nested setting
        manager.update_config({'data.symbol': 'AAPL'})
        assert manager._config.data.symbol == 'AAPL'

    def test_update_config_invalid_path(self):
        """Test updating invalid config path raises error"""
        manager = ConfigurationManager()

        with pytest.raises(ValueError, match="Invalid config path"):
            manager.update_config({'invalid.path': 'value'})

    def test_get_current_config_summary(self):
        """Test getting configuration summary"""
        manager = ConfigurationManager()
        summary = manager.get_current_config_summary()
        assert 'experiment' in summary
        assert 'description' in summary
        assert 'show_dates' in summary
        assert 'strategic_journal' in summary
        assert 'feeling_log' in summary

    def test_list_experiments(self):
        """Test listing experiments"""
        manager = ConfigurationManager()
        experiments = manager.list_experiments()
        assert isinstance(experiments, dict)
        assert 'baseline' in experiments
        assert 'memory_feeling' in experiments
        assert len(experiments) == 6  # Default experiments

    def test_create_experiment(self):
        """Test creating new experiment"""
        manager = ConfigurationManager()

        experiment = manager.create_experiment(
            name="custom_exp",
            description="Custom experiment",
            show_dates=True,
            strategic_journal=False,
            feeling_log=True
        )

        assert experiment.name == "custom_exp"
        assert experiment.show_dates == True
        assert experiment.features.memory.strategic_journal == False
        assert experiment.features.memory.feeling_log == True

        # Should be added to experiments dict
        assert "custom_exp" in manager._config.experiments

    @patch('src.configuration_manager.ConfigurationManager._load_config')
    def test_config_validation_failure(self, mock_load):
        """Test configuration validation failure"""
        # Create invalid config
        invalid_config = GlobalConfig()
        invalid_config.active_experiment = "nonexistent_experiment"
        mock_load.return_value = invalid_config

        with pytest.raises(ValueError, match="Invalid configuration"):
            ConfigurationManager()


class TestPromptBuilder:
    """Test PromptBuilder functionality"""

    def test_creation(self):
        """Test PromptBuilder initialization"""
        manager = ConfigurationManager()
        builder = PromptBuilder(manager)
        assert builder.config_manager == manager

    def test_build_system_prompt_basic(self):
        """Test building basic system prompt"""
        manager = ConfigurationManager()
        manager.set_active_experiment("baseline")  # Minimal features
        builder = PromptBuilder(manager)

        prompt = builder.build_system_prompt()
        assert "cautious but rational equity index hedge fund trader" in prompt
        assert "BUY" in prompt
        assert "HOLD" in prompt
        assert "SELL" in prompt

    def test_build_system_prompt_with_features(self):
        """Test building system prompt with all features enabled"""
        manager = ConfigurationManager()
        # memory_feeling has most features enabled
        builder = PromptBuilder(manager)

        prompt = builder.build_system_prompt()
        assert "Strategic journal" in prompt
        assert "RSI measures momentum" in prompt
        assert "MACD crossing" in prompt

    def test_build_period_summary_prompt(self):
        """Test building period summary prompt"""
        manager = ConfigurationManager()
        builder = PromptBuilder(manager)

        prompt = builder.build_period_summary_prompt("Month", {})
        assert "three clearly separated sections" in prompt
        assert "Explanation:" in prompt
        assert "Strategic journal:" in prompt
        assert "Feeling log:" in prompt

    def test_build_technical_indicators_description_enabled(self):
        """Test technical indicators description when enabled"""
        manager = ConfigurationManager()
        builder = PromptBuilder(manager)

        desc = builder._build_technical_indicators_description()
        assert "RSI" in desc
        assert "MACD" in desc
        assert "Stochastic" in desc
        assert "Bollinger" in desc

    def test_build_technical_indicators_description_disabled(self):
        """Test technical indicators description when disabled"""
        manager = ConfigurationManager()
        # Create a custom experiment with technical indicators disabled
        manager.create_experiment(
            name="no_tech",
            description="No technical indicators",
            show_dates=False,
            strategic_journal=True,
            feeling_log=True,
            technical_indicators=False
        )
        manager.set_active_experiment("no_tech")
        builder = PromptBuilder(manager)

        desc = builder._build_technical_indicators_description()
        assert "RSI" not in desc
        assert "MACD" not in desc


class TestBackwardCompatibility:
    """Test backward compatibility with legacy config"""

    def test_legacy_variables_exist(self):
        """Test that legacy global variables are available"""
        from src.config_compat import (
            ENABLE_STRATEGIC_JOURNAL,
            ENABLE_TECHNICAL_INDICATORS,
            SHOW_DATE_TO_LLM,
            USE_DUMMY_MODEL,
            SYMBOL,
            DATA_START,
            DATA_END,
            get_experiment_suffix,
            get_current_config_summary
        )

        # Check that variables exist and have expected types
        assert isinstance(ENABLE_STRATEGIC_JOURNAL, bool)
        assert isinstance(ENABLE_TECHNICAL_INDICATORS, bool)
        assert isinstance(SHOW_DATE_TO_LLM, bool)
        assert isinstance(USE_DUMMY_MODEL, bool)
        assert isinstance(SYMBOL, str)
        assert DATA_START == "2015-01-01" or isinstance(DATA_START, str)
        assert DATA_END == "2023-12-31" or isinstance(DATA_END, str)
        assert callable(get_experiment_suffix)
        assert callable(get_current_config_summary)

    def test_experiment_suffix_matches(self):
        """Test that new and old suffix generation match"""
        from src.config_compat import get_experiment_suffix
        from src.configuration_manager import ConfigurationManager

        manager = ConfigurationManager()
        new_suffix = manager.get_experiment_suffix()
        old_suffix = get_experiment_suffix()

        assert new_suffix == old_suffix

    def test_config_summary_matches(self):
        """Test that new and old config summary match"""
        from src.config_compat import get_current_config_summary
        from src.configuration_manager import ConfigurationManager

        manager = ConfigurationManager()
        new_summary = manager.get_current_config_summary()
        old_summary = get_current_config_summary()

        assert new_summary == old_summary

    def test_legacy_experiment_configs_exist(self):
        """Test that legacy EXPERIMENT_CONFIGS still exists"""
        from src.config_compat import EXPERIMENT_CONFIGS

        assert isinstance(EXPERIMENT_CONFIGS, dict)
        assert 'baseline' in EXPERIMENT_CONFIGS
        assert 'memory_feeling' in EXPERIMENT_CONFIGS
        assert len(EXPERIMENT_CONFIGS) == 6
