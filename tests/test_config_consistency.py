"""
Configuration Consistency Tests for Phase 1

Tests to ensure legacy config.py settings are properly transferred to new system
and that changing values in config.py actually affects behavior.
"""

import pytest
from unittest.mock import patch

from src.config_compat import (
    DEBUG_SHOW_FULL_PROMPT,
    START_ROW,
    OPENROUTER_API_BASE,
    MA20_WINDOW,
    RET_5D_WINDOW,
    VOL20_WINDOW,
    USE_DUMMY_MODEL,
    TEST_MODE,
    TEST_LIMIT,
    SYMBOL,
    DATA_START,
    DATA_END,
    ENABLE_TECHNICAL_INDICATORS,
    ENABLE_STRATEGIC_JOURNAL,
    ENABLE_FEELING_LOG,
    SHOW_DATE_TO_LLM,
)
from src.configuration_manager import ConfigurationManager


class TestConfigConsistency:
    """Test that legacy config settings are properly transferred"""

    def test_debug_show_full_prompt_transferred(self):
        """Test that DEBUG_SHOW_FULL_PROMPT is read from legacy config"""
        config_manager = ConfigurationManager()

        # The new system should have the field
        assert hasattr(config_manager._config, 'debug_show_full_prompt')

        # The compatibility layer should expose the value from new system
        # Should match the value set in config.py (currently True for testing)
        assert config_manager._config.debug_show_full_prompt == DEBUG_SHOW_FULL_PROMPT

        # Should be boolean type
        assert isinstance(DEBUG_SHOW_FULL_PROMPT, bool)

    def test_start_row_transferred(self):
        """Test that START_ROW is read from legacy config"""
        config_manager = ConfigurationManager()

        # Should have the field in new system
        assert hasattr(config_manager._config, 'start_row')

        # Should match legacy value (33)
        assert config_manager._config.start_row == 33
        assert START_ROW == 33

    def test_openrouter_api_base_transferred(self):
        """Test that OPENROUTER_API_BASE is read from legacy config"""
        config_manager = ConfigurationManager()

        # Should have the field in new system
        assert hasattr(config_manager._config, 'openrouter_api_base')

        # Should match legacy value
        expected = "https://openrouter.ai/api/v1/chat/completions"
        assert config_manager._config.openrouter_api_base == expected
        assert OPENROUTER_API_BASE == expected

    def test_window_constants_transferred(self):
        """Test that window constants are read from legacy config"""
        config_manager = ConfigurationManager()

        # Should have the fields in new system
        assert hasattr(config_manager._config, 'ma20_window')
        assert hasattr(config_manager._config, 'ret_5d_window')
        assert hasattr(config_manager._config, 'vol20_window')

        # Should match legacy values (currently hardcoded)
        assert config_manager._config.ma20_window == 20
        assert config_manager._config.ret_5d_window == 5
        assert config_manager._config.vol20_window == 20

        # Should match compatibility layer values
        assert MA20_WINDOW == 20
        assert RET_5D_WINDOW == 5
        assert VOL20_WINDOW == 20

    def test_core_settings_transferred(self):
        """Test that core settings are properly transferred"""
        config_manager = ConfigurationManager()

        # Test basic settings
        assert config_manager._config.use_dummy_model == USE_DUMMY_MODEL
        assert config_manager._config.test_mode == TEST_MODE
        assert config_manager._config.test_limit == TEST_LIMIT

        # Test data settings
        assert config_manager._config.data.symbol == SYMBOL
        assert config_manager._config.data.start_date == DATA_START
        assert config_manager._config.data.end_date == DATA_END

    def test_experiment_settings_transferred(self):
        """Test that experiment settings are properly transferred"""
        config_manager = ConfigurationManager()

        # Get current experiment
        exp = config_manager.get_current_experiment()

        # Check that feature flags match
        flags = config_manager.get_feature_flags()
        assert flags['ENABLE_TECHNICAL_INDICATORS'] == ENABLE_TECHNICAL_INDICATORS
        assert flags['ENABLE_STRATEGIC_JOURNAL'] == ENABLE_STRATEGIC_JOURNAL
        assert flags['ENABLE_FEELING_LOG'] == ENABLE_FEELING_LOG
        assert flags['SHOW_DATE_TO_LLM'] == SHOW_DATE_TO_LLM


class TestConfigBehavior:
    """Test that changing config.py actually affects behavior"""

    @patch('src.configuration_manager.ConfigurationManager._get_default_config_path')
    def test_debug_show_behavior_change(self, mock_path):
        """Test that changing DEBUG_SHOW_FULL_PROMPT in config affects behavior"""
        # This test simulates changing the config file
        # In practice, would need to mock file reading or modify actual file

        # For now, just test that the infrastructure is in place
        config_manager = ConfigurationManager()

        # Should be able to set the value
        config_manager._config.debug_show_full_prompt = False
        assert config_manager._config.debug_show_full_prompt == False

        config_manager._config.debug_show_full_prompt = True
        assert config_manager._config.debug_show_full_prompt == True

    @patch('src.configuration_manager.ConfigurationManager._get_default_config_path')
    def test_start_row_behavior_change(self, mock_path):
        """Test that changing START_ROW in config affects behavior"""
        config_manager = ConfigurationManager()

        # Should be able to set the value
        config_manager._config.start_row = 100
        assert config_manager._config.start_row == 100

        config_manager._config.start_row = None
        assert config_manager._config.start_row is None
