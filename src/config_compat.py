"""
Backward Compatibility Layer for Legacy Configuration

This module provides the same global variables that existed before,
but now they're computed from the new ConfigurationManager.

This allows existing code to continue working while we migrate to the new system.
"""

import logging
from typing import Dict, Any

# Import the new configuration system
from .configuration_manager import ConfigurationManager
from .prompt_builder import PromptBuilder

# Create global configuration manager instance
_config_manager = ConfigurationManager()
_prompt_builder = PromptBuilder(_config_manager)
_logger = logging.getLogger(__name__)

def _get_feature_flags() -> Dict[str, bool]:
    """Get current feature flags"""
    try:
        return _config_manager.get_feature_flags()
    except Exception as e:
        _logger.warning(f"Failed to get feature flags, using defaults: {e}")
        return {
            'ENABLE_STRATEGIC_JOURNAL': True,
            'ENABLE_FEELING_LOG': True,
            'ENABLE_FULL_TRADING_HISTORY': True,
            'ENABLE_TECHNICAL_INDICATORS': True,
            'SHOW_DATE_TO_LLM': False,
            'ENABLE_COMPREHENSIVE_REPORTS': True,
            'ENABLE_PLOTS': True,
            'ENABLE_STATISTICAL_VALIDATION': True,
        }

def _get_data_settings() -> Dict[str, Any]:
    """Get current data settings"""
    try:
        return _config_manager.get_data_settings()
    except Exception as e:
        _logger.warning(f"Failed to get data settings, using defaults: {e}")
        return {
            'SYMBOL': '^GSPC',
            'DATA_START': '2015-01-01',
            'DATA_END': '2023-12-31',
        }

def _get_model_settings() -> Dict[str, Any]:
    """Get current model settings"""
    try:
        return _config_manager.get_model_settings()
    except Exception as e:
        _logger.warning(f"Failed to get model settings, using defaults: {e}")
        return {
            'USE_DUMMY_MODEL': True,
            'TEST_MODE': True,
            'TEST_LIMIT': 15,
            'LLM_MODELS': [],
        }

# Legacy global variables - now computed from ConfigurationManager
flags = _get_feature_flags()
data_settings = _get_data_settings()
model_settings = _get_model_settings()

# Debug settings - now read from new system
DEBUG_SHOW_FULL_PROMPT = _config_manager._config.debug_show_full_prompt
START_ROW = _config_manager._config.start_row
OPENROUTER_API_BASE = _config_manager._config.openrouter_api_base

# Feature flags
ENABLE_STRATEGIC_JOURNAL = flags['ENABLE_STRATEGIC_JOURNAL']
ENABLE_FEELING_LOG = flags['ENABLE_FEELING_LOG']
ENABLE_FULL_TRADING_HISTORY = flags['ENABLE_FULL_TRADING_HISTORY']
ENABLE_TECHNICAL_INDICATORS = flags['ENABLE_TECHNICAL_INDICATORS']
SHOW_DATE_TO_LLM = flags['SHOW_DATE_TO_LLM']

# Data settings
SYMBOL = data_settings['SYMBOL']
DATA_START = data_settings['DATA_START']
DATA_END = data_settings['DATA_END']

# Model settings
USE_DUMMY_MODEL = model_settings['USE_DUMMY_MODEL']
TEST_MODE = model_settings['TEST_MODE']
TEST_LIMIT = model_settings['TEST_LIMIT']
LLM_MODELS = model_settings['LLM_MODELS']

# Utility functions for backward compatibility
def get_experiment_suffix():
    """Get experiment suffix for file naming"""
    return _config_manager.get_experiment_suffix()

def get_current_config_summary():
    """Get configuration summary for display"""
    return _config_manager.get_current_config_summary()

def list_experiments():
    """List available experiments with descriptions"""
    return _config_manager.list_experiments()

# Legacy experiment configs (for reference only - now managed by ConfigurationManager)
EXPERIMENT_CONFIGS = {
    "baseline": {
        "description": "Minimal context: no dates, no memory, no feeling",
        "SHOW_DATE_TO_LLM": False,
        "ENABLE_STRATEGIC_JOURNAL": False,
        "ENABLE_FEELING_LOG": False,
    },
    "memory_only": {
        "description": "Memory/journal only: no dates, no feeling",
        "SHOW_DATE_TO_LLM": False,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": False,
    },
    "memory_feeling": {
        "description": "Memory + feeling: no dates",
        "SHOW_DATE_TO_LLM": False,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": True,
    },
    "dates_only": {
        "description": "Dates only: no memory, no feeling",
        "SHOW_DATE_TO_LLM": True,
        "ENABLE_STRATEGIC_JOURNAL": False,
        "ENABLE_FEELING_LOG": False,
    },
    "dates_memory": {
        "description": "Dates + memory: no feeling",
        "SHOW_DATE_TO_LLM": True,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": False,
    },
    "dates_full": {
        "description": "Full context: dates + memory + feeling",
        "SHOW_DATE_TO_LLM": True,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": True,
    },
}

# Generate prompts using new system
SYSTEM_PROMPT = _prompt_builder.build_system_prompt()
JOURNAL_SYSTEM_PROMPT = _prompt_builder.build_period_summary_prompt("Period", {})

# For backward compatibility - these may be referenced by legacy code
ACTIVE_EXPERIMENT = _config_manager._config.active_experiment

# Technical indicator constants
MA20_WINDOW = 20  # 20-day moving average window
RET_5D_WINDOW = 5  # 5-day return window
VOL20_WINDOW = 20  # 20-day volatility window

# Position mapping
POSITION_MAP = {
    "BUY": 1.0,
    "HOLD": 0.0,
    "SELL": -1.0,
}
