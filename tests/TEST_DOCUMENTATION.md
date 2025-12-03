# Test Suite Documentation

## Overview

This document provides comprehensive documentation for the LLM Finance Framework test suite, covering **148 individual tests** across **10 test files**. The test suite ensures the reliability, correctness, and robustness of the framework's core components.

The tests are organized by functional areas and follow a systematic approach to validate all critical system components.

---

## Test Categories

### 🔧 Configuration Tests

#### **test_configuration.py** - Configuration System Testing
**Purpose**: Comprehensive validation of the new type-safe configuration system and backward compatibility.

**Test Classes**:
- **TestConfigurationClasses**: Validates configuration data classes and type safety
  - GlobalConfig creation and validation
  - ExperimentConfig conversion between dict and object formats
  - DataSettings validation (symbol, date ranges)
  - FeatureFlags structure and defaults
- **TestConfigurationManager**: Tests the central configuration management
  - Manager initialization and experiment retrieval
  - Feature flag extraction and flattening
  - Configuration updates and validation
  - Experiment creation and switching
- **TestPromptBuilder**: Validates prompt construction logic
  - System prompt building with/without features
  - Period summary prompt generation
  - Technical indicators description handling
- **TestBackwardCompatibility**: Ensures legacy config compatibility
  - Legacy global variables exposure
  - Config summary consistency
  - Experiment configuration migration

#### **test_config_consistency.py** - Configuration Migration Testing
**Purpose**: Ensures proper migration from legacy config.py settings to the new configuration system.

**Test Classes**:
- **TestConfigConsistency**: Validates setting transfer accuracy
  - Debug flags, API endpoints, window constants
  - Core model and data settings
  - Experiment feature flag mapping
- **TestConfigBehavior**: Tests runtime configuration changes
  - Dynamic setting updates
  - Behavior modification validation

---

### 🧮 Core System Tests

#### **test_memory_system.py** - Memory Management Architecture
**Purpose**: Comprehensive testing of the 4-layer hierarchical memory system that enables LLM learning and adaptation.

**Test Classes**:
- **TestMemoryItem**: Memory data structure validation
  - Creation with/without technical statistics
  - Serialization/deserialization (dict/string formats)
- **TestPeriodStats**: Period performance tracking
  - Statistics accumulation and reset
  - Data persistence and conversion
- **TestPeriodConfig**: Period configuration management
  - Time period definitions and naming
  - Custom parameter handling
- **TestMemoryManager**: Memory storage and retrieval
  - Item addition with limits and technical stats
  - Memory block generation and formatting
  - Memory clearing operations
- **TestPeriodManager**: Period boundary and summarization logic
  - Time boundary detection (weekly/monthly/yearly)
  - Statistics updates and period summaries
  - Active period tracking

#### **test_performance_tracker.py** - Performance Metrics Tracking
**Purpose**: Validates the performance tracking system that monitors trading decisions, returns, and position management.

**Test Classes**:
- **TestPerformanceTracker**: Comprehensive performance validation
  - Position tracking across decision changes
  - Performance summary generation
  - Win rate calculations and edge cases
  - Final metrics computation and reset functionality

---

### 📊 Data Processing Tests

#### **test_indicators.py** - Technical Indicators Computation
**Purpose**: Ensures accurate calculation of technical indicators used in LLM prompts and analysis.

**Test Classes**:
- **TestTechnicalIndicators**: Technical analysis validation
  - MACD computation (line, signal, histogram)
  - Stochastic Oscillator calculation (%K, %D)
  - Bollinger Bands (upper, middle, lower)
  - Custom parameter handling
  - Edge cases (constant prices, insufficient data, NaN inputs)

#### **test_integration.py** - Full Pipeline Integration
**Purpose**: End-to-end testing of the complete data processing pipeline from raw OHLC data to LLM prompts.

**Test Classes**:
- **TestFullPipelineIntegration**: Pipeline validation
  - Complete data preparation with new indicators
  - Feature engineering and prompt building
  - Configuration flag handling
  - Data validation and formatting

---

### 📝 Component Tests

#### **test_journal_manager.py** - Strategic Journal Management
**Purpose**: Tests the rolling window journal system that maintains recent trading decisions and reasoning.

**Test Classes**:
- **TestJournalManager**: Journal functionality validation
  - Entry addition and rolling window management
  - Time-relative formatting (days/weeks/months ago)
  - Empty state handling and data preservation
  - Journal block generation for prompts

#### **test_trade_history_manager.py** - Trade History Management
**Purpose**: Validates the complete chronological trading record formatting and management.

**Test Classes**:
- **TestTradeHistoryManager**: History management testing
  - Entry addition with/without date visibility
  - Result rounding and formatting
  - History block generation
  - Large dataset handling and mixed decision scenarios

---

### 📈 Analysis Tests

#### **test_calibration.py** - Model Calibration Analysis
**Purpose**: Generates calibration plots and statistics to assess LLM confidence vs. actual performance.

**Functions**:
- **test_calibration_plots**: End-to-end calibration analysis
  - Loads parsed results from completed experiments
  - Generates calibration plots for all model configurations
  - Computes win rates, prediction accuracy, and statistical summaries
  - Handles missing data gracefully (skips if no results exist)

#### **test_strategic_journal_config.py** - Strategic Journal Configuration
**Purpose**: Verifies that strategic journal and feeling log features work correctly in prompts and response parsing.

**Functions**:
- **test_config**: Configuration validation
  - Feature flag verification
  - System prompt generation with features
  - Response parsing with journal/feelings
  - Expected output format validation

---

## Test Coverage Overview

| Category | Files | Tests | Coverage Focus |
|----------|-------|-------|----------------|
| Configuration | 2 | ~50 | Settings, compatibility, validation |
| Core Systems | 2 | ~60 | Memory, performance tracking |
| Data Processing | 2 | ~25 | Indicators, pipeline integration |
| Components | 2 | ~40 | Journals, trade history |
| Analysis | 2 | ~10 | Calibration, feature validation |
| **Total** | **10** | **148** | **Complete system validation** |

---

## Running the Tests

### Full Test Suite
```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_memory_system.py -v

# Run specific test class
python -m pytest tests/test_configuration.py::TestConfigurationManager -v
```

### Development Workflow
```bash
# Quick test run during development
python scripts/dev-workflow.py test

# Full quality check (includes tests)
python scripts/dev-workflow.py check
```

### CI/CD Integration
The test suite runs automatically on:
- All pull requests
- Main branch pushes
- Release preparation

---

## Test Organization Principles

### **Isolation**: Each test is self-contained with proper setup/teardown
### **Comprehensive**: Tests cover both happy paths and edge cases
### **Maintainable**: Clear naming and documentation for easy maintenance
### **Reliable**: Deterministic results with proper mocking of external dependencies

---

## Contributing to Tests

When adding new features:

1. **Create corresponding test files** in the `tests/` directory
2. **Follow naming conventions**: `test_<component>.py`
3. **Include comprehensive docstrings** explaining test purpose
4. **Test edge cases and error conditions**
5. **Update this README** with new test documentation

---

## Test Dependencies

The test suite requires:
- `pytest` - Test framework and runner
- `pytest-cov` - Coverage reporting
- Test data fixtures and mock objects
- External dependencies (pandas, numpy, etc.)

All dependencies are included in the project's `pyproject.toml` and can be installed with:
```bash
pip install -e .[dev]
```
