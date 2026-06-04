# Test Suite Documentation

## Overview

This document provides comprehensive documentation for the LLM Finance Framework test suite, covering **230+ individual tests** across **19 test files**. The test suite ensures the reliability, correctness, and robustness of the framework's core components.

The tests are organized by functional areas and follow a systematic approach to validate all critical system components, including dynamic symbol naming, chain of thought reasoning, trader personality systems, and comprehensive end-to-end integration testing.

---

### 📋 Report Generation Tests

#### **test_report_generator.py** - Comprehensive Report Generation Testing
**Purpose**: Validates the enhanced report generation system with professional HTML styling, multiple report sections, and robust data processing.

**Test Classes**:
- **TestBaselineStrategiesSection**: Baseline strategy showcase validation
  - Category-based performance aggregation and sorting
  - Individual strategy details and metrics display
  - Data presence and missing data handling
  - HTML version generation and formatting
- **TestLLMIndicatorAlignmentSection**: Technical indicator alignment analysis
  - Effectiveness ranking across all 6 indicators
  - Pattern analysis and decision insights
  - Missing alignment data handling
  - HTML version with proper formatting
- **TestStrategyComparisonInsights**: Strategy positioning and comparison
  - LLM performance extraction and category comparison
  - Win rate calculations and formatting
  - Missing comparison data graceful handling
  - HTML version with visual indicators
- **TestReportGeneratorIntegration**: End-to-end report generation
  - All report sections integration testing
  - Mock data consistency and error handling
  - Report generation pipeline validation

#### **test_dynamic_symbols.py** - Dynamic Symbol Naming & Personality Integration
**Purpose**: Integration tests for dynamic symbol naming functionality and trader personality system across all system components.

**Test Classes**:
- **TestDynamicSymbols**: End-to-end symbol integration testing
  - Configuration symbol info validation
  - ConfigurationManager symbol exposure
  - PromptBuilder dynamic symbol usage
  - PerformanceTracker symbol integration
  - Parameterized testing across different symbols (SPY, QQQ, AAPL, etc.)
  - Complete symbol flow from config to prompts
- **TestPersonalityConfiguration**: Personality system validation
  - All 5 personality types (cautious, aggressive, balanced, momentum, contrarian)
  - Personality-specific prompt generation and behavioral frameworks
  - Decision pattern differences by personality type
  - Market regime performance variations

---

## Test Categories

### 📊 Data Source Tests

#### **test_stooq_data.py** - Stooq Historical Data Integration
**Purpose**: Comprehensive testing of the Stooq API integration for historical market data, ensuring reliable data access and error handling.

**Test Classes**:
- **TestStooqDirectAPI**: Direct API call validation
  - API connection and response format validation
  - Individual symbol data retrieval (AAPL, SPY)
  - Multiple symbol batch processing
  - Date parsing and format validation
  - Data quality checks (price ranges, volume validation)
- **TestStooqDataTransformation**: Data processing validation
  - CSV to DataFrame conversion
  - Column mapping and data type validation
  - Historical vs current data format handling
- **TestStooqHistoricalData**: Historical data validation
  - API endpoint functionality for historical ranges
  - Data completeness and null value checking
  - Historical data volume validation

#### **test_data_sources.py** - Data Source Management System
**Purpose**: Testing of the unified data source abstraction layer supporting multiple data providers with fallback mechanisms.

**Test Classes**:
- **TestDataSourceManager**: Core data source management
  - Manager initialization and source registration
  - Data retrieval with configuration-based source selection
- **TestStooqDataSource**: Stooq-specific functionality
  - Symbol formatting (.us suffix handling)
  - Date range parameter conversion (YYYY-MM-DD → YYYYMMDD)
  - Network error handling and timeout management
  - Invalid response format detection
- **TestDataSourceIntegration**: End-to-end integration
  - Primary source failure with automatic CSV fallback
  - Configuration validation and error messages
  - Data format consistency across sources

#### **test_stooq_data.py** - Stooq Historical Data Integration
**Purpose**: Comprehensive testing of the Stooq API integration for historical market data, ensuring reliable data access and error handling.

**Test Classes**:
- **TestStooqDirectAPI**: Direct API call validation
  - API connection and response format validation
  - Individual symbol data retrieval (AAPL, SPY)
  - Multiple symbol batch processing
  - Date parsing and format validation
  - Data quality checks (price ranges, volume validation)
- **TestStooqDataTransformation**: Data processing validation
  - CSV to DataFrame conversion
  - Column mapping and data type validation
  - Historical vs current data format handling
- **TestStooqHistoricalData**: Historical data validation
  - API endpoint functionality for historical ranges
  - Data completeness and null value checking
  - Historical data volume validation

#### **test_integration.py** - End-to-End System Integration
**Purpose**: Complete pipeline testing from data ingestion through feature computation to prompt generation and trading simulation.

**Test Classes**:
- **TestIntegration**: System-level integration testing
  - Configuration validation and user guidance
  - Stooq network error handling without fallback
  - Successful historical data fetching and processing
  - CSV data source functionality as fallback
  - Date range parameter handling and validation
  - Backward compatibility with old variable names
- **TestErrorHandling**: Resilience and error recovery
  - Stooq API server errors and recovery
  - Invalid response format handling
  - Data quality issue detection and reporting
- **TestDataQuality**: Data integrity validation
  - Stooq data structure consistency checks
  - Symbol formatting edge case handling
  - Historical data completeness verification

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

#### **test_indicators.py** - Technical Indicators Computation
**Purpose**: Ensures accurate calculation of technical indicators used in LLM prompts and analysis.

**Test Classes**:
- **TestTechnicalIndicators**: Technical analysis validation
  - MACD computation (line, signal, histogram)
  - Stochastic Oscillator calculation (%K, %D)
  - Bollinger Bands (upper, middle, lower)
  - Custom parameter handling
  - Edge cases (constant prices, insufficient data, NaN inputs)

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

#### **test_integration.py** - Full Pipeline Integration
**Purpose**: End-to-end testing of the complete data processing pipeline from raw OHLC data to LLM prompts.

**Test Classes**:
- **TestFullPipelineIntegration**: Pipeline validation
  - Complete data preparation with new indicators
  - Feature engineering and prompt building
  - Configuration flag handling
  - Data validation and formatting

---

## Test Coverage Overview

| Category | Files | Tests | Coverage Focus |
|----------|-------|-------|----------------|
| Configuration | 2 | ~50 | Settings, compatibility, validation |
| Core Systems | 2 | ~60 | Memory, performance tracking |
| Data Processing | 2 | ~25 | Indicators, pipeline integration |
| Components | 2 | ~40 | Journals, trade history |
| Analysis | 3 | ~15 | Calibration, feature validation, chain of thought |
| Report Generation | 1 | ~18 | Enhanced reporting, HTML styling, data processing |
| Integration | 2 | ~14 | Dynamic symbols, personality system, end-to-end |
| Chain of Thought | 4 | ~25 | Reasoning parsing, integration, dummy model, end-to-end |
| **Total** | **19** | **230+** | **Complete system validation** |

#### **test_chain_of_thought_analysis.py** - Chain of Thought Reasoning
**Purpose**: Validates the structured analytical reasoning feature that enables step-by-step LLM decision processes.

**Test Classes**:
- **TestChainOfThoughtParsing**: Response parsing validation
  - Conditional line indexing (reasoning on line 0, decision on line 1)
  - ENABLE_CHAIN_OF_THOUGHT toggle functionality
  - CSV output format with chain_of_thought column
  - Backward compatibility when feature disabled
- **TestChainOfThoughtIntegration**: Feature integration testing
  - Independent toggle operation across all experiment types
  - Prompt enhancement validation
  - Decision quality analysis
  - Performance impact assessment

#### **test_chain_of_thought_integration.py** - Chain of Thought System Integration
**Purpose**: Tests integration of chain of thought reasoning with other system components.

**Test Classes**:
- **TestChainOfThoughtWithExperiments**: Cross-experiment compatibility
  - Works with all 6 experiment types (baseline through dates_full)
  - Independent of memory, personality, and date settings
  - Breaking change validation (independent toggle)
- **TestChainOfThoughtPrompts**: Prompt construction validation
  - Reasoning prompt inclusion based on toggle
  - Response format expectations
  - Token usage and context management

#### **test_dummy_model_chain_of_thought.py** - Dummy Model Chain of Thought
**Purpose**: Ensures dummy model generates correct response formats for chain of thought feature.

**Test Classes**:
- **TestDummyModelFormats**: Response format generation
  - Correct format generation for all 8 feature flag combinations
  - ENABLE_CHAIN_OF_THOUGHT = True/False variations
  - Compatible with other feature toggles

#### **test_end_to_end_chain_of_thought.py** - End-to-End Chain of Thought
**Purpose**: Complete pipeline testing with chain of thought reasoning enabled.

**Test Classes**:
- **TestEndToEndChainOfThought**: Full pipeline validation
  - Data processing through decision making with reasoning
  - Result parsing and CSV output generation
  - Report generation with chain of thought data
  - Performance metrics with reasoning enabled

#### **test_dynamic_symbols.py** - Personality System Integration
**Purpose**: Tests the 5 trader personality types (cautious, aggressive, balanced, momentum, contrarian) integration across the system.

**Additional Test Classes**:
- **TestPersonalityConfiguration**: Personality setting validation
  - All 5 personality types configuration
  - Personality-specific prompt generation
  - Behavioral framework application
- **TestPersonalityImpact**: Decision pattern analysis by personality
  - Risk tolerance differences (cautious vs aggressive)
  - Decision style variations (systematic, reactive, contrarian)
  - Market regime performance differences

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
