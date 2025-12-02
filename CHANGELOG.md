# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-12-02

### Added
- Complete technical indicators suite (RSI, MACD, Stochastic Oscillator, Bollinger Bands)
- Multi-model LLM testing framework with Chimera, GPT-OSS-20B, and Dummy Model support
- Enhanced technical indicators memory system for improved LLM context
- Neutral and factual RSI system prompts for better AI decision-making
- Comprehensive technical analysis integration in LLM prompts

### Changed
- Redesigned HOLD decision analysis with meaningful performance metrics
- Enhanced comprehensive reporting with improved structure and decision analysis
- Updated CI/CD pipeline to Python 3.11 with relaxed type checking for compatibility
- Improved project documentation and collaboration guidelines

### Fixed
- HOLD decision context success rate calculation accuracy
- Syntax errors in development workflow scripts
- CI pipeline formatting and style check compatibility issues
- Repository URL placeholders and documentation references

### Technical Details
- Core dependencies: pandas, scipy, matplotlib, requests
- Python 3.8+ compatibility maintained
- Removed seaborn dependency for cleaner requirements

## [0.1.0] - 2025-01-28

### Added
- Initial release of LLM Finance Framework
- Core experiment configurations (baseline, memory_only, memory_feeling)
- OpenRouter API integration for multiple LLM models
- Dual-criteria HOLD decision analysis
- Comprehensive statistical validation
- Example results and documentation

### Research Features
- Memory adaptation tracking with strategic journals
- Probabilistic calibration assessment
- Behavioral pattern analysis
- Out-of-sample validation
- Bootstrap significance testing

---

## Version Numbering Convention

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
  - **MAJOR**: Breaking changes, incompatible API changes
  - **MINOR**: New features, backward-compatible
  - **PATCH**: Bug fixes, backward-compatible

### Pre-release Labels
- `alpha`: Early testing, API may change
- `beta`: Feature complete, bug fixing
- `rc`: Release candidate, final testing

### Development Versions
- `dev`: Current development state
- `SNAPSHOT`: Automatic builds

---

## Release Process

### For Private Development:
```bash
# Check current version
python scripts/version.py get

# Bump version (patch, minor, or major)
python scripts/version.py bump patch

# Create git tag
python scripts/version.py tag

# Push changes and tags
git push origin main --tags
```

### For Public Releases:
1. Update CHANGELOG.md with release notes
2. Run full test suite
3. Create release on GitHub
4. Update documentation if needed

---

## Types of Changes

- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities
