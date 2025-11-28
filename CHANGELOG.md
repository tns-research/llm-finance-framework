# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial framework setup for LLM finance experiments
- Baseline trading strategies (Buy & Hold, Momentum, Mean Reversion, etc.)
- Statistical validation with bootstrap testing
- Comprehensive reporting and visualization
- GitHub Actions CI/CD pipeline
- Development tools and linting configuration

### Changed
- Migrated from master to main branch

### Technical Details
- Python 3.8+ compatibility
- Core dependencies: pandas, scipy, matplotlib, seaborn, requests

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
