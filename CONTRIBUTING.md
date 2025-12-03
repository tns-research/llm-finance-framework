# Contributing to LLM Finance Framework

We welcome contributions from researchers, developers, and financial practitioners! This project aims to advance understanding of AI in financial decision-making through rigorous empirical research.

## 🎯 Ways to Contribute

### 🔬 Research Contributions
- **New LLM Models**: Test additional language models (Claude, Gemini, Llama, etc.)
- **Strategy Development**: Implement new baseline strategies or LLM prompting techniques
- **Statistical Methods**: Enhance validation frameworks or develop new metrics
- **Market Analysis**: Test on different markets, time periods, or asset classes
- **Behavioral Analysis**: Develop new methods for detecting AI trading biases
- **Experiment Design**: Create new experimental workflows and configurations

### 💻 Code Contributions
- **Bug Fixes**: Identify and fix issues in the codebase
- **Feature Implementation**: Add new analysis tools or visualization methods
- **Performance Optimization**: Improve execution speed or memory usage
- **Documentation**: Enhance code documentation and examples
- **Configuration System**: Extend the flexible configuration architecture

### 🧪 Testing & Validation
- **Test Coverage**: Add unit tests and integration tests
- **Result Validation**: Verify experimental results and statistical analyses
- **Cross-Platform Testing**: Ensure compatibility across different environments
- **Validation Scripts**: Enhance mathematical and indicator validation tools

### 📚 Documentation
- **README Improvements**: Enhance setup instructions and examples
- **Research Documentation**: Document methodologies and findings
- **Tutorial Creation**: Develop guides for specific use cases
- **API Documentation**: Document configuration options and experimental workflows

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **Git**: For version control and collaboration
- **OpenRouter Account**: For LLM API access (optional, dummy model available for testing)

### OpenRouter Setup (Optional)
- **Get API Key**: Sign up at [OpenRouter.ai](https://openrouter.ai/)
- **Documentation**: [OpenRouter API Docs](https://openrouter.ai/docs)
- **Model List**: [Available Models](https://openrouter.ai/docs/models)
- **Environment Variable**: Set `OPENROUTER_API_KEY` in your environment

### Quick Development Setup (Windows)
```bash
# Clone repository
git clone https://github.com/tns-research/llm-finance-framework.git
cd llm-finance-framework

# Use development helper (recommended)
dev.bat setup

# Run full check suite
dev.bat check

# Test basic functionality
dev.bat run-dummy
```

### Manual Development Setup
```bash
# Clone repository
git clone https://github.com/tns-research/llm-finance-framework.git
cd llm-finance-framework

# Install in development mode (includes all tools)
pip install -e .[dev]

# Or install just core dependencies
pip install -e .
```

### Linux/Mac Setup (using Makefile)
```bash
# Clone repository
git clone https://github.com/tns-research/llm-finance-framework.git
cd llm-finance-framework

# Install dependencies
make setup

# Run full development check
make check

# Test basic functionality
make run-dummy
```

## 🔧 Development Workflow

### Development Commands (Windows)
```bash
# Full development setup and check
dev.bat setup     # Install dependencies
dev.bat check     # Run lint + types + tests
dev.bat test      # Run tests only
dev.bat lint      # Run code quality checks
dev.bat clean     # Clean build artifacts
dev.bat version   # Version management (get/set/bump/tag)

# Examples
dev.bat version get                    # Show current version
dev.bat version bump patch            # Increment patch version
dev.bat version tag                   # Create git tag
```

### Development Commands (Linux/Mac)
```bash
# Full development setup and check
make setup        # Install dependencies
make check        # Run lint + types + tests
make test         # Run tests only
make lint         # Run code quality checks
make clean        # Clean build artifacts

# Version management
make version-get  # Show current version
make version-patch # Increment patch version
make version-tag  # Create git tag

# Quick development workflow
make dev          # Setup + check
make release      # Full release preparation
```

### Advanced Development Scripts
```bash
# Development workflow helper
python scripts/dev-workflow.py setup     # Setup environment
python scripts/dev-workflow.py check     # Run all quality checks
python scripts/dev-workflow.py test      # Run tests with coverage
python scripts/dev-workflow.py lint      # Code quality checks

# Validation scripts
python scripts/validate_core_math.py     # Validate mathematical functions
python scripts/validate_indicators.py    # Validate technical indicators
python scripts/generate_report.py        # Generate analysis reports
```

## 💻 Development Workflow

### Code Standards
- **Line Length**: 127 characters (configured in `pyproject.toml`)
- **Python Style**: Follow PEP 8 with Black formatting
- **Imports**: Sorted with isort (Black compatible profile)
- **Type Hints**: Use type annotations where beneficial, mypy compatible
- **Documentation**: Add docstrings to all public functions and classes
- **Error Handling**: Implement appropriate error handling and logging
- **Security**: Follow security best practices, no hardcoded secrets

### Code Quality Tools
```bash
# Automatic formatting
black src/ tests/                    # Format code
isort src/ tests/                    # Sort imports

# Linting and type checking
flake8 src/ tests/                   # Lint code (max 127 chars, extended ignores)
mypy src/                           # Type checking (relaxed for research code)

# All-in-one quality check
make lint                           # Runs all quality tools
# or: dev.bat lint
```

### Testing Requirements
- **Framework**: pytest with coverage reporting
- **Coverage**: Aim for >80% coverage on new code
- **Types**: Unit tests, integration tests, and validation tests
- **Mocking**: Use pytest fixtures for external dependencies
- **CI/CD**: All tests must pass before merge

```python
# Example test structure (see tests/ directory)
def test_llm_decision_parsing():
    """Test that LLM responses are correctly parsed."""
    # Implementation with proper assertions

def test_statistical_validation():
    """Test bootstrap significance calculation."""
    # Implementation with statistical validation
```

## 🤝 Collaboration Tools

### Git Workflow Helper
```bash
# Collaboration workflow management
python scripts/collaborate.py setup    # Initial collaboration setup
python scripts/collaborate.py sync     # Sync with upstream repository
python scripts/collaborate.py branch   # Create feature branch interactively
python scripts/collaborate.py pr       # Prepare for pull request
python scripts/collaborate.py status   # Show collaboration status
```

### Branch Naming Convention
- `feature/` - New functionality (e.g., `feature/add-new-baseline-strategy`)
- `research/` - Research experiments (e.g., `research/test-new-llm-model`)
- `fix/` - Bug fixes (e.g., `fix/resolve-memory-leak`)
- `docs/` - Documentation updates (e.g., `docs/update-contributing-guide`)

### Collaboration Workflow
1. **Fork & Setup**:
   ```bash
   python scripts/collaborate.py setup
   ```

2. **Sync with Upstream**:
   ```bash
   python scripts/collaborate.py sync
   ```

3. **Create Feature Branch**:
   ```bash
   python scripts/collaborate.py branch  # Interactive branch creation
   ```

4. **Develop & Test**:
   ```bash
   make check  # Run quality checks
   ```

5. **Prepare Pull Request**:
   ```bash
   python scripts/collaborate.py pr
   ```

## 📋 Pull Request Process

### Before Submitting
- [ ] **Quality Checks**: Run `make check` or `dev.bat check`
- [ ] **Tests Pass**: All existing tests pass with coverage >80%
- [ ] **New Tests**: Added tests for new functionality
- [ ] **Documentation**: Updated relevant documentation
- [ ] **Code Review**: Self-reviewed code for clarity and efficiency
- [ ] **Research Validation**: Experimental results statistically validated
- [ ] **Collaboration Tools**: Used `python scripts/collaborate.py pr`

### PR Template
```markdown
## Description
Brief description of changes and rationale

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Research enhancement (new experiments/methods)
- [ ] Breaking change (affects existing functionality)
- [ ] Documentation update
- [ ] Test addition/update

## Research Impact (if applicable)
- [ ] New experimental results
- [ ] Methodology improvements
- [ ] Statistical validation enhancements
- [ ] Performance improvements

## Testing & Validation
- [ ] All tests pass (`make test`)
- [ ] New tests added for new functionality
- [ ] Code quality checks pass (`make lint`)
- [ ] Manual testing performed
- [ ] Statistical validation completed (if research-related)
```

### Review Process
1. **Automated Validation**: Pre-merge checks run via GitHub Actions
   - Code quality (linting, formatting, types)
   - Test suite with coverage reporting
   - Build verification
   - Security scanning

2. **Peer Review**: At least one maintainer reviews changes
   - Code quality and architecture
   - Test coverage and correctness
   - Documentation completeness

3. **Research Review**: Research contributions reviewed by domain experts
   - Statistical validity
   - Experimental methodology
   - Result interpretation

4. **Integration**: Approved changes merged with rebase strategy
   - Linear history maintained
   - Release notes updated automatically

## 🔬 Research Collaboration Guidelines

### Experimental Standards
- **Reproducibility**: All experiments must be reproducible with documented parameters
- **Statistical Rigor**: Use appropriate statistical tests with confidence intervals
- **Data Integrity**: Validate data sources and preprocessing steps
- **Configuration Tracking**: Use the flexible configuration system for experiment parameters
- **Result Documentation**: Include methodology, assumptions, limitations, and code versions

### Sharing Research
- **Open Results**: Share both positive and negative findings
- **Complete Documentation**: Document experimental setup, prompts, configurations, and code versions
- **Statistical Reporting**: Include p-values, effect sizes, confidence intervals, and sample sizes
- **Replication Instructions**: Enable others to reproduce experiments exactly

### Model Testing Protocol
1. **Baseline Comparison**: Always compare against buy-and-hold, momentum, and mean-reversion strategies
2. **Statistical Validation**: Use bootstrap testing (≥1000 resamples) and out-of-sample testing
3. **Multiple Runs**: Report results across multiple random seeds for robustness
4. **Risk-Adjusted Metrics**: Include Sharpe ratio, maximum drawdown, and volatility measures
5. **Configuration Flexibility**: Test different `START_ROW` values for various market conditions

### Documentation Resources
- **[Experiment Design Guide](docs/experiment_design.md)**: Practical workflows for configuring experiments
- **[Configuration Reference](docs/configuration.md)**: Complete guide to configurable settings
- **[Methodology Guide](docs/methodology.md)**: Research methodology and validation procedures
- **[Collaboration Guide](docs/COLLABORATION_GUIDE.md)**: Advanced collaboration workflows

## 🐛 Issue Reporting

### Bug Reports
**Please include:**
- Python version and OS (`python --version` and OS details)
- Full error traceback (copy entire error message)
- Steps to reproduce (minimal example if possible)
- Expected vs actual behavior
- Configuration used (`python -c "import src.config as cfg; print(cfg.__dict__)"`)
- Framework version (`python scripts/version.py get`)

### Research Issues
**For research-related issues:**
- Experimental setup details (configuration file or key parameters)
- Statistical results and test outputs
- Data sources and date ranges used
- Reproducibility steps with exact commands
- Expected vs actual research outcomes

### Enhancement Requests
**For new features:**
- Use case description with concrete examples
- Proposed implementation approach
- Expected impact on research or development workflow
- Alternative approaches considered
- Links to relevant research papers or similar implementations

## 🎯 Recognition & Impact

### Contributor Recognition
- All contributors listed in repository contributors and CHANGELOG.md
- Research contributions acknowledged in publications and methodology documentation
- Significant contributions may be invited to co-authorship opportunities
- Recognition in framework development and maintenance

### Citation Guidelines
If your contribution leads to research outputs, please cite:
```bibtex
@misc{llm_finance_framework,
  title={Large Language Models in Financial Decision-Making: An Empirical Framework},
  author={LLM Finance Framework Contributors},
  year={2025},
  url={https://github.com/tns-research/llm-finance-framework},
  note={Version 0.3.1}
}
```

For specific research contributions, also consider citing relevant papers or methodologies implemented.

## 📞 Getting Help

### Primary Resources
- **[README.md](README.md)**: Quick start guide and framework overview
- **[Configuration Guide](docs/configuration.md)**: Complete configuration reference
- **[Experiment Design Guide](docs/experiment_design.md)**: Practical experiment setup workflows
- **[Methodology Guide](docs/methodology.md)**: Research methodology and validation

### Support Channels
- **Issues**: Use GitHub Issues for bugs, feature requests, and research questions
- **Discussions**: Use GitHub Discussions for research methodology questions and collaboration
- **Documentation**: Check inline code documentation and docstrings first
- **Validation Scripts**: Use `scripts/validate_*.py` for framework validation

## 📜 Code of Conduct & Ethics

### Research Integrity
- Report results honestly, including negative findings and failed experiments
- Acknowledge limitations, assumptions, and potential biases in all research
- Respect intellectual property, data privacy, and research ethics
- Maintain academic and professional standards in all contributions
- Ensure reproducible research with proper documentation

### Security & Privacy
- Never commit API keys, passwords, or sensitive credentials
- Use environment variables for all sensitive configuration
- Follow security best practices in code implementation
- Respect data privacy and avoid exposing sensitive information

### Community Standards
- Be respectful, inclusive, and collaborative
- Provide constructive feedback on all contributions
- Share knowledge openly and help newcomers get started
- Respect diverse perspectives, backgrounds, and expertise levels
- Maintain professional discourse in research discussions

---

**Thank you for contributing to the advancement of AI in finance research! 🚀📊**

*This project combines rigorous academic research standards with modern open-source development practices to advance understanding of AI in financial decision-making.*
