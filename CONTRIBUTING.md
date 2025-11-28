# Contributing to LLM Finance Experiment

We welcome contributions from researchers, developers, and financial practitioners! This project aims to advance understanding of AI in financial decision-making through rigorous empirical research.

## 🎯 Ways to Contribute

### 🔬 Research Contributions
- **New LLM Models**: Test additional language models (Claude, Gemini, etc.)
- **Strategy Development**: Implement new baseline strategies or LLM prompting techniques
- **Statistical Methods**: Enhance validation frameworks or develop new metrics
- **Market Analysis**: Test on different markets, time periods, or asset classes
- **Behavioral Analysis**: Develop new methods for detecting AI trading biases

### 💻 Code Contributions
- **Bug Fixes**: Identify and fix issues in the codebase
- **Feature Implementation**: Add new analysis tools or visualization methods
- **Performance Optimization**: Improve execution speed or memory usage
- **Documentation**: Enhance code documentation and examples

### 🧪 Testing & Validation
- **Test Coverage**: Add unit tests and integration tests
- **Result Validation**: Verify experimental results and statistical analyses
- **Cross-Platform Testing**: Ensure compatibility across different environments

### 📚 Documentation
- **README Improvements**: Enhance setup instructions and examples
- **Research Documentation**: Document methodologies and findings
- **Tutorial Creation**: Develop guides for specific use cases

## 🚀 Getting Started

### Prerequisites
```bash
# Required
pip install pandas numpy scipy matplotlib requests

# Optional (for development)
pip install pytest black flake8 mypy
```

### OpenRouter Setup
- **Get API Key**: Sign up at [OpenRouter.ai](https://openrouter.ai/)
- **Documentation**: [OpenRouter API Docs](https://openrouter.ai/docs)
- **Model List**: [Available Models](https://openrouter.ai/docs/models)

### Development Setup
1. **Fork and Clone**:
   ```bash
   git clone https://github.com/tns-research/llm-finance-framework.git
   cd llm-finance-experiment
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run Tests**:
   ```bash
   python -m pytest tests/  # If tests exist
   ```

4. **Test Basic Functionality**:
   ```bash
   python -m src.main  # Should run with dummy model
   ```

## 🔬 Research Collaboration Guidelines

### Experimental Standards
- **Reproducibility**: All experiments should be reproducible with documented parameters
- **Statistical Rigor**: Use appropriate statistical tests and report confidence intervals
- **Data Integrity**: Validate data sources and preprocessing steps
- **Result Documentation**: Include methodology, assumptions, and limitations

### Sharing Research
- **Open Results**: Share both positive and negative findings
- **Methodology Details**: Document experimental setup, prompts, and configurations
- **Statistical Reporting**: Include p-values, effect sizes, and confidence intervals
- **Replication Instructions**: Enable others to reproduce your experiments

### Model Testing Protocol
1. **Baseline Comparison**: Always compare against buy-and-hold and momentum strategies
2. **Statistical Validation**: Use bootstrap testing with at least 1000 resamples
3. **Multiple Runs**: Report results across multiple random seeds
4. **Hold-out Testing**: Validate on unseen data periods

## 💻 Development Workflow

### Branching Strategy
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or for research
git checkout -b research/llm-model-comparison

# Or for bug fixes
git checkout -b fix/issue-description
```

### Code Standards
- **Python Style**: Follow PEP 8 guidelines
- **Type Hints**: Use type annotations for function parameters and returns
- **Documentation**: Add docstrings to all functions and classes
- **Error Handling**: Implement appropriate error handling and logging

### Testing Requirements
```python
# Example test structure
def test_llm_decision_parsing():
    """Test that LLM responses are correctly parsed."""
    # Test implementation
    pass

def test_statistical_significance():
    """Test bootstrap significance calculation."""
    # Test implementation
    pass
```

## 📋 Pull Request Process

### Before Submitting
- [ ] **Tests Pass**: All existing tests pass
- [ ] **New Tests**: Added tests for new functionality
- [ ] **Documentation**: Updated relevant documentation
- [ ] **Code Review**: Self-reviewed code for clarity and efficiency
- [ ] **Research Validation**: Experimental results are statistically sound

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Research enhancement
- [ ] Documentation update
- [ ] Test addition

## Research Impact (if applicable)
- [ ] New experimental results
- [ ] Methodology improvements
- [ ] Statistical validation enhancements

## Testing
- [ ] Existing tests pass
- [ ] New tests added
- [ ] Manual testing performed
```

### Review Process
1. **Automated Checks**: CI/CD pipeline validates code quality
2. **Peer Review**: At least one maintainer reviews changes
3. **Research Review**: Research contributions reviewed by domain experts
4. **Integration**: Approved changes merged with rebase strategy

## 🐛 Issue Reporting

### Bug Reports
**Please include:**
- Python version and OS
- Full error traceback
- Steps to reproduce
- Expected vs actual behavior
- Configuration used

### Research Issues
**For research-related issues:**
- Experimental setup details
- Statistical results
- Data sources used
- Reproducibility steps

### Enhancement Requests
**For new features:**
- Use case description
- Proposed implementation
- Expected impact
- Alternative approaches considered

## 🎯 Recognition

### Contributor Recognition
- All contributors listed in repository contributors
- Research contributions acknowledged in publications
- Significant contributions may be invited to co-authorship

### Citation Guidelines
If your contribution leads to research outputs, please cite:
```bibtex
@misc{llm_finance_experiment,
  title={Large Language Models in Financial Decision-Making},
  author={Contributors},
  year={2025},
  url={https://github.com/tns-research/llm-finance-framework}
}
```

## 📞 Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for research questions
- **Documentation**: Check README.md and inline code documentation first

## 📜 Code of Conduct

### Research Integrity
- Report results honestly, including negative findings
- Acknowledge limitations and potential biases
- Respect intellectual property and data privacy
- Maintain academic and professional standards

### Community Standards
- Be respectful and inclusive
- Provide constructive feedback
- Collaborate openly and transparently
- Respect diverse perspectives and backgrounds

---

**Thank you for contributing to the advancement of AI in finance research! 🚀📊**

*This project follows academic research standards while embracing open-source development practices.*
