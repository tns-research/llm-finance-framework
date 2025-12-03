# 🤝 Collaboration Guide

This guide explains how to collaborate effectively on the LLM Finance Framework using our established tools and processes.

## 📋 **Quick Start for Collaborators**

### **1. Repository Access**
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/tns-research/llm-finance-framework.git
cd llm-finance-framework

# Add upstream remote
git remote add upstream https://github.com/tns-research/llm-finance-framework.git
```

### **2. Development Setup**
```bash
# Modern installation (recommended) - includes all dependencies + dev tools
pip install -e .[dev]

# Traditional installation
pip install -r requirements.txt
pip install pytest black flake8 mypy isort  # Dev tools separately

# Windows users can also use:
dev setup

# Linux/Mac users can also use:
make setup
```

### **3. Create Feature Branch**
```bash
# For new features
git checkout -b feature/your-feature-name

# For research experiments
git checkout -b research/llm-model-comparison

# For bug fixes
git checkout -b fix/issue-number-description

# For documentation
git checkout -b docs/update-contributing-guide
```

## 🛠️ Available Development Tools

### Automated Quality Checks
```bash
# Full development check suite
python scripts/dev-workflow.py check
# Includes: linting, testing, type checking, basic functionality

# Individual checks
python scripts/dev-workflow.py lint   # Code style (flake8, black, isort)
python scripts/dev-workflow.py test   # Unit tests with coverage
python scripts/dev-workflow.py types  # Type checking (mypy)
```

### Version Management
```bash
# Check current version
python scripts/version.py get

# Bump versions (semantic versioning)
python scripts/version.py bump patch  # 0.1.0 → 0.1.1 (bug fixes)
python scripts/version.py bump minor  # 0.1.1 → 0.2.0 (new features)
python scripts/version.py bump major  # 0.2.0 → 1.0.0 (breaking changes)

# Create git tag for releases
python scripts/version.py tag
```

### Collaboration Helpers
```bash
# Repository status overview
python scripts/collaborate.py status

# Interactive branch creation
python scripts/collaborate.py branch

# Prepare for pull request submission
python scripts/collaborate.py pr
```

### Windows-Specific Tools
```batch
dev setup     # Install all dependencies
dev check     # Full quality check suite
dev test      # Run tests only
dev lint      # Code style checks
dev clean     # Clean build artifacts
dev collab    # Collaboration tools menu
```

### Linux/Mac Tools
```bash
make setup    # Install all dependencies
make check    # Full quality check suite
make test     # Run tests only
make lint     # Code style checks
make clean    # Clean build artifacts
```

## 🤖 Automated Quality Assurance

### GitHub Actions CI/CD
Every pull request and push to main automatically triggers our CI pipeline:

#### Quality Gates
- ✅ **Dependency Installation**: `pip install -e .[dev]`
- ✅ **Code Linting**: flake8 (PEP 8), black (formatting), isort (imports)
- ✅ **Type Checking**: mypy validation of type hints
- ✅ **Test Execution**: pytest with coverage reporting
- ✅ **Import Validation**: All dependencies properly declared
- ✅ **Basic Functionality**: Core imports and basic operations tested

#### CI Workflow Details
```yaml
# Runs on every push/PR to main
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

# Checks performed:
# 1. Python 3.8 environment setup
# 2. Dependency caching for faster builds
# 3. Code quality validation
# 4. Test execution with coverage
# 5. Automated feedback on PRs
```

#### Interpreting CI Results
- **✅ Green Checkmark**: All quality gates passed
- **❌ Red X**: Issues found - check the "Details" link for specifics
- **🟡 Yellow Dot**: CI in progress

#### Common CI Issues & Solutions
- **Import Errors**: Check `pyproject.toml` dependencies
- **Linting Failures**: Run `python scripts/dev-workflow.py lint` locally
- **Test Failures**: Run `python scripts/dev-workflow.py test` locally
- **Type Errors**: Run `python scripts/dev-workflow.py types` locally

### Automated Dependency Management
- **Dependabot**: Automatically creates PRs for dependency updates
- **Security Updates**: Priority handling of security vulnerabilities
- **Version Constraints**: Smart updating that avoids breaking changes

## 🔄 **Collaboration Workflow**

### **Daily Development Cycle**
```bash
# 1. Sync with upstream
git fetch upstream
git rebase upstream/main

# 2. Work on your feature
# Make changes to code...

# 3. Run quality checks before committing
python scripts/dev-workflow.py check
# Or: dev check (Windows) / make check (Linux/Mac)

# 4. Fix any issues found by checks
# Run individual checks if needed:
python scripts/dev-workflow.py lint   # Fix code style
python scripts/dev-workflow.py test   # Fix failing tests

# 5. Commit your work with conventional format
git add .
git commit -m "feat: Add new baseline strategy

- Implement momentum-based strategy
- Add configuration options
- Include comprehensive tests"

# 6. Push to your fork
git push origin feature/your-feature-name
```

### **Pull Request Process**
```bash
# 1. Ensure your branch is up-to-date
git fetch upstream
git rebase upstream/main

# 2. Run comprehensive final checks
python scripts/dev-workflow.py check
# This ensures all quality gates pass before PR

# 3. Use collaboration helper for final prep
python scripts/collaborate.py pr
# This provides guidance on PR readiness

# 4. Push final version (force-with-lease for safety)
git push origin feature/your-feature-name --force-with-lease

# 5. Create PR on GitHub
# - Use the PR template (auto-filled)
# - Fill in all required sections
# - Request reviews from appropriate maintainers
# - CI will automatically run quality checks
```

## 🎯 **Branch Types & Naming**

| Branch Type | Naming Pattern | Purpose | Merge Strategy |
|-------------|----------------|---------|----------------|
| **Feature** | `feature/description` | New functionality | Squash merge |
| **Research** | `research/experiment-name` | Research experiments | Merge commit |
| **Fix** | `fix/issue-number-description` | Bug fixes | Squash merge |
| **Docs** | `docs/update-section` | Documentation | Squash merge |
| **Hotfix** | `hotfix/critical-bug` | Critical fixes | Merge commit |

### **Examples:**
```
feature/add-technical-indicators
research/claude-model-comparison
fix/123-incorrect-sharpe-calculation
docs/update-api-reference
hotfix/critical-calculation-error
```

## 🔍 **Code Review Process**

### **For Contributors:**
1. **Self-Review First**: Run `dev check` and review your own code
2. **Fill PR Template**: Complete all sections thoroughly
3. **Add Tests**: Include tests for new functionality
4. **Update Docs**: Update README/docs if needed
5. **Request Review**: Tag appropriate reviewers

### **For Reviewers:**
1. **Check CI Status**: Ensure all checks pass
2. **Review Code**: Focus on logic, style, and correctness
3. **Test Changes**: Run experiments if applicable
4. **Verify Documentation**: Check docs are updated
5. **Approve or Request Changes**: Be specific with feedback

## 🧪 **Testing Strategy**

### **Unit Tests**
```bash
# Run all tests
dev test

# Run specific test
python -m pytest tests/test_calibration.py -v

# Run with coverage
python -m pytest tests/ --cov=src/ --cov-report=html
```

### **Integration Tests**
```bash
# Test full pipeline (dummy model)
python -m src.main

# Test with specific configuration
TEST_MODE=True python -m src.main
```

### **Research Validation**
- Always compare against baseline strategies
- Use bootstrap testing for significance
- Document experimental parameters
- Share reproducible results

## 📊 **Version Management in Teams**

### **Version Bumping Guidelines**
- **Patch** (`0.1.0` → `0.1.1`): Bug fixes, documentation
- **Minor** (`0.1.0` → `0.2.0`): New features, backward compatible
- **Major** (`0.1.0` → `1.0.0`): Breaking changes, API changes

### **Release Process**
```bash
# Team lead coordinates releases
dev version bump minor          # Bump version
dev version tag                # Create tag
git push upstream main --tags  # Push to main repo

# Create GitHub release with changelog
```

## 🐛 **Issue Management**

### **Creating Issues**
Use appropriate templates:
- 🐛 **Bug Report**: Clear reproduction steps
- ✨ **Feature Request**: Detailed requirements
- 🔬 **Research Question**: Methodology guidance

### **Issue Lifecycle**
1. **Triage**: Team reviews and labels issues
2. **Assignment**: Team members claim issues
3. **Development**: Work in feature branches
4. **Review**: Pull requests for changes
5. **Close**: When issue is resolved

## 📚 **Documentation Standards**

### **Code Documentation**
```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio for a series of returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy
    risk_free_rate : float, default 0.02
        Annual risk-free rate (2%)

    Returns
    -------
    float
        Annualized Sharpe ratio

    Examples
    --------
    >>> returns = pd.Series([0.01, -0.005, 0.008])
    >>> sharpe = calculate_sharpe_ratio(returns)
    >>> print(f"Sharpe ratio: {sharpe:.3f}")
    Sharpe ratio: 1.234
    """
    # Implementation...
```

### **Research Documentation**
- Document experimental setup
- Include statistical methodology
- Share reproducible code
- Reference academic sources

## 🔒 **Security & API Keys**

### **API Key Management**
```bash
# Never commit API keys
# Use environment variables
export OPENROUTER_API_KEY="your_key_here"

# For team development, use .env files
# Add .env to .gitignore
```

### **Security Reviews**
- Review code for hardcoded secrets
- Validate input sanitization
- Check for dependency vulnerabilities
- Regular security audits

## 🎯 **Communication Channels**

### **GitHub**
- **Issues**: Bug reports, feature requests, research questions
- **Discussions**: Research methodology, design decisions
- **Pull Requests**: Code review and integration
- **Projects**: Track milestones and releases

### **Meeting Cadence**
- **Daily Standups**: Quick progress updates (optional)
- **Weekly Reviews**: Code review session
- **Monthly Planning**: Feature planning and priorities

## 📈 **Performance & Quality Metrics**

### **Code Quality**
- **Test Coverage**: Aim for >80% coverage
- **Code Style**: PEP 8 compliance
- **Documentation**: All public APIs documented
- **Type Hints**: Use type annotations

### **Research Quality**
- **Statistical Rigor**: Bootstrap validation
- **Reproducibility**: Document all parameters
- **Peer Review**: Research contributions reviewed
- **Open Results**: Share both successes and failures

## 🚨 **Conflict Resolution**

### **Merge Conflicts**
```bash
# When conflicts occur
git fetch upstream
git rebase upstream/main
# Resolve conflicts in editor
git add resolved_files
git rebase --continue
```

### **Design Disagreements**
1. Document both approaches
2. Create proof-of-concept implementations
3. Data-driven decision making
4. Consensus or team lead decision

## 🎉 **Recognition & Attribution**

### **Contribution Recognition**
- All contributors listed in repository
- Research contributions acknowledged in papers
- Significant contributions may lead to co-authorship
- Regular shoutouts in team meetings

### **Authorship Guidelines**
- Substantial intellectual contributions
- Code contributions > 10% of feature
- Research methodology development
- Documentation of novel approaches

---

## 📞 **Getting Help**

- **Documentation First**: Check docs/ (especially experiment_design.md for experiments), and README.md
- **Search Issues**: Look for similar problems
- **Ask Questions**: Use GitHub Discussions for general questions
- **Team Communication**: Reach out to team members directly

**Remember**: This is a research project. Focus on scientific rigor, collaboration, and advancing our understanding of AI in finance! 🔬🤖📊
