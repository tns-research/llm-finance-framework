# LLM Finance Framework - Development Makefile

.PHONY: help setup test lint types check clean build release

# Default target
help: ## Show this help message
	@echo "LLM Finance Framework - Development Commands"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup development environment
setup: ## Install development dependencies
	python -m pip install -e .[dev]

# Run tests
test: ## Run test suite with coverage
	python -m pytest tests/ -v --cov=src/ --cov-report=term-missing

# Run linting
lint: ## Run code quality checks (flake8, black, isort)
	flake8 src/ tests/ --max-line-length=127 --extend-ignore=E203,W503
	black --check --diff src/ tests/
	isort --check-only --diff src/ tests/

# Run type checking
types: ## Run mypy type checking
	mypy src/ --ignore-missing-imports

# Run all checks
check: lint types test ## Run full development check suite

# Clean build artifacts
clean: ## Clean build artifacts and cache files
	rm -rf build/ dist/ *.egg-info/
	rm -rf .coverage .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Build package
build: clean ## Build distribution packages
	python -m build

# Release workflow
release: check build ## Prepare for release (run checks, build package)
	@echo "Ready for release!"
	@echo "Current version: $$(python scripts/version.py get)"
	@echo "Next steps:"
	@echo "1. Update CHANGELOG.md"
	@echo "2. python scripts/version.py bump [patch|minor|major]"
	@echo "3. python scripts/version.py tag"
	@echo "4. git push origin main --tags"
	@echo "5. Create GitHub release"

# Quick development workflow
dev: setup check ## Quick setup and check for development

# Version management
version-get: ## Get current version
	python scripts/version.py get

version-patch: ## Bump patch version
	python scripts/version.py bump patch

version-minor: ## Bump minor version
	python scripts/version.py bump minor

version-major: ## Bump major version
	python scripts/version.py bump major

version-tag: ## Create git tag for current version
	python scripts/version.py tag

# Run basic functionality test
run-dummy: ## Run framework with dummy model
	python -m src.main

# Development server (if applicable)
# serve: ## Start development server
# 	python -m src.main --dev
