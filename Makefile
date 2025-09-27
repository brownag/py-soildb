.PHONY: help install test lint format docs clean build
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install the package in development mode
	pip install -e ".[dev,dataframes,spatial,docs,security]"

install-prod: ## Install only production dependencies
	pip install -e .

test: ## Run the test suite
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=soildb --cov-report=html --cov-report=term-missing

test-integration: ## Run integration tests (requires network)
	pytest tests/test_integration.py -v

lint: ## Run linting checks
	ruff check src/ tests/ examples/
	mypy src/soildb --ignore-missing-imports

format: ## Format code
	ruff format src/ tests/ examples/

format-check: ## Check code formatting without making changes
	ruff format --check src/ tests/ examples/

security: ## Run security checks
	bandit -r src/
	safety check

docs: ## Build documentation with Quarto
	quartodoc build --config docs/_quarto.yml
	quarto render docs

docs-serve: ## Serve documentation with Quarto and watch for changes
	quartodoc build --config docs/_quarto.yml
	quarto preview docs

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

examples: ## Run example scripts (basic functionality test)
	@echo "Testing basic examples..."
	@python -c "import soildb; print('✓ Package imports successfully')"
	@python examples/basic_examples.py
	@echo "✓ All examples completed successfully"