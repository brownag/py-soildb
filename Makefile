.PHONY: help install test lint lint-fix format docs docs-validate clean build all examples-validate examples-test
.DEFAULT_GOAL := help
RUFF_ARGS ?=

all: clean install lint-fix test build docs examples-validate ## Run full build pipeline (clean, install, lint-fix, test, build, docs, validate examples)

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
	ruff check $(RUFF_ARGS) src/ tests/ docs/examples/
	mypy src/soildb --ignore-missing-imports

lint-fix: ## Run linting checks and auto-fix issues (use RUFF_ARGS="--unsafe-fixes" for unsafe fixes)
	ruff check --fix $(RUFF_ARGS) src/ tests/ docs/examples/
	ruff format src/ tests/ docs/examples/
	mypy src/soildb --ignore-missing-imports

format: ## Format code
	ruff format $(RUFF_ARGS) src/ tests/ docs/examples/

format-check: ## Check code formatting without making changes
	ruff format --check $(RUFF_ARGS) src/ tests/ docs/examples/

security: ## Run security checks
	bandit -r src/
	safety check

docs: docs-validate ## Build documentation with Quarto (includes validation)
	@echo "Extracting version from pyproject.toml..."
	@SOILDB_VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Version: $$SOILDB_VERSION"; \
	sed "s/\$${SOILDB_VERSION}/$$SOILDB_VERSION/g" docs/_quarto.yml > docs/_quarto.yml.tmp && mv docs/_quarto.yml.tmp docs/_quarto.yml; \
	quartodoc build --config docs/_quarto.yml; \
	sed -i "s/\"version\": \"0.0.9999\"/\"version\": \"$$SOILDB_VERSION\"/g" docs/objects.json; \
	quarto render docs

docs-serve: docs-validate ## Serve documentation with Quarto and watch for changes (includes validation)
	@echo "Extracting version from pyproject.toml..."
	@SOILDB_VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Version: $$SOILDB_VERSION"; \
	sed "s/\$${SOILDB_VERSION}/$$SOILDB_VERSION/g" docs/_quarto.yml > docs/_quarto.yml.tmp && mv docs/_quarto.yml.tmp docs/_quarto.yml; \
	quartodoc build --config docs/_quarto.yml; \
	sed -i "s/\"version\": \"0.0.9999\"/\"version\": \"$$SOILDB_VERSION\"/g" docs/objects.json; \
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

examples-validate: ## Validate all example scripts (import check, not runtime)
	python scripts/validate_examples.py

examples-test: ## Run example scripts (requires network access to SDA/AWDB)
	@echo "Running example scripts..."
	@echo "  Testing 04_schema.py (SDA schema inspection)..."
	@timeout 60 python docs/examples/04_schema.py > /dev/null && echo "  ✓ 04_schema.py" || echo "  ✗ 04_schema.py"
	@echo "  Testing 05_awdb.py (AWDB with throttling)..."
	@timeout 120 python docs/examples/05_awdb.py > /dev/null && echo "  ✓ 05_awdb.py" || echo "  ✗ 05_awdb.py (may timeout due to AWDB rate limits)"
	@echo ""
	@echo "Note: Other examples require network access and may timeout due to SDA/AWDB limits."
	@echo "      Run individually with: python docs/examples/0X_*.py"

docs-validate: ## Validate that doc code blocks are syntactically correct
	@echo "Validating documentation code examples..."
	@for qmd in docs/*.qmd docs/examples/*.qmd; do \
		[ ! -f "$$qmd" ] && continue; \
		echo "  Checking $$qmd..."; \
		if grep -qE "from.*awdb_integration|import.*get_component_water_properties|import.*get_scan_soil_moisture|from.*import.*auto_schema|\.auto_schema\(" "$$qmd"; then \
			echo "  ✗ $$qmd contains phantom functions"; \
			grep -n "awdb_integration\|get_component_water_properties\|get_scan_soil_moisture" "$$qmd" | head -3; \
			exit 1; \
		fi; \
		echo "  ✓ $$qmd"; \
	done
	@echo "✓ All documentation validated"
