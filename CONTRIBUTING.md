# Contributing

Guidelines for contributing to soildb.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/py-soildb.git
   cd py-soildb
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev,dataframes,spatial]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=soildb --cov-report=html
```

Run only unit tests (skip integration tests):
```bash
pytest -m "not integration"
```

## Code Style

We use several tools to maintain code quality:

- **ruff** for code formatting and linting (combines Black, flake8, and more)
- **mypy** for type checking

Run all checks:
```bash
ruff check src/ tests/ examples/
ruff format src/ tests/ examples/
mypy src/
```

Or use the Makefile shortcuts:
```bash
make lint    # Run ruff check and mypy
make format  # Format code with ruff
```

## Making Changes

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests to cover them.

3. Ensure all tests pass and code style checks pass.

4. Commit your changes with a clear commit message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

5. Push your changes and create a pull request.

## Pull Request Guidelines

- Include a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style

## Reporting Issues

When reporting issues, please include:

- Python version
- soildb version
- Operating system
- Complete error traceback
- Minimal code example that reproduces the issue

## Integration Tests

Some tests require network access to query the actual SDA service. These are marked with `@pytest.mark.integration` and are skipped by default.

To run integration tests:
```bash
pytest -m integration
```

**Note:** Integration tests may fail if the SDA service is under maintenance.

## Documentation

Documentation is built using Sphinx. To build documentation locally:

```bash
cd docs/
make html
```

The built documentation will be in `docs/_build/html/`.

## Release Process

1. Update version in `pyproject.toml` and `src/soildb/__init__.py`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag v0.x.x`
4. Push tag: `git push origin v0.x.x`
5. GitHub Actions will automatically build and publish to PyPI

## License

By contributing to soildb, you agree that your contributions will be licensed under the MIT License.