default: lint format check test

# Install dependencies
install:
    uv sync

install-dev:
    uv sync --dev

# Run linting checks
lint:
    uv run ruff check src

# Format code
format:
	uv run ruff check --select I --fix src tests
	uv run ruff format src tests

# Run type checking
check:
    uv run ty check src

# Run tests
test:
    uv run pytest tests

# Run tests with coverage (XML + JUnit XML for Codecov)
test-cov:
    uv run pytest tests --cov=src/spxtacular --cov-report=xml

# Run tests with JUnit XML report for Codecov test results upload
codecov-tests:
    uv run pytest tests --junit-xml=junit.xml

# Build and serve docs
docs:  
    uv run mkdocs serve --dev-addr=localhost:8003

# Build docs to site/
docs-build:  
    uv run mkdocs build