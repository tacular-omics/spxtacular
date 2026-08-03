default: lint format check test

# Install dependencies
install:
    uv sync

install-dev:
    uv sync --dev

# Run linting checks
lint:
    uv run ruff check src tests

# Format code
format:
    uv run ruff check --select I --fix src tests
    uv run ruff format src tests

# Verify formatting without rewriting files (used by CI)
fmt-check:
    uv run ruff format --check src tests

# Run type checking
check:
    uv run ty check src tests

# Run tests
test:
    uv run pytest tests

# Run tests once, emitting both coverage XML and JUnit XML for Codecov
test-cov:
    uv run pytest tests --cov=src/spxtacular --cov-report=xml --junit-xml=junit.xml

# Build and serve docs
docs:
    uv run mkdocs serve --dev-addr=localhost:8003

# Build docs to site/
docs-build:
    uv run mkdocs build
