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

# Build and serve docs
docs:  
    uv run mkdocs serve --dev-addr=localhost:8003

# Build docs to site/
docs-build:  
    uv run mkdocs build