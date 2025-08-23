.PHONY: install dev-install format lint type-check test clean run

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev-install:
	pip install -e ".[dev]"

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Lint code
lint:
	flake8 src/ tests/

# Type checking
type-check:
	mypy src/

# Run tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/

# Run the application
run:
	python main.py

# Run in development mode
dev:
	ENVIRONMENT=development python main.py

# Setup development environment
setup-dev: dev-install
	cp .env.example .env
	mkdir -p data/parquet data/backups models logs

# Check code quality
check: format lint type-check test