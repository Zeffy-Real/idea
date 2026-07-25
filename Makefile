# =============================================================================
# Cognita RAG - Development Makefile
# =============================================================================

.PHONY: help install dev test test-unit test-integration lint format typecheck \
        docker-up docker-down docker-logs docker-build clean serve cli init

PYTHON ?= python
PIP ?= pip

help: ## Show this help message
	@echo "Cognita RAG - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in editable mode
	$(PIP) install -e ".[dev]"

dev: ## Install in development mode with all extras
	$(PIP) install -e ".[dev,tracing]"

test: ## Run all tests
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short --cov=cognita --cov-report=term

test-integration: ## Run integration tests (requires Qdrant)
	pytest tests/integration/ -v --tb=short -m "not slow"

lint: ## Run linter
	ruff check cognita/ tests/

format: ## Format code
	ruff format cognita/ tests/
	ruff check --fix cognita/ tests/

typecheck: ## Run type checker
	mypy cognita/ --ignore-missing-imports

docker-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## Tail Docker logs
	docker-compose logs -f

docker-build: ## Build Docker image
	docker build -t cognita-rag:latest .

docker-up-monitoring: ## Start with monitoring stack (Prometheus)
	docker-compose --profile monitoring up -d

serve: ## Start the API server locally
	$(PYTHON) -m uvicorn cognita.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

cli: ## Start the CLI
	$(PYTHON) -m cognita.cli.main

init: ## Initialize the system (create vectorstore collection)
	$(PYTHON) -c "import asyncio; from cognita.ingestion.pipeline import IngestionPipeline; asyncio.run(IngestionPipeline().initialize()); print('System initialized successfully')"

clean: ## Clean up build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up build artifacts"
