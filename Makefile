.PHONY: setup format lint run-app dev-up dev-up-gpu dev-down prod-up prod-up-gpu prod-down offline-up offline-down config-help config-generate config-validate config-summary config-env config-docker config-services

# ==============================================================================
# Dependency Management & Code Quality
# ==============================================================================

setup:
	@echo ">> Installing dependencies with Poetry..."
	@poetry install

format:
	@echo ">> Formatting code with ruff and isort..."
	@poetry run ruff format .
	@poetry run isort .

lint:
	@echo ">> Linting code with ruff..."
	@poetry run ruff check .

# ==============================================================================
# Configuration Management
# ==============================================================================

config-help:
	@echo ">> Configuration Management Commands:"
	@echo "  make config-generate    - Generate all configuration files"
	@echo "  make config-validate    - Validate current configuration"
	@echo "  make config-summary     - Show configuration summary"
	@echo "  make config-env         - Instructions for setting up .env file"
	@echo "  make config-services    - Check if services are healthy"

config-generate:
	@echo ">> Rendering LiteLLM configuration from template..."
	@poetry run python utility/config_utils.py render-config utility/litellm.config.template.yaml utility/litellm.config.yaml

config-validate:
	@echo ">> Validating configuration..."
	@poetry run python utility/config_utils.py validate

config-summary:
	@echo ">> Configuration summary..."
	@poetry run python utility/config_utils.py summary

config-env:
	@echo ">> Please copy env.example to .env and customize it"
	@echo ">> cp env.example .env"

config-docker:
	@echo ">> Docker uses .env file directly. Please ensure .env is configured."

config-services:
	@echo ">> Checking service health..."
	@poetry run python utility/config_utils.py check-services

# ==============================================================================
# Application Execution
# ==============================================================================

run-app:
	@echo ">> Running Streamlit app (expects LLM_PROVIDER=openai)..."
	@poetry run streamlit run main.py

# Development mode (with source code mounted for hot reloading)
dev-up:
	@echo ">> Starting Docker services in development mode..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml up -d --build

dev-up-gpu:
	@echo ">> Starting Docker services in development mode with GPU support..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.gpu.yml up -d --build

dev-down:
	@echo ">> Stopping Docker services..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml down

# Production mode (optimized, no source code mount)
prod-up:
	@echo ">> Starting Docker services in production mode..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml up -d --build

prod-up-gpu:
	@echo ">> Starting Docker services in production mode with GPU support..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml -f docker/docker-compose.gpu.yml up -d --build

prod-down:
	@echo ">> Stopping Docker services..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml down

# Development utilities
dev-logs:
	@echo ">> Showing logs for all services..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml logs -f

dev-shell:
	@echo ">> Opening shell in app container..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml exec app /bin/bash

dev-restart:
	@echo ">> Restarting app service only..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml restart app

# Production utilities
prod-logs:
	@echo ">> Showing logs for all services..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml logs -f

prod-shell:
	@echo ">> Opening shell in app container..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml exec app /bin/bash

prod-restart:
	@echo ">> Restarting app service only..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml restart app

# Docker utilities
docker-clean:
	@echo ">> Cleaning up Docker resources..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml down -v --remove-orphans
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml down -v --remove-orphans
	@docker system prune -f

docker-status:
	@echo ">> Docker services status..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml ps 2>/dev/null || echo "Development services not running"
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.prod.yml ps 2>/dev/null || echo "Production services not running"

# Model management
download-model:
	@echo ">> Downloading LLM model (first time only)..."
	@docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml exec ollama sh -lc 'ollama pull "$${LOCAL_LLM_MODEL:-llama3.2:3b}" && ollama list'

# Legacy aliases for backward compatibility
offline-up: dev-up
offline-down: dev-down
