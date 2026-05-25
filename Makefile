.DEFAULT_GOAL := help

SHELL := /bin/bash

.PHONY: help install install-backend install-frontend validate-config run-api run-frontend test stop

help: ## Show available manual-install commands.
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-18s %s\n", $$1, $$2}'

install: install-backend install-frontend ## Install backend and frontend dependencies.

install-backend: ## Create/use the virtualenv and install Python dependencies.
	@scripts/dev/install.sh --backend-only

install-frontend: ## Install frontend pnpm dependencies.
	@scripts/dev/install.sh --frontend-only

validate-config: ## Validate runtime configuration without starting the API.
	@scripts/dev/validate-config.sh

run-api: ## Validate config and run the FastAPI backend.
	@scripts/dev/run-api.sh

run-frontend: ## Run the Vite frontend dev server.
	@scripts/dev/run-frontend.sh

test: ## Run backend and frontend tests.
	@scripts/dev/validate-config.sh
	@. scripts/dev/venv.sh && activate_venv && pytest -q
	@cd frontend && pnpm test

stop: ## Stop processes listening on the configured API and frontend ports.
	@scripts/dev/stop.sh
