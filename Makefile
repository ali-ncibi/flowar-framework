.PHONY: help install run demo lint format typecheck test check docker-up docker-down docker-demo

help:
	@printf '%s\n' \
		'FlowAR commands:' \
		'  make install       Install/sync project dependencies' \
		'  make run           Run the Streamlit app' \
		'  make demo          Run the end-to-end demo locally' \
		'  make lint          Lint and apply Ruff fixes' \
		'  make format        Format Python files with Ruff' \
		'  make typecheck     Run mypy and pyright' \
		'  make test          Run the pytest suite' \
		'  make check         Run formatting, lint, type, and test checks' \
		'  make docker-up     Build and start the Docker app' \
		'  make docker-down   Stop and remove the Docker app' \
		'  make docker-demo   Run the demo in Docker'

install:
	uv sync

run:
	uv run streamlit run frontend/main.py

demo:
	uv run python scripts/demo.py

lint:
	uv run ruff check --fix

format:
	uv run ruff format

typecheck:
	uv run mypy
	uv run pyright

test:
	uv run pytest

check: lint format typecheck test

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-demo:
	docker compose --profile demo run --rm demo
