.PHONY: sync lock test lint typecheck run api docker-build docker-run docker-test docker-lint docker-typecheck clean

sync:
	uv sync --dev

lock:
	uv lock

test:
	uv run pytest -q

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

run:
	uv run robot-sort run --headless --objects 5 --seed 42 --output outputs/run

api:
	uv run uvicorn robot_sorting.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-run:
	docker compose run --rm sim

docker-test:
	docker compose run --rm test

docker-lint:
	docker compose run --rm lint

docker-typecheck:
	docker compose run --rm typecheck

clean:
	rm -rf outputs/run .pytest_cache .ruff_cache .mypy_cache
