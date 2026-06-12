SERVICE := app
DOMAIN ?= industrial
STAGE ?= full
ARGS ?=

.PHONY: up up-d down build logs ps shell pipeline restart

up:
	docker compose up --build

up-d:
	docker compose up --build -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f $(SERVICE)

ps:
	docker compose ps

shell:
	docker compose exec $(SERVICE) /bin/sh

pipeline:
	docker compose run --rm $(SERVICE) uv run python scripts/run_pipeline.py --domain $(DOMAIN) --stage $(STAGE) $(ARGS)

restart:
	docker compose down && docker compose up --build -d
