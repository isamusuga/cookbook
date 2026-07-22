.PHONY: train predict lint format

CONFIG ?= config.yaml
TEXT ?= I was charged twice for express delivery, and my package has not shipped.

train:
	uv run train.py --config $(CONFIG)

predict:
	uv run predict.py --config $(CONFIG) --text "$(TEXT)"

lint:
	uv run ruff check .

format:
	uv run ruff format .
