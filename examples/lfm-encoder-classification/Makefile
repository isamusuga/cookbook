.PHONY: train predict lint format

CONFIG ?= config.yaml
TEXT ?= My payment was accepted, but the order has not shipped.

train:
	uv run train.py --config $(CONFIG)

predict:
	uv run predict.py --config $(CONFIG) --text "$(TEXT)"

lint:
	uv run ruff check .

format:
	uv run ruff format .
