init:
	pipenv install --dev

test:
	pipenv run python -m pytest tests

lint:
	-flake8
	mypy --namespace-packages tensor_networks/

test-all: lint test

.PHONY: test lint