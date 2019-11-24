init:
	pipenv install --dev

test:
	pipenv run python -m pytest tests

.PHONY: init test