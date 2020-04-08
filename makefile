init:
	pipenv install --dev

test:
	pipenv run python -m pytest tests

lint:
	-flake8
	mypy
