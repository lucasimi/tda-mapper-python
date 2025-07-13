PYTHON = python
PIP = pip

.PHONY: all
all: install

.PHONY: install
install:
	$(PIP) install -e .[dev]

.PHONY: typecheck
typecheck:
	mypy src --ignore-missing-imports

.PHONY: test
test:
	coverage run --source=src -m pytest tests/test_unit_*.py
	coverage xml

.PHONY: bench
bench:
	$(PYTHON) -m pytest tests/test_bench_*.py -s -o log_cli=true --log-level=INFO

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
