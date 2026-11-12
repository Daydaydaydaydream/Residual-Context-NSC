.PHONY: help install lint format test check starter demo verify baseline sweep clean

PYTHON ?= python

help:
	@echo "install  Install the package with development extras"
	@echo "lint     Run ruff lint and format checks"
	@echo "format   Apply ruff autofixes and formatting"
	@echo "test     Run the test suite with coverage"
	@echo "check    Lint, format check and tests (what CI runs)"
	@echo "starter  Run the course starter self-check"
	@echo "demo     Run the controlled FFN reordering example"
	@echo "verify   Check beta=0 adapter scores against original NSC on all 200 configs"
	@echo "baseline Reproduce the development baseline and compare it with the reference"
	@echo "sweep    Run the 33-run development beta sweep for the three settings"

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format src tests examples scripts

test:
	$(PYTHON) -m pytest --cov=cs5491_nsc --cov-report=term-missing

check: lint test

starter:
	cd course/nsc-starter && $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON)) check_starter.py

demo:
	$(PYTHON) -m examples.reorder_demo

verify:
	$(PYTHON) scripts/verify_adapter.py

baseline:
	$(PYTHON) scripts/verify_baseline.py

sweep:
	$(PYTHON) scripts/sweep_beta.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -not -path "./.git/*" -exec rm -rf {} +
