PY ?= python
DATA ?= data/yaseen
RUN ?=

.PHONY: install test test-fast env smoke verify report

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

env:
	$(PY) -m heartsim env

smoke:
	$(PY) -m heartsim smoke --data-root $(DATA) --workers 2

verify:
	$(PY) -m heartsim verify $(RUN) --data-root $(DATA)

report:
	$(PY) -m heartsim report $(RUN)
