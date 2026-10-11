# Повне відтворення: make all (близько 12 хв на Apple Silicon, Python 3.14.8). Окремі етапи: make data | test | audit | baselines | model | blocks | final | levels | robustness
PY ?= python

.PHONY: all data test audit baselines model blocks final levels robustness
all: data test audit baselines model blocks final levels robustness

data:
	$(PY) src/download_data.py
test:
	$(PY) -m pytest tests -q
audit:
	$(PY) src/audit_data.py
baselines:
	$(PY) src/run_baselines.py
model:
	$(PY) src/run_model.py
blocks:
	$(PY) src/run_block_length.py
final:                        # єдиний запуск на тесті; лог лишається в reports/test_run.log
	$(PY) -u src/run_test.py > reports/test_run.log
levels:
	$(PY) -u src/run_levels.py > reports/levels_run.log
robustness:
	$(PY) -u src/run_robustness.py > reports/robustness_run.log
