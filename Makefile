# Повне відтворення: make all (близько 12 хв на Apple Silicon, Python 3.14.8). Окремі етапи: make data | test | audit | baselines | model | blocks | final | levels | robustness | maps. Опційно (не входять у all): make importance neighbors; текстовий експеримент make telegram text
PY ?= python

.PHONY: all data test audit baselines model blocks final levels robustness maps importance neighbors telegram text
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
importance:                   # опційно, лише валідація: важливість окремих ознак і AUC по областях (~6 хв), не входить у all
	$(PY) -u src/run_importance.py > reports/importance_run.log
neighbors:                    # опційно, лише валідація: прототип ознак сусідів (~1 хв), не входить у all
	$(PY) -u src/run_neighbors.py > reports/neighbors_run.log
maps:                         # перемальовує рис. 12, 16, 21 і 22 (карти й важливість ознак) з готових таблиць, без тесту й без обчислень
	$(PY) src/make_maps.py
telegram:                     # опційно: завантаження сторінок каналу (потрібен інтернет), не входить у all
	$(PY) src/download_telegram.py
text:                         # опційно: текстовий міні-експеримент, потребує make telegram
	$(PY) -u src/run_text.py > reports/text_run.log
