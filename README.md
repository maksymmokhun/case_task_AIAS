# Air-raid alert forecasting in Ukraine

Forecast the probability that a **new air-raid alert starts in an oblast within the next 3 hours**, using only information available at the forecast moment. Hourly grid, 24 oblasts including Kyiv City (Luhansk excluded).

**TL;DR.** On the last 8 weeks of data (a single run on a held-out test), a logistic regression beats the simple 90-day-frequency baseline: Brier skill score **+0.089, 95% CI [+0.056; +0.122]**. The advantage over the *strongest* baseline on the test (an ARIMA-based one) is not statistically significant (+0.0095, CI [−0.010; +0.027]). The success rule is met, but the conclusion is limited: see [Limitations](#5-limitations).

## 1. Task and scope

- **Target:** P(an alert starts in oblast r during (t, t+3h]), given that no alert is active at t.
- **Data:** the freshest available snapshot, 2026-10-10. **Test:** the last 8 weeks, touched once, after all choices were fixed.
- **No leakage:** neither training nor features use information from after the forecast moment.

## 2. Reproducibility

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test     # 43 test cases (one needs the downloaded data and is skipped without it), ~1 min: leakage and boundary checks on synthetic data
make all      # downloads data, runs every stage, ~12 min; rewrites reports/
```

- Data are downloaded from **fixed commits** and verified by hash (`src/download_data.py`), so numbers do not drift when the upstream repositories update daily. Package versions are pinned.
- Checked from a clean copy of the repository (fresh virtual environment, `make all`): all 43 tests pass, every table in `reports/tables/` matches the tables in this repository (only a run-time column differs) and all figures are byte-identical. Verified on Apple Silicon with Python 3.14.8; other platforms may differ in the last digits.
- **Not reproducible by code:** the analysis of raw Telegram messages behind the parser findings (see [Limitations](#5-limitations)).

## 3. Method

Details for each point are in the linked documents.

- **Data.** Two alert series (volunteer-collected and official) and an alerts.in.ua archive with alert levels. The volunteer series is the main one because it is the longest and freshest. → [`reports/data_audit.md`](reports/data_audit.md)
- **Target and rows.** One row per oblast and hour; only moments with no active alert are forecast. An alert without an observed all-clear is treated as still active, which is what a live system would know. → [`docs/problem_statement.md`](docs/problem_statement.md)
- **Validation.** Four consecutive 4-week windows before the test are used to compare models and tune settings, with a 4-hour gap between training and the predicted period. The success rule was fixed before the test: the 95% interval (resampled 3-day blocks of time) of the skill score against the best validation baseline must lie above zero. → [`docs/methodology_details.md`](docs/methodology_details.md)
- **No leakage.** Every feature uses only events started by the forecast moment and all-clears observed by then. 43 automated test cases (truncation to the forecast moment, future perturbation, label and boundary checks), verified by deliberately introducing leaks that the tests must catch. → [`tests/`](tests/)
- **Baselines.** Frequency by oblast and hour of day over 28 and 90 days, a trailing 90-day rate, a seasonal logistic model, and classical time-series models (ETS, ARIMA) with and without recalibration. → [`reports/baselines_val.md`](reports/baselines_val.md)
- **Models.** 32 features in 7 groups (own state, counts, priors, neighbours by shared border, country-wide, calendar, oblast). Logistic regression is the primary model; gradient boosting is a secondary one. → [`reports/model_val.md`](reports/model_val.md)

## 4. Results

**Test (last 8 weeks, 23,561 forecast rows).**

| Model | Brier score | Skill vs 90-day baseline | Skill vs best test baseline |
|---|---|---|---|
| 90-day frequency baseline | 0.1743 | 0 | n/a |
| 28-day frequency baseline | 0.1644 | n/a | n/a |
| ARIMA + recalibration (best baseline on test) | 0.1603 | n/a | 0 |
| **Logistic regression (primary)** | **0.1588** | **+0.089 [+0.056; +0.122]** | +0.0095 [−0.010; +0.027] |
| Gradient boosting (secondary) | 0.1572 | +0.098 [+0.065; +0.131] | +0.019 [+0.002; +0.036] |

Lower Brier is better; skill is `1 − Brier(model) / Brier(baseline)`. Breakdown by oblast, by hour and before/after the 4 September regime change: [`reports/test_results.md`](reports/test_results.md) (figures 14–17).

**Alert levels (yellow/red), stage 2.** Levels exist only for about 5 weeks. A layer P(red | alert starts), scored on the last 9 days: Brier 0.0975 vs 0.1347 (7-day baseline) and 0.1476 (all-history baseline). Descriptive only, no verdict. → [`reports/levels.md`](reports/levels.md)

**Robustness.** On independent labels from alerts.in.ua the skill score against the 90-day baseline is +0.124 [+0.087; +0.159]. Without 7 oblasts where the volunteer series defines an alert start differently it is +0.078 [+0.047; +0.109]. Neither changes the verdict. → [`reports/robustness.md`](reports/robustness.md)

## 5. Limitations

- **Narrow margin over the strongest baseline.** The 90-day baseline was the best on validation but not on the test: the share of alert starts rose, and a long window lags. Most of the gain over it reflects the baseline getting worse, not the model getting better.
- **The volunteer series is not a clean ground truth.** Its parser overwrites an alert start when a repeated start message arrives within 3 hours, message formats changed over time, and "an alert start" means different things in different oblasts. The scale of the effect on the scores is unknown. Independent labels (see Robustness) point the same way but do not remove it.
- **Choices made on the same validation windows:** baseline, hyperparameters, recalibration. Intervals do not account for this. The data audit also saw the test-period base rate.
- **Short test crossing a regime change** (alert levels were introduced on 4 September).
- **Alert levels:** about 5 weeks of data and a non-stationary share of red alerts. The product P(start) × P(red | start) was not evaluated, because the two sources define a start differently.
- **Right edge of the data.** The latest hours of the snapshot may still be incomplete; the forecast grid ends at 2026-10-10 01:00 UTC, 3 hours before the hour of the latest observed alert, so that every forecast has a complete 3-hour target window.
- **Synthetic end times and non-stationarity.** About 5% of end times in the volunteer series are synthetic (start + 30 minutes); they are never used as features. The series is non-stationary: alert frequency changes over time and the message format changed, so older data may describe a different regime.
- Luhansk is excluded. Very long alerts and an anomaly on 2026-04-17 are described in the audit. Numbers under the dataset's own labelling scheme contain retrospective information.

**What to do next**
1. Collect the raw message stream to measure how much the parser's overwriting affects eligibility and scores.
2. Evaluate the whole chain (start × level) against red alert starts from an independent source.
3. Re-test on a longer period of the new alert regime.
4. Try the alternative target "an alert is active in the window", which needs no eligibility filter.
5. Re-check on the official series.

## 6. Additional materials

**Data sources and credits**
- Volunteer and official alert series: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT per repository metadata). The volunteer series is built from the eTryvoga Telegram channel; the parser code was read to understand the data.
- Alert levels: alerts.in.ua archive from [JustJimer/ukraine-air-alert-stats](https://github.com/JustJimer/ukraine-air-alert-stats). License not verified.
- Oblast adjacency was checked against OpenStreetMap-based boundaries ([EugeneBorshch/ukraine_geojson](https://github.com/EugeneBorshch/ukraine_geojson)).
- Terms of use of the original Telegram channels were not checked.
- Pinned commits and file sizes: [`src/download_data.py`](src/download_data.py).

**Repository layout.** `src/airalerts/` (package), `src/run_*.py` (stages), `tests/`, `reports/` (results, tables, figures), `docs/` (problem statement, definition of done, methodology details). Reports and docs are in Ukrainian.

**Note on git history.** The commits were created at the end of the project, grouping the final state of files by logical stage (from problem statement to test results and README). The history is therefore not chronological: it does not show the order in which the work was actually done, and intermediate commits are not guaranteed to pass the tests on their own.

## 7. Reflection

*TODO: author (about 100 words).*
