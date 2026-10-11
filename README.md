# Air-raid alert forecasting in Ukraine

How likely is it that a **new air-raid alert starts in a given region (oblast) within the next 3 hours?** This project answers that for all 24 oblasts (including Kyiv City; Luhansk is excluded), every hour, using only information that was available at the moment of the forecast.

**In short**
- A logistic regression forecasts better than simple "how often did alerts start here at this hour" baselines. On the last 8 weeks of data, tested once, its error is about 9% smaller than the 90-day baseline's (skill score **+0.089**, 95% CI +0.056 to +0.122).
- Against the *strongest* baseline on that test (an ARIMA-based one) the difference is small and not statistically reliable (+0.0095, CI −0.010 to +0.027).
- The success rule we set in advance is met, but the conclusion is limited. The main reasons are in [Limitations](#5-limitations).
- Most of the signal is how often alerts started in the oblast at that hour recently. Neighbours' recent alerts add a little, and richer neighbour features added nothing (a descriptive check on validation only, see [section 4](#4-results)).

**How to read the numbers.** *Brier score* is the average squared error of the forecast probabilities (lower is better). *Skill score* compares a model with a baseline: `1 − Brier(model) / Brier(baseline)`. Zero means "no better than the baseline", a positive value means better. *CI* is the 95% range of plausible values.

## 1. Task and scope

- **Question:** at hour *t*, in an oblast with no alert active, what is the probability that a new alert starts within the next 3 hours?
- **Data:** a snapshot downloaded on 2026-10-10, the freshest available then, and pinned for reproducibility (the source repositories add new data daily, so later snapshots reach past this date). The **test** is the last 8 weeks. The verdict was computed once, after all choices were made; later descriptive checks (alert levels, robustness, text) look at parts of the same period and do not change it.
- **No peeking at the future:** neither training nor features use anything that happened after the forecast moment.

## 2. Reproducibility

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test     # 53 test cases (one needs the downloaded data and is skipped without it), about 1 min in a fresh environment
make all      # downloads data, runs every stage, ~12 min; rewrites reports/
make maps     # redraws figures 12, 16, 21, 22 from the saved tables, without recomputing anything and without reading the test
```

Two optional stages that read the validation periods only are not part of `make all`: `make importance` (per-feature importance and within-oblast AUC, ~6 min) and `make neighbors` (the neighbour-feature prototype, ~1 min). The maps need the oblast boundaries, which `make data` downloads together with the alert data.

- **Python version.** Developed and verified on **Python 3.14.8** only (development and the clean-copy check used the same version). The pinned packages need at least Python 3.12 (numpy and scipy, per their package metadata); other Python versions were not tested.
- Data are downloaded from **fixed commits** and checked by hash (`src/download_data.py`), so numbers do not change when the source repositories update. Package versions are pinned.
- Checked from a clean clone of the pushed repository (fresh virtual environment, `make all`; the optional text experiment is not part of this check): all 53 tests pass, every table in `reports/tables/` matches (only a run-time column differs) and all figures are identical. Verified on Apple Silicon (macOS); other platforms may differ in the last digits. **This check predates the maps (figures 12 and 16 are now maps), figures 21–22 and the two optional stages.** After those changes the 53 tests were run again and pass, and the new stages were run once; the clean-clone check was not repeated.
- **Not reproducible by code:** the analysis of raw Telegram messages behind the parser findings (see [Limitations](#5-limitations)). The optional text experiment (`make telegram text`, not part of `make all`, about 150 MB, tens of minutes to download) reads a live channel, so a re-download gives a different set of messages.

## 3. Method

Each point is one short paragraph; the link leads to the details.

- **Data.** Two alert histories (volunteer-collected and official) and an alerts.in.ua archive that has alert levels. The volunteer history is the main one: it is the longest and the freshest. → [`reports/data_audit.md`](reports/data_audit.md)
- **What is forecast.** One row per oblast and hour, only when no alert is active at that moment. An alert with no recorded all-clear counts as still active, as it would for a live system. → [`docs/problem_statement.md`](docs/problem_statement.md)
- **Test setup.** Four consecutive 4-week periods before the test are used to compare models and tune settings, with a 4-hour gap between training and the period being predicted. The success rule was fixed before the test: the 95% interval of the skill score against the best baseline from validation must lie above zero. → [`docs/methodology_details.md`](docs/methodology_details.md)
- **No leakage.** Every feature uses only alerts that had started, and all-clears that had been recorded, by the forecast moment. 53 automated test cases check this (cutting history at the forecast moment, changing the future, boundary cases). We also deliberately introduced leaks to confirm the tests catch them. → [`tests/`](tests/)
- **Baselines.** Alert frequency by oblast and hour of day over 28 and 90 days, a trailing 90-day rate, a seasonal logistic model, and classical time-series models (ETS, ARIMA) with and without a recalibration step (probabilities adjusted using past data). → [`reports/baselines_val.md`](reports/baselines_val.md)
- **Models.** 31 features in 7 groups (current alert state, recent counts, past frequencies, neighbouring oblasts, whole country, calendar, oblast). Logistic regression is the primary model, gradient boosting the secondary one. Dropping one feature group at a time on validation shows that past frequencies and the oblast's own recent counts matter most; neighbours, whole country, calendar and oblast add little or nothing, and the gains from any group are small. → [`reports/model_val.md`](reports/model_val.md). A later per-feature check (below, "What drives the forecast") refines this for neighbours.

## 4. Results

**Test: last 8 weeks, 23,561 forecasts.**

| Model | Brier score | Skill vs 90-day baseline | Skill vs best test baseline |
|---|---|---|---|
| 90-day frequency baseline | 0.1743 | 0 | n/a |
| 28-day frequency baseline | 0.1644 | n/a | n/a |
| ARIMA with recalibration (best baseline on the test) | 0.1603 | n/a | 0 |
| **Logistic regression (primary)** | **0.1588** | **+0.089 [+0.056; +0.122]** | +0.0095 [−0.010; +0.027] |
| Gradient boosting (secondary) | 0.1572 | +0.098 [+0.065; +0.131] | +0.019 [+0.002; +0.036] |

Details by oblast, hour and before/after the 4 September regime change: [`reports/test_results.md`](reports/test_results.md).

![Forecast vs actual on the test](reports/figures/15_test_forecast_vs_actual.png)

*Daily average forecast probability vs the actual share of alert starts on the test. Black: actual, red: logistic regression, blue: 90-day baseline. The baseline stays flat while the real rate rises; the model follows the trend but underestimates the peaks.*

![Skill by oblast](reports/figures/16_test_bss_by_oblast.png)

*Skill of the logistic regression against the 90-day baseline in each oblast, on the test. Blue is better than the baseline, red is worse; hatched grey marks oblasts with fewer than 30 alert starts (not interpreted), Luhansk (excluded) and Crimea (no data). The model is worse than the baseline only in Zaporizka (about −0.10); in Kharkivska it is equal (−0.004). The largest gain is in Kyivska (+0.31), which may be partly an artefact of the volunteer parser (see Limitations). Kyiv City has no separate boundary in the map data and is drawn as a circle.*

### What drives the forecast and where it works (validation, descriptive)

These checks use the four validation periods only; the test was not read. They describe the models and do not change the verdict. → [`reports/feature_importance.md`](reports/feature_importance.md), [`reports/neighbors.md`](reports/neighbors.md)

![Feature importance](reports/figures/21_feature_importance.png)

*How much within-oblast AUC each model loses when one feature is shuffled (5 shuffles). Both models mostly rely on past alert frequency by oblast and hour and on the oblast's own recent alert counts. Shuffling is unreliable for correlated features; refitting without a feature changes the Brier score by less than 0.0005 for every feature, because others can stand in. Only one feature is clearly not replaceable: the neighbours' alert starts over the last 3 hours in gradient boosting. Without it the within-oblast AUC falls by 0.009 and Brier is worse in all four periods.*

- **Richer neighbour features did not help.** Adding other time scales, sum/maximum/share over neighbours, time since the last start in any neighbour, and second-ring neighbours (nine features in all) changed the within-oblast AUC by at most +0.004, and every 95% interval of the skill score against the same model without them includes zero (one variant is slightly worse). The neighbour signal is already in the base feature set.
- **Features that did nothing.** Seasonal lags, the whole-country counts, the length of the last alert, the share of the last day or week spent under alert, and the 1-hour and 3-hour own counts had no measurable effect; removing them would simplify the model but not improve it.

![Within-oblast AUC by oblast](reports/figures/22_val_auc_by_oblast.png)

*Within-oblast AUC on validation: the baseline (left) against the primary model (right), same scale; 0.50 is chance. The baseline is below chance in Rivnenska, Zhytomyrska and Khmelnytska, where the model reaches 0.64–0.67. In Donetska, Zaporizka and Khersonska the model is not better than the baseline (the baseline alone gives 0.57–0.68 there, because the daily pattern of attacks is stable). Oblasts marked * have fewer than 100 alert starts in the period and are noisy.*

**Alert levels (yellow and red).** Levels exist for only about 5 weeks, so this part is descriptive. A layer that estimates the chance that an alert is red, scored on the last 9 days, has Brier 0.0975, against 0.1347 for a 7-day baseline and 0.1476 for an all-history baseline. → [`reports/levels.md`](reports/levels.md)

![Alert levels](reports/figures/18_levels.png)

*Left: share of red alerts per day (black) falls over the period; the layer (blue) follows it better than the all-history baseline (red). Right: calibration on the last 9 days.*

**Text data (optional extra).** Message counts from the public Air Force channel (how many messages in the last 1 and 3 hours mention the oblast, and how many in total) were added to the logistic regression. They were chosen on validation (better in all four periods) and then tested once: Brier 0.1560 against 0.1578 for the same model without text, skill score +0.0117 (95% CI +0.0035 to +0.0199). Against the strongest baseline (ARIMA with recalibration) it is +0.027 (CI +0.006 to +0.049), but this comparison was added after the run and the text model, unlike the main one, was not fixed before the test, so the verdict stays with the main model. → [`reports/text.md`](reports/text.md)

**Robustness.** With labels taken from the independent alerts.in.ua archive, the skill score against the 90-day baseline is +0.124 [+0.087; +0.159]. Without 7 oblasts where the volunteer series defines an alert start differently, it is +0.078 [+0.047; +0.109]. Neither changes the verdict. → [`reports/robustness.md`](reports/robustness.md)

![Robustness checks](reports/figures/19_robustness.png)

*Skill of the logistic regression under the checks above (points) with 95% intervals; against the strongest baseline most intervals include zero.*

## 5. Limitations

- **Small margin over the strongest baseline.** The 90-day baseline was best on validation but not on the test: the share of alert starts rose and a long window reacts slowly. Much of the gain over it comes from the baseline getting worse, not from the model getting better.
- **The volunteer history is not a clean ground truth.** Its parser overwrites an alert's start when a repeated start message arrives within 3 hours, the message format changed over time, and "an alert start" means different things in different oblasts. How much this affects the scores is unknown. Independent labels (see Robustness) point the same way but do not remove the problem.
- **Choices made on the same validation periods:** baseline, settings and recalibration. The intervals do not account for this, and the data audit had also seen the test-period base rate.
- **Short test across a regime change:** alert levels were introduced on 4 September.
- **Alert levels:** about 5 weeks of data and a changing share of red alerts. The combination "alert starts" × "alert is red" was not evaluated because the two sources define a start differently.
- **Right edge of the data.** The latest hours of the snapshot may be incomplete, so the forecast grid ends at 2026-10-10 01:00 UTC, 3 hours before the hour of the latest observed alert. Every forecast then has a complete 3-hour target window.
- **Synthetic end times and non-stationarity.** About 5% of end times in the volunteer history are synthetic (start + 30 minutes), rising to about 9% in October 2026, and are never used as features. Alert frequency and message format changed over time, so older data may describe a different regime.
- **Text data (optional extra): selected on validation, but not fixed before the test.** The variant (alert counts from the Air Force channel) was chosen on validation, no tuning, and run once on the test. Part of the test had already been seen in an earlier, shorter run. The intervals do not include the choice between two feature variants. The channel's post volume grew about 3.5 times over the period. Details: [`reports/text.md`](reports/text.md).
- **Feature importance and the neighbour experiment are descriptive.** They use the validation periods on which the settings were also chosen, have no intervals across periods for the refitting check, and the per-oblast AUC is noisy where an oblast has few alert starts. They explain the models; they are not new tests.
- Luhansk is excluded. Very long alerts and an anomaly on 2026-04-17 are described in the audit. Numbers under the dataset's own labelling scheme contain retrospective information.

**What to do next**

What would be convincing: a skill score against the *strongest* baseline whose 95% interval lies above zero, and a within-oblast AUC of roughly 0.70 or more (now 0.63 on the test). This is the author's judgement, not a standard. A ceiling is set by the task itself: about half of the 3-hour windows contain an alert start, so large jumps are not realistic.

Any change made after the test has to be evaluated on data after the pinned snapshot (2026-10-10), or reported as post hoc; the test period has been used.

1. Collect the raw message stream to measure how much the parser's overwriting affects eligibility and scores. Labels are the likeliest bottleneck: with independent labels the skill score rose from +0.089 to +0.124.
2. Use the *content* of Air Force messages (type of target, direction, launches of carriers), not only their count. This is the one information source the models have not seen; the count alone gave +0.0117.
3. Separate models or oblast interactions for the frontline south and east. There the baseline already reaches a within-oblast AUC of 0.57–0.68 and the models do not beat it (Donetska, Zaporizka, Khersonska), while the logistic model has the oblast only as a constant.
4. Neighbours on a finer time grid (15 minutes) and with learned directional weights per pair of oblasts. Richer hourly neighbour features were tried and added nothing, so only these variants remain untested.
5. Evaluate the whole chain (start × level) against red alert starts from an independent source.
6. Re-test on a longer period of the new alert regime.
7. Try the alternative target "an alert is active in the window", which needs no eligibility filter.
8. Re-check on the official series.

## 6. Additional materials

**Data sources and credits**
- Volunteer and official alert histories: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT per repository metadata). The volunteer history is built from the eTryvoga Telegram channel; the parser code was read to understand the data.
- Alert levels: alerts.in.ua archive from [JustJimer/ukraine-air-alert-stats](https://github.com/JustJimer/ukraine-air-alert-stats). License not verified.
- Text experiment: public Telegram channel of the Air Force of Ukraine ([t.me/s/kpszsu](https://t.me/s/kpszsu)), pages from 2026-03-26 downloaded on 2026-10-11, not stored in the repository.
- Oblast boundaries (OpenStreetMap-based, so under the ODbL; the repository states no license of its own): [EugeneBorshch/ukraine_geojson](https://github.com/EugeneBorshch/ukraine_geojson), pinned to a commit and checked by hash. Used to check oblast adjacency and to draw the maps. The file has no separate boundary for Kyiv City or Sevastopol.
- Terms of use of the Telegram channels (the one behind the volunteer history and the Air Force channel) were not checked.
- Pinned commits and file sizes: [`src/download_data.py`](src/download_data.py).

**Repository layout.** `src/airalerts/` (package), `src/run_*.py` (stages), `tests/`, `reports/` (results, tables, figures), `docs/` (problem statement, definition of done, methodology details). Reports and docs are in Ukrainian, and so are the figures of the data audit and validation (01–13); figures 14–22 are in English.

**License.** The code is under the [MIT License](LICENSE). The data are not part of this repository and keep the terms of their original sources.

**Note on git history.** The commits were created at the end of the project, grouping the final state of files by logical stage (from problem statement to test results and README). The history is therefore not chronological: it does not show the order in which the work was actually done, and intermediate commits are not guaranteed to pass the tests on their own.

## 7. Reflection

I planned to forecast how many alerts there would be per day. AI suggested the same. I chose a harder question instead: will an alert start in a region in the next 3 hours?

The data gave me the most trouble: duplicates in the official file, and about 5% of end times were fake. I used only end times already known at forecast time.

My first plan had no success rule. I wrote one before the test and ran it once.

The model beats a simple baseline, but not the strongest one reliably. The result is modest, and I trust it within the limits listed above.
