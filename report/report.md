# Forecasting hourly air pollution in Paris: baselines, a foundation model and an LSTM

All code, tables and plots are in `notebooks/time_series_challenge.ipynb`. MAE is in µg/m³ (CO in mg/m³). The validation period is the 504 hours from 2023-10-18 00:00 to 2023-11-07 23:00, and all models use only data before it.

## 1. Data understanding and preparation

**Why time series and not standard regression?** The hourly observations are ordered and strongly dependent on each other: the lag-1 autocorrelation is 0.85–0.96 and the lag-24 autocorrelation is 0.47–0.71. Standard regression assumes independent rows. Here the task is to extrapolate 21 days beyond the last observation, and the usual explanatory variables (weather, traffic) are not available for that future.

**Why would a random split mislead?** Neighbouring hours are almost copies of each other. With a random split, each test hour has training hours just before and just after it, so the model is tested on interpolation, not on forecasting. Seasonal level and trend also leak from the future into training. The error would be far too optimistic. We therefore split by time: train strictly before 2023-10-18, validate on the next 504 hours.

**Visible patterns.**
- *Daily:* O3 peaks around 14:00 and is lowest around 06:00 (peak/trough ratio 2.0). NO2 peaks at 06:00 and is lowest at 13:00 (ratio 1.9). The one-day plot shows the same rise and fall.
- *Weekly:* NO2 is 1.39× higher on weekdays than on Sundays, and PM10 is 1.22× higher, which fits traffic. CO and PM2.5 show a weak weekly pattern, and O3 is slightly higher on Sundays (0.93×).
- *Yearly:* NO2, CO and PM2.5 peak in winter (January), O3 peaks in June (monthly means differ by 2.3×), and PM10 peaks in March. The long plot also shows short pollution episodes, for example early October 2023.
- *Relations:* PM10 and PM2.5 move together (correlation 0.89), NO2 and CO are positively linked (0.71), and O3 is negatively linked to NO2 (−0.64).

**Missing values.** The share missing is NO2 8%, CO 31%, O3 1.7%, PM10 17.5% and PM2.5 4.4%. CO has one gap of 10,949 hours. The rules, all computed from training data only:
- Negative O3 values are set to missing.
- Gaps of 6 hours or less are forward-filled.
- Gaps of 7–48 hours are filled with the weekday × hour mean profile of the training set.
- Longer gaps stay missing. The foundation model accepts missing values, and the LSTM ignores missing targets in its loss.

We must not interpolate with future validation or test data, because an interpolation across a gap uses the values after it. Those are exactly the values the model is supposed to predict. The score would then reflect leaked information and could not be reproduced at forecast time.

## 2. Validation strategy and baselines

We built two baselines: **persistence** (repeat the last value) and **seasonal naive** (repeat the last L hours). We tested L = 24, 168 and 336. MAE on validation was 8.4 for L=24, 8.6 for L=168 and 15.0 for L=336 (average over the five pollutants).

**Why a 168-hour lag?** It is the length of one full week, so it keeps both the hour of the day and the day of the week. Urban pollution follows traffic and human activity, which are different on weekdays and weekends: NO2 is 1.39× higher on weekdays than on Sundays. A 24-hour lag copies the wrong kind of day around weekends, while 168 hours always compares like with like. In our validation it helped only for O3 (17.5 vs 18.6 for persistence). Overall it was worse than persistence (8.6 vs 6.9), because the weekly signal is weak for PM and the last week before validation had unusually high pollution. Longer lags (336 hours) copy older episodes and are worse still.

**Why report MAE per pollutant?** The average is in raw units, so it is dominated by O3 and NO2 (tens of µg/m³) while CO (0.04 mg/m³) hardly counts. Pollutants also have different patterns, so a model can be best on one and worst on another. In our results, the best model on average is not the best for every pollutant (see section 5).

## 3. Time-series foundation models

**What is a time-series foundation model?** It is a large model pretrained on a very large and varied collection of time series from many domains, which can forecast a new series directly from its history. A model trained only on this dataset learns from just one city, 4.7 years and five variables. A foundation model brings patterns learned elsewhere (daily and weekly cycles, trends, noise), but it knows nothing specific about Paris.

**What does zero-shot mean here?** We use `amazon/chronos-2` (Chronos-2, 120M parameters, chronos-forecasting 2.3.2) as published. We give it the history before the validation period (8,192 hours of context) and read off a forecast for the next 504 hours. No weights are updated on the Paris data and nothing is tuned on the validation period. Hardware: Apple M-series GPU (MPS). The notebook also runs on Colab.

**Forecasting all pollutants together.**
- *Advantage:* the pollutants are related (NO2–CO 0.71, PM10–PM2.5 0.89), so one series can help forecast another, for example when one has missing data or a clear signal.
- *Risk:* a misleading relation can spread errors from one pollutant to the others, and a pollutant with a different behaviour (O3 is anti-correlated with NO2) can be pulled in the wrong direction.

**Variable roles.**

| Variable | Role |
|---|---|
| Pollutant history | Target (its past values are the model input) |
| Historical weather | Past-only covariate |
| Hour of day | Known-future covariate (the calendar is known in advance) |
| Future weather observations | Not usable as is: observations of the future do not exist at forecast time. Only weather forecasts could be a known-future covariate, and using true future values would be leakage. |

We use no weather data in this work.

**Why the median forecast for MAE?** The constant that minimises the expected absolute error is the median of the predictive distribution, not the mean. The median is also less affected by pollution spikes. We therefore use the 0.5 quantile of Chronos-2 as the point forecast.

## 4. Comparative experiments

Validation MAE (same period, same metric):

| Model | NO2 | CO | O3 | PM10 | PM2.5 | Average |
|---|---|---|---|---|---|---|
| Persistence | **7.92** | 0.043 | 18.60 | **5.02** | **2.94** | 6.90 |
| Seasonal naive (168 h) | 10.99 | 0.045 | 17.54 | 9.49 | 5.07 | 8.63 |
| Chronos-2, one model per pollutant | 19.23 | 0.076 | 20.17 | 9.09 | 4.08 | 10.53 |
| Chronos-2, joint | 14.77 | 0.081 | 19.40 | 9.39 | 5.62 | 9.85 |
| LSTM (trained on the Paris data) | 8.89 | **0.033** | **14.98** | 5.95 | 3.64 | **6.70** |

**Which approach works best?** The LSTM has the lowest average MAE and is best on CO and O3. It is also clearly better than both Chronos-2 variants on NO2, CO and O3. It is a joint model that reads all five pollutants and outputs all five. Between the two Chronos-2 setups, joint forecasting is better for NO2 (14.8 vs 19.2) and slightly for O3, which fits the strong NO2–O3 and NO2–CO relations. Independent forecasting is better for PM2.5 (4.1 vs 5.6) and slightly for PM10, so there is no single winner and the joint advantage is not general. The LSTM probably benefits from having learned the Paris-specific cycles, and from being trained directly on 504-hour outputs with an MAE loss. Its edge over persistence is small (6.7 vs 6.9), and persistence is still better on NO2, PM10 and PM2.5. A single run on a single window is not enough to be sure of the ranking.

## 5. Results and discussion

**Which model is best overall? Best for every pollutant?** The LSTM is best overall (average MAE 6.70). It is not best for every pollutant: persistence wins NO2, PM10 and PM2.5. The average hides this, and it is also dominated by O3 and NO2 because it is in raw units. As a share of each pollutant's mean level, the ranking by pollutant is clearer (see the bar chart in the notebook).

**Horizon.** Errors usually grow from week 1 to week 3. The average error as a share of the mean level goes from 40% to 58% for Chronos-2 joint, and from 27% to 33% for the LSTM. Persistence stays flat at about 29%, so it does not degrade over this window. Chronos-2 on NO2 gets worse from 53% to 81%.

**Where does the foundation model fail compared with seasonal naive?** Joint Chronos-2 is worse than seasonal naive on all five pollutants, most clearly NO2 (14.8 vs 11.0) and CO (0.081 vs 0.045). The reason is visible in the observed-vs-predicted plot. The model reproduces the daily shape well, but its level is wrong: it reverts to the level of its recent context, the two polluted weeks before validation (mean NO2 about 39 µg/m³), and forecasts about 32 µg/m³ when the validation period was cleaner (about 18). Seasonal naive copies the previous week, whose level is closer to the next days. The one-per-pollutant Chronos-2 beats seasonal naive on PM10 and PM2.5.

**Three limitations.**
- *Data:* the data are one series per pollutant, with no weather or traffic variables, and with a lot of missing data (CO 31%, including a 10,949-hour gap). The imputation is simple and affects what the models learn.
- *Evaluation:* we use one validation window of 504 hours in autumn, while the competition test period is in September. A single window is noisy (one pollution episode changes the ranking), and the season differs. The average MAE is in raw units.
- *Model:* Chronos-2 is used with no covariates, no fine-tuning and one context length. The LSTM is small, uses fixed untuned hyperparameters and one random seed, so its result will change somewhat between runs.

## 6. Final forecast

The final method was chosen with validation results only: the lowest average MAE, which is the LSTM. It was retrained on all available data (up to 2024-09-03 22:00) and used to forecast the 504 hours of the test period. The file `submission.csv` has the `id` column exactly as in `test.csv`, then NO2, CO, O3, PM10 and PM2.5 in the same order as `train.csv`. The notebook checks the columns, the 504 rows in test order, the absence of missing values and the absence of negative values.
