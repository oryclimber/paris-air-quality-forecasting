# Paris air quality forecasting

Forecast of hourly NO2, CO, O3, PM10 and PM2.5 in Paris for the next 504 hours (three weeks). The project compares two baselines (persistence, seasonal naive), a zero-shot time-series foundation model (Chronos-2) and a small LSTM, all scored with MAE on a 504-hour validation window.

## Results

Validation MAE (CO in mg/m³, the rest in µg/m³):

| Model | NO2 | CO | O3 | PM10 | PM2.5 | Average |
|---|---|---|---|---|---|---|
| Persistence | 7.92 | 0.043 | 18.60 | 5.02 | 2.94 | 6.90 |
| Seasonal naive (168 h) | 10.99 | 0.045 | 17.54 | 9.49 | 5.07 | 8.63 |
| Chronos-2 (one model per pollutant) | 19.23 | 0.076 | 20.17 | 9.09 | 4.08 | 10.53 |
| Chronos-2 (joint) | 14.77 | 0.081 | 19.40 | 9.39 | 5.62 | 9.85 |
| LSTM | 8.89 | 0.033 | 14.98 | 5.95 | 3.64 | 6.70 |

The LSTM has the lowest average and is the model used for the final forecast. See the report for the discussion.

## Contents

| Path | What it is |
|---|---|
| `notebooks/time_series_challenge.ipynb` | Full analysis: data checks, validation split, baselines, Chronos-2, LSTM, results, final forecast |
| `report/report.md` | Short written report answering the discussion points |
| `submission/submission.csv` | Final forecast for the 504 test hours (`id` plus the five pollutants) |
| `data/` | `train.csv` and `test.csv` |

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Open the notebook, select the `.venv` kernel and run the cells in order. The first run downloads the Chronos-2 weights (`amazon/chronos-2`) from Hugging Face. On Google Colab, uncomment the `pip install` line in the setup cell of Part Three.

The LSTM is trained from a single fixed seed, so rerunning it can change its numbers slightly. The notebook writes the submission file to `../submission.csv`, so move it to `submission/` if you want to keep the repo layout.
