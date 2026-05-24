# Colab + Kaggle Forecasting Workflow (Fast and Safe)

Use this guide when you get stuck.

## Goal

- Run `06_sarima_forecasting.ipynb` in Colab (CPU is fine).
- Run `07_lstm_forecasting.ipynb` and `08_gru_forecasting.ipynb` in Kaggle (GPU).
- Run `09_model_comparison.ipynb` in Kaggle with all CSV outputs.

---

## 1) Where each notebook should run

- Colab:
  - `06_sarima_forecasting.ipynb`
- Kaggle:
  - `07_lstm_forecasting.ipynb`
  - `08_gru_forecasting.ipynb`
  - `09_model_comparison.ipynb`

Run notebooks one by one, not all at once.

---

## 2) Find the correct parquet path (Kaggle)

If you get `FileNotFoundError`, do not guess the path. Discover it.

Run this cell:

```python
from pathlib import Path
matches = list(Path("/kaggle/input").rglob("city_traffic.parquet"))
print(matches)
```

Use the exact returned path, for example:

```python
DATA_PATH = Path("/kaggle/input/datasets/miraclenanenmbanaade/telecom-traffic-parquet/city_traffic.parquet")
```

Important: do not write nested `Path(...)` calls like this (wrong):

```python
DATA_PATH = Path(DATA_PATH = Path("..."))
```

---

## 3) Fix missing-column errors quickly

If you get `KeyError: 'square_id'` or `KeyError: 'time_interval'`, inspect columns first:

```python
import pandas as pd
from pathlib import Path

DATA_PATH = Path("/kaggle/input/datasets/miraclenanenmbanaade/telecom-traffic-parquet/city_traffic.parquet")
df = pd.read_parquet(DATA_PATH)
print(df.columns.tolist())
print(df.head(2))
```

Then adapt names if needed:

```python
rename_map = {
    "squareid": "square_id",
    "square": "square_id",
    "time": "time_interval",
    "timestamp": "time_interval",
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
```

---

## 4) SARIMA warning about date frequency

If you see:

- `ValueWarning: ... date index ... has no associated frequency information ...`

it is a warning, not a crash.

Use this before fitting SARIMA:

```python
series = series.copy()
series.index = pd.to_datetime(series.index, utc=True)
series = series.sort_index()

freq = pd.infer_freq(series.index)
if freq is None:
    series = series.resample("10min").mean().interpolate(limit_direction="both")
else:
    series = series.asfreq(freq)
```

---

## 5) Colab -> Kaggle artifact transfer (for final comparison)

You only need CSV outputs for comparison, not trained model objects.

### In Colab (after SARIMA finishes)

Expected files:

- `outputs/metrics_sarima.csv`
- `outputs/sarima_<square_id>.csv` (3 files)

Zip and download:

```python
!zip -r sarima_outputs.zip outputs
```

Download `sarima_outputs.zip` from Colab.

### In Kaggle

1. Create a new Kaggle dataset from those SARIMA CSV files (or zip).
2. Open `09_model_comparison.ipynb`.
3. Click **Add Data** and attach:
   - parquet dataset
   - SARIMA outputs dataset
   - your LSTM/GRU outputs (if stored in another dataset)

---

## 6) Recommended run order

1. `06_sarima_forecasting.ipynb` (Colab)
2. `07_lstm_forecasting.ipynb` (Kaggle, GPU)
3. `08_gru_forecasting.ipynb` (Kaggle, GPU)
4. `09_model_comparison.ipynb` (Kaggle)

Do not run LSTM and GRU at the same time in separate notebooks unless you are ready for memory conflicts or session limits.

---

## 7) Fast checks before long runs

### Kaggle: verify GPU

```python
!nvidia-smi
```

### Kaggle: verify parquet path

```python
from pathlib import Path
print(list(Path('/kaggle/input').rglob('city_traffic.parquet')))
```

### Confirm output files after each notebook

```python
from pathlib import Path
print(sorted([str(p) for p in Path('outputs').glob('*.csv')]))
```

---

## 8) Common errors and exact fixes

- `FileNotFoundError: Parquet not found ...`
  - Fix: discover path under `/kaggle/input` with `rglob`, then set exact `DATA_PATH`.

- `KeyError: 'square_id'` or `KeyError: 'time_interval'`
  - Fix: print `df.columns.tolist()`, then rename columns to expected names.

- SARIMA very slow (10+ minutes)
  - Normal for long series and seasonal models.
  - For quicker runs: fewer squares, shorter history window, fewer parameter combinations.

---

## 9) Minimum files you need at the end

- `outputs/metrics_sarima.csv`
- `outputs/metrics_lstm.csv`
- `outputs/metrics_gru.csv`
- `outputs/sarima_<square>.csv` for the 3 target squares
- `outputs/lstm_<square>.csv` for the 3 target squares
- `outputs/gru_<square>.csv` for the 3 target squares

With these files, `09_model_comparison.ipynb` can produce final tables and plots.
