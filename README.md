# Mobile Network Time Series Forecasting

Comparative time series analysis and forecasting of large-scale mobile network traffic using statistical and deep learning models.

## Assignment Scope

This project implements the full workflow requested in the coursework:

- Task 1: memory-efficient large data handling and transformation
- Task 2: exploratory time series and spatial characterization
- Task 3: one statistical model (SARIMA) + two neural models (LSTM, GRU)

## Recommended Workflow (Hybrid)

Use local resources for data engineering and EDA, then move long deep-learning runs to GPU.

- Local now:
  - data loading
  - memory optimization
  - parquet conversion
  - EDA and stationarity checks
- Kaggle or Colab later:
  - repeated LSTM training
  - repeated GRU training
  - heavy hyperparameter tuning

## Quick Start

1. Create and activate your environment.
2. Install dependencies.
3. Build optimized parquet from raw txt files.

```bash
./run.sh install
./run.sh local-prep
```

## Run Notebooks

```bash
./run.sh notebook
```

## Core Modules

- `src/data/loader.py`: chunked loading of TIM telecom files
- `src/data/memory_utils.py`: dtype downcasting and memory reporting
- `src/data/preprocessing.py`: aggregation and feature-ready subsets
- `src/data/parquet_converter.py`: CLI conversion and optimization stats
- `src/analysis/*.py`: EDA, stationarity, decomposition, anomalies, spatial analysis
- `src/models/*.py`: SARIMA, LSTM, GRU, metrics, and training utilities
- `src/visualization/*.py`: plots for EDA and forecast overlays

## Project Objectives

This project focuses on:

- Large-scale mobile traffic data handling
- Memory-efficient preprocessing
- Exploratory time series analysis
- Spatial traffic analysis
- Traffic forecasting using:
  - SARIMA
  - LSTM
  - GRU
- Comparative evaluation of forecasting models

## Dataset

The dataset is based on the Telecom Italia Mobile (TIM) dataset for the city of Milan.

### Data Download Checklist

- Download the Telecommunications activity dataset from Harvard Dataverse.
- Place daily files into data/raw/telecom.
- Expected full coverage is daily files from 2013-11-01 to 2013-12-31 (61 files).
- Use the preprocessing notebook to report missing files before heavy runs.

References:
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

## Project Structure

```text
src/            -> reusable source code
notebooks/      -> experimentation and analysis
reports/        -> figures, tables, and report drafts
data/           -> raw and processed datasets
```

## Technologies

- Python
- Pandas
- NumPy
- Statsmodels
- TensorFlow
- Scikit-learn
- Matplotlib
- Seaborn
- PyArrow

## Author

Nanen Miracle Mbanaade