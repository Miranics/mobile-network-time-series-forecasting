from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def rolling_statistics(series: pd.Series, window: int = 144) -> pd.DataFrame:
	return pd.DataFrame(
		{
			"value": series,
			"rolling_mean": series.rolling(window=window).mean(),
			"rolling_std": series.rolling(window=window).std(),
		}
	)


def adf_test(series: pd.Series) -> dict[str, float]:
	clean = series.dropna()
	stat, pvalue, used_lag, nobs, _, critical_values = adfuller(clean)
	out: dict[str, float] = {
		"adf_stat": float(stat),
		"p_value": float(pvalue),
		"used_lag": float(used_lag),
		"n_obs": float(nobs),
	}
	for k, v in critical_values.items():
		out[f"critical_{k}"] = float(v)
	return out
