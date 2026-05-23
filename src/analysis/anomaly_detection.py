from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
	clean = series.dropna()
	std = clean.std()
	if std == 0:
		return pd.Series(False, index=series.index)
	z = (series - clean.mean()) / std
	return z.abs() > threshold


def iqr_anomalies(series: pd.Series, factor: float = 1.5) -> pd.Series:
	q1 = series.quantile(0.25)
	q3 = series.quantile(0.75)
	iqr = q3 - q1
	lower = q1 - factor * iqr
	upper = q3 + factor * iqr
	return (series < lower) | (series > upper)


def anomaly_summary(series: pd.Series) -> dict[str, float]:
	z_flags = zscore_anomalies(series)
	iqr_flags = iqr_anomalies(series)
	n = len(series)
	return {
		"zscore_count": float(np.sum(z_flags)),
		"zscore_pct": float(np.mean(z_flags) * 100.0) if n else 0.0,
		"iqr_count": float(np.sum(iqr_flags)),
		"iqr_pct": float(np.mean(iqr_flags) * 100.0) if n else 0.0,
	}
