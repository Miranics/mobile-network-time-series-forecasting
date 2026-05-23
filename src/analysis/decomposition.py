from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose_series(
	series: pd.Series,
	period: int = 144,
	model: str = "additive",
):
	clean = series.dropna()
	return seasonal_decompose(clean, period=period, model=model)
