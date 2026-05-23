from __future__ import annotations

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SarimaForecaster:
	def __init__(
		self,
		order: tuple[int, int, int] = (1, 1, 1),
		seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 144),
	) -> None:
		self.order = order
		self.seasonal_order = seasonal_order
		self.model_fit = None

	def fit(self, train_series: np.ndarray):
		model = SARIMAX(
			train_series,
			order=self.order,
			seasonal_order=self.seasonal_order,
			enforce_stationarity=False,
			enforce_invertibility=False,
		)
		self.model_fit = model.fit(disp=False)
		return self

	def forecast(self, steps: int) -> np.ndarray:
		if self.model_fit is None:
			raise RuntimeError("Model must be fit before forecasting")
		return np.asarray(self.model_fit.forecast(steps=steps))

	def rolling_one_step_forecast(
		self,
		train_series: np.ndarray,
		test_series: np.ndarray,
	) -> np.ndarray:
		history = list(train_series)
		preds = []
		for y_true in test_series:
			model = SARIMAX(
				history,
				order=self.order,
				seasonal_order=self.seasonal_order,
				enforce_stationarity=False,
				enforce_invertibility=False,
			)
			fit = model.fit(disp=False)
			yhat = fit.forecast(steps=1)[0]
			preds.append(float(yhat))
			history.append(float(y_true))
		return np.asarray(preds, dtype=np.float32)
