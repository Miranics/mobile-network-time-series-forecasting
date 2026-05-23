from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
	denom = np.maximum(np.abs(y_true), eps)
	return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
	return {
		"MAE": mae(y_true, y_pred),
		"MAPE": mape(y_true, y_pred),
		"RMSE": rmse(y_true, y_pred),
	}


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
	df = pd.DataFrame(results).T
	return df[["MAE", "MAPE", "RMSE"]].sort_index()
