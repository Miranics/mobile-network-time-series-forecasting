from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


def build_sequences(series: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
	if sequence_length <= 0:
		raise ValueError("sequence_length must be > 0")
	if len(series) <= sequence_length:
		raise ValueError("series length must be greater than sequence_length")

	x, y = [], []
	for i in range(len(series) - sequence_length):
		x.append(series[i : i + sequence_length])
		y.append(series[i + sequence_length])
	return np.asarray(x), np.asarray(y)


def chronological_split(
	x: np.ndarray,
	y: np.ndarray,
	val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if not 0.0 < val_ratio < 1.0:
		raise ValueError("val_ratio must be in (0, 1)")
	split_idx = int(len(x) * (1.0 - val_ratio))
	return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


@dataclass
class TimedResult:
	result: object
	elapsed_seconds: float


def time_call(fn, *args, **kwargs) -> TimedResult:
	start = time.perf_counter()
	result = fn(*args, **kwargs)
	elapsed = time.perf_counter() - start
	return TimedResult(result=result, elapsed_seconds=elapsed)
