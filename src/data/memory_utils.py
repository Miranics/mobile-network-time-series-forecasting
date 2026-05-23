from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def bytes_to_mb(n_bytes: int | float) -> float:
	return float(n_bytes) / (1024.0 ** 2)


def dataframe_memory_mb(df: pd.DataFrame) -> float:
	return bytes_to_mb(df.memory_usage(deep=True).sum())


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
	optimized = df.copy()
	for col in optimized.columns:
		series = optimized[col]
		if pd.api.types.is_integer_dtype(series):
			optimized[col] = pd.to_numeric(series, downcast="integer")
		elif pd.api.types.is_float_dtype(series):
			optimized[col] = pd.to_numeric(series, downcast="float")
	return optimized


@dataclass(frozen=True)
class MemoryComparison:
	before_mb: float
	after_mb: float

	@property
	def reduction_mb(self) -> float:
		return self.before_mb - self.after_mb

	@property
	def reduction_pct(self) -> float:
		if self.before_mb == 0:
			return 0.0
		return (self.reduction_mb / self.before_mb) * 100.0


def compare_memory(before_df: pd.DataFrame, after_df: pd.DataFrame) -> MemoryComparison:
	return MemoryComparison(
		before_mb=dataframe_memory_mb(before_df),
		after_mb=dataframe_memory_mb(after_df),
	)
