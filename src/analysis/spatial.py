from __future__ import annotations

import numpy as np
import pandas as pd


def square_id_to_grid(square_id: int) -> tuple[int, int]:
	idx = square_id - 1
	row = idx // 100
	col = idx % 100
	return row, col


def traffic_heatmap_matrix(df: pd.DataFrame) -> np.ndarray:
	totals = df.groupby("square_id")["internet_traffic"].sum()
	mat = np.zeros((100, 100), dtype=np.float64)
	for square_id, value in totals.items():
		row, col = square_id_to_grid(int(square_id))
		if 0 <= row < 100 and 0 <= col < 100:
			mat[row, col] = float(value)
	return mat
