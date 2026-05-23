from __future__ import annotations

import pandas as pd


def traffic_pdf_sample(df: pd.DataFrame) -> pd.Series:
	return df.groupby("square_id")["internet_traffic"].sum()


def top_traffic_square(df: pd.DataFrame) -> int:
	totals = traffic_pdf_sample(df)
	return int(totals.idxmax())


def first_two_weeks_three_areas(
	df: pd.DataFrame,
	area_a: int,
	area_b: int = 4159,
	area_c: int = 4556,
) -> pd.DataFrame:
	t0 = df["time_interval"].min()
	t1 = t0 + pd.Timedelta(days=14)

	areas = [area_a, area_b, area_c]
	subset = df[df["square_id"].isin(areas)].copy()
	mask = (subset["time_interval"] >= t0) & (subset["time_interval"] < t1)
	subset = subset.loc[mask]
	return subset.sort_values(["square_id", "time_interval"]).reset_index(drop=True)
