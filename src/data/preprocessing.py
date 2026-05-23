from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import load_telecom_files
from src.utils.helpers import sorted_files


def collect_telecom_files(raw_dir: Path) -> list[Path]:
	files = sorted_files(raw_dir.glob("sms-call-internet-mi-*.txt"))
	if not files:
		raise FileNotFoundError(f"No telecom files found in {raw_dir}")
	return files


def build_city_traffic_dataframe(raw_dir: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
	files = collect_telecom_files(raw_dir)
	aggregated_frames: list[pd.DataFrame] = []

	for chunk in load_telecom_files(files, chunksize=chunksize, to_datetime=True):
		grouped = (
			chunk.groupby(["square_id", "time_interval"], as_index=False)["internet_traffic"]
			.sum()
			.sort_values(["square_id", "time_interval"])
		)
		aggregated_frames.append(grouped)

	city_df = pd.concat(aggregated_frames, ignore_index=True)
	city_df = (
		city_df.groupby(["square_id", "time_interval"], as_index=False)["internet_traffic"]
		.sum()
		.sort_values(["square_id", "time_interval"])
		.reset_index(drop=True)
	)
	return city_df


def save_city_parquet(df: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(output_path, index=False)


def load_city_parquet(parquet_path: Path) -> pd.DataFrame:
	return pd.read_parquet(parquet_path)


def get_total_traffic_per_square(df: pd.DataFrame) -> pd.Series:
	return df.groupby("square_id")["internet_traffic"].sum().sort_values(ascending=False)


def get_top_square_id(df: pd.DataFrame) -> int:
	return int(get_total_traffic_per_square(df).index[0])


def get_area_series(df: pd.DataFrame, square_id: int) -> pd.Series:
	subset = df[df["square_id"] == square_id].sort_values("time_interval")
	return subset.set_index("time_interval")["internet_traffic"]


def subset_timerange(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
	start_ts = pd.Timestamp(start, tz="UTC")
	end_ts = pd.Timestamp(end, tz="UTC")
	mask = (df["time_interval"] >= start_ts) & (df["time_interval"] <= end_ts)
	return df.loc[mask].copy()
