from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.data.memory_utils import optimize_dtypes


ALL_COLUMNS = [
	"square_id",
	"time_interval",
	"country_code",
	"sms_in",
	"sms_out",
	"call_in",
	"call_out",
	"internet_traffic",
]

USE_COLUMNS = ["square_id", "time_interval", "internet_traffic"]


def load_telecom_file(
	file_path: Path,
	chunksize: int = 2_000_000,
	to_datetime: bool = True,
) -> Iterator[pd.DataFrame]:
	col_indices = [ALL_COLUMNS.index(c) for c in USE_COLUMNS]
	for chunk in pd.read_csv(
		file_path,
		sep="\t",
		header=None,
		names=ALL_COLUMNS,
		usecols=col_indices,
		dtype={
			"square_id": "int32",
			"time_interval": "int64",
			"internet_traffic": "float32",
		},
		chunksize=chunksize,
		low_memory=True,
	):
		if to_datetime:
			chunk["time_interval"] = pd.to_datetime(
				chunk["time_interval"], unit="ms", utc=True
			)
		yield optimize_dtypes(chunk)


def load_telecom_files(
	file_paths: list[Path],
	chunksize: int = 2_000_000,
	to_datetime: bool = True,
) -> Iterator[pd.DataFrame]:
	for path in file_paths:
		yield from load_telecom_file(path, chunksize=chunksize, to_datetime=to_datetime)
