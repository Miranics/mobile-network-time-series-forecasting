from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.memory_utils import compare_memory, optimize_dtypes
from src.data.preprocessing import build_city_traffic_dataframe, save_city_parquet
from src.utils.config import get_config
from src.utils.timer import Timer


def _memory_probe(sample_file: Path, nrows: int = 300_000) -> tuple[float, float, float]:
	cols = [0, 1, 7]
	names = [
		"square_id",
		"time_interval",
		"country_code",
		"sms_in",
		"sms_out",
		"call_in",
		"call_out",
		"internet_traffic",
	]
	raw = pd.read_csv(
		sample_file,
		sep="\t",
		header=None,
		names=names,
		usecols=cols,
		nrows=nrows,
	)
	optimized = optimize_dtypes(raw)
	cmp = compare_memory(raw, optimized)
	return cmp.before_mb, cmp.after_mb, cmp.reduction_pct


def main() -> None:
	cfg = get_config()
	parser = argparse.ArgumentParser(description="Convert telecom txt files to optimized parquet")
	parser.add_argument("--raw-dir", type=Path, default=cfg.data_raw)
	parser.add_argument(
		"--output",
		type=Path,
		default=cfg.data_processed / "city_traffic.parquet",
	)
	parser.add_argument("--chunksize", type=int, default=2_000_000)
	args = parser.parse_args()

	files = sorted(args.raw_dir.glob("sms-call-internet-mi-*.txt"))
	if not files:
		raise FileNotFoundError(f"No telecom files found in {args.raw_dir}")

	before_mb, after_mb, reduction_pct = _memory_probe(files[0])

	with Timer("parquet_conversion") as t:
		city_df = build_city_traffic_dataframe(args.raw_dir, chunksize=args.chunksize)
		save_city_parquet(city_df, args.output)

	print("=== Memory Optimization Probe (sample) ===")
	print(f"Before optimization: {before_mb:.2f} MB")
	print(f"After optimization:  {after_mb:.2f} MB")
	print(f"Reduction:           {reduction_pct:.2f}%")
	print("=== Conversion ===")
	print(f"Rows written: {len(city_df):,}")
	print(f"Output file:  {args.output}")
	print(f"Elapsed:      {t.elapsed_seconds:.2f} s")


if __name__ == "__main__":
	main()
