from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.memory_utils import compare_memory, optimize_dtypes
from src.data.loader import load_telecom_files
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

	args.output.parent.mkdir(parents=True, exist_ok=True)
	if args.output.exists():
		args.output.unlink()

	with Timer("parquet_conversion") as t:
		writer = None
		row_count = 0
		for chunk in load_telecom_files(files, chunksize=args.chunksize, to_datetime=True):
			# Each row is already aggregated per square/time interval in the raw dataset.
			table = pa.Table.from_pandas(chunk, preserve_index=False)
			if writer is None:
				writer = pq.ParquetWriter(str(args.output), table.schema, compression="snappy")
			writer.write_table(table)
			row_count += len(chunk)
		if writer is not None:
			writer.close()

	print("=== Memory Optimization Probe (sample) ===")
	print(f"Before optimization: {before_mb:.2f} MB")
	print(f"After optimization:  {after_mb:.2f} MB")
	print(f"Reduction:           {reduction_pct:.2f}%")
	print("=== Conversion ===")
	print(f"Rows written: {row_count:,}")
	print(f"Output file:  {args.output}")
	print(f"Elapsed:      {t.elapsed_seconds:.2f} s")


if __name__ == "__main__":
	main()
