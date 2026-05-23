#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cmd="${1:-help}"

case "$cmd" in
	install)
		python -m pip install --upgrade pip
		python -m pip install -r "$ROOT_DIR/requirements.txt"
		;;
	local-prep)
		python -m src.data.parquet_converter \
			--raw-dir "$ROOT_DIR/data/raw/telecom" \
			--output "$ROOT_DIR/data/processed/city_traffic.parquet"
		;;
	notebook)
		jupyter lab "$ROOT_DIR/notebooks"
		;;
	help|*)
		echo "Usage: ./run.sh [install|local-prep|notebook]"
		;;
esac
