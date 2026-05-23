from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def sorted_files(paths: Iterable[Path]) -> list[Path]:
	return sorted(paths, key=lambda p: p.name)


def seconds_to_hms(seconds: float) -> str:
	h = int(seconds // 3600)
	m = int((seconds % 3600) // 60)
	s = seconds % 60
	if h > 0:
		return f"{h}h {m}m {s:.2f}s"
	if m > 0:
		return f"{m}m {s:.2f}s"
	return f"{s:.2f}s"
