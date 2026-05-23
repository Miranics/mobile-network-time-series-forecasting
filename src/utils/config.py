from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import sys


@dataclass(frozen=True)
class ProjectConfig:
	root: Path
	data_raw: Path
	data_processed: Path
	reports_figures: Path
	reports_tables: Path


def get_config() -> ProjectConfig:
	root = Path(__file__).resolve().parents[2]
	return ProjectConfig(
		root=root,
		data_raw=root / "data" / "raw" / "telecom",
		data_processed=root / "data" / "processed",
		reports_figures=root / "reports" / "figures",
		reports_tables=root / "reports" / "tables",
	)


def get_system_info() -> dict[str, str]:
	return {
		"os": f"{platform.system()} {platform.release()}",
		"python": sys.version.split()[0],
		"processor": platform.processor() or "unknown",
		"machine": platform.machine(),
	}
