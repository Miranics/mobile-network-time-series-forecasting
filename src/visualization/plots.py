from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_pdf(total_traffic: pd.Series, title: str = "Traffic PDF"):
	fig, ax = plt.subplots(figsize=(8, 4))
	sns.kdeplot(total_traffic.values, fill=True, ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Total internet traffic per square")
	ax.set_ylabel("Density")
	fig.tight_layout()
	return fig, ax


def plot_three_area_series(df: pd.DataFrame, title: str = "Two-week traffic"):
	fig, ax = plt.subplots(figsize=(12, 4))
	for square_id, part in df.groupby("square_id"):
		ax.plot(part["time_interval"], part["internet_traffic"], label=f"Square {square_id}")
	ax.set_title(title)
	ax.set_xlabel("Time")
	ax.set_ylabel("Internet traffic")
	ax.legend()
	fig.tight_layout()
	return fig, ax


def plot_heatmap(matrix: np.ndarray, title: str = "Spatial Traffic Heatmap"):
	fig, ax = plt.subplots(figsize=(8, 7))
	sns.heatmap(matrix, cmap="magma", ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Grid column")
	ax.set_ylabel("Grid row")
	fig.tight_layout()
	return fig, ax


def save_figure(fig, output_path: Path, dpi: int = 180) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
