from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_overlay(
	timestamps: pd.Index | pd.Series,
	y_true,
	y_pred,
	model_name: str,
	square_id: int,
):
	fig, ax = plt.subplots(figsize=(12, 4))
	ax.plot(timestamps, y_true, label="Observed", linewidth=1.8)
	ax.plot(timestamps, y_pred, label="Predicted", linewidth=1.6, alpha=0.9)
	ax.set_title(f"{model_name} - Square {square_id} (Dec 16-22)")
	ax.set_xlabel("Time")
	ax.set_ylabel("Internet traffic")
	ax.legend()
	fig.tight_layout()
	return fig, ax
