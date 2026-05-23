from __future__ import annotations

import numpy as np

try:
	from tensorflow.keras import Sequential
	from tensorflow.keras.layers import Dense, LSTM
except Exception:  # pragma: no cover
	Sequential = None
	Dense = None
	LSTM = None


class LstmForecaster:
	def __init__(
		self,
		sequence_length: int,
		hidden_units: int = 64,
		learning_rate: float = 1e-3,
	) -> None:
		self.sequence_length = sequence_length
		self.hidden_units = hidden_units
		self.learning_rate = learning_rate
		self.model = None

	def build(self) -> None:
		if Sequential is None:
			raise ImportError("TensorFlow is required for LSTM model")
		from tensorflow.keras.optimizers import Adam

		model = Sequential(
			[
				LSTM(self.hidden_units, input_shape=(self.sequence_length, 1)),
				Dense(1),
			]
		)
		model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
		self.model = model

	def fit(
		self,
		x_train: np.ndarray,
		y_train: np.ndarray,
		x_val: np.ndarray | None = None,
		y_val: np.ndarray | None = None,
		epochs: int = 20,
		batch_size: int = 64,
		verbose: int = 0,
	):
		if self.model is None:
			self.build()
		validation_data = None
		if x_val is not None and y_val is not None:
			validation_data = (x_val, y_val)
		history = self.model.fit(
			x_train,
			y_train,
			validation_data=validation_data,
			epochs=epochs,
			batch_size=batch_size,
			verbose=verbose,
		)
		return history

	def predict(self, x: np.ndarray) -> np.ndarray:
		if self.model is None:
			raise RuntimeError("Model must be fit before prediction")
		preds = self.model.predict(x, verbose=0)
		return preds.reshape(-1)
