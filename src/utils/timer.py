from __future__ import annotations

from contextlib import ContextDecorator
from dataclasses import dataclass
import time


@dataclass
class TimerResult:
	name: str
	elapsed_seconds: float


class Timer(ContextDecorator):
	def __init__(self, name: str = "timer") -> None:
		self.name = name
		self._start: float | None = None
		self.elapsed_seconds = 0.0

	def __enter__(self) -> "Timer":
		self._start = time.perf_counter()
		return self

	def __exit__(self, exc_type, exc, exc_tb) -> None:
		if self._start is not None:
			self.elapsed_seconds = time.perf_counter() - self._start

	@property
	def result(self) -> TimerResult:
		return TimerResult(self.name, self.elapsed_seconds)
