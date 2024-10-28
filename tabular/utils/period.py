from abc import ABC, abstractmethod
import numpy as np

class Period(ABC):
    def __init__(self, initial_period: int, final_period: int) -> None:
        super().__init__()
        self.initial_period = initial_period
        self.final_period = int(np.ceil(final_period))
        self.num_updates = 0
        self.current_value = initial_period

    def get(self) -> int:
        return self.current_value

    def update(self, step: int) -> None:
        self.current_value = min(self.final_period, self._update(step))
        self.num_updates += 1
    
    @abstractmethod
    def _update(self, step: int) -> int:
        raise NotImplementedError('method not implemented')


class ConstantPeriod(Period):
    def __init__(self, period: int) -> None:
        super().__init__(period, period)

    def _update(self, step: int) -> int:
        return self.final_period


class LinearPeriod(Period):
    def __init__(self, initial_period: int, slope: float, total_updates: int) -> None:
        final_period = initial_period + total_updates * slope
        super().__init__(initial_period, final_period)
        self.initial_period = initial_period
        self.slope = slope
        self.total_updates = total_updates

    def _update(self, step: int) -> int:
        y0 = self.initial_period
        yf = self.final_period
        value = y0 + (yf - y0) * self.num_updates / self.total_updates
        return int(np.ceil(value))

class ExponentialPeriod(Period):
    def __init__(self, initial_period: int, factor: float, final_period: int) -> None:
        super().__init__(initial_period, final_period)
        self.initial_period = initial_period
        self.factor = factor

    def _update(self, step: int) -> int:
        # log_2 y(k) =  log_2(y0) + num_updates * factor * log_2(2)
        y0 = np.log2(self.initial_period)
        y1 = self.num_updates * self.factor
        y = 2 ** (y0 + y1)
        return np.ceil(y)