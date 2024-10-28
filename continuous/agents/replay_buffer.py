from __future__ import annotations
import numpy as np
import os
import random
import pickle
import lzma
from typing import Any, Optional, Sequence
from numpy.typing import NDArray
from d3rlpy.dataset import MDPDataset
from utils.random import Generators

class ReplayBuffer:
    """Uniform replay buffer. Allocates all required memory at initialization."""
    _data: Optional[Sequence[NDArray]]
    _capacity: int
    _num_added: int
    _np_rng: np.random.Generator

    def __init__(self, capacity: int, rng: np.random.Generator):
        """Initializes a new `Replay`.

        Args:
            capacity: The maximum number of items allowed in the replay. Adding
            items to a replay that is at maximum capacity will overwrite the oldest
            items.
        """
        self._data = None
        self._capacity = capacity
        self._num_added = 0
        self._np_rng = rng

    def add(self, items: Sequence[Any]):
        """Adds a single sequence of items to the replay.

        Args:
            items: Sequence of items to add. Does not handle batched or nested items.
        """
        if self._data is None:
            self._preallocate(items)

        for slot, item in zip(self._data, items):
            slot[self._num_added % self._capacity] = item

        self._num_added += 1

    def sample(self, size: int) -> Sequence[NDArray]:
        """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
        #indices = np.random.randint(self.size, size=size)
        #indices = np.random.randint(self.size, size=size)
        indices = np.zeros(size, dtype=np.int64, order='C')
        indices[1:] = self._np_rng.choice(self.size, size=size-1, replace=False)
        indices[0] = (self._num_added - 1) % self._capacity
        return [slot[indices] for slot in self._data]

    def reset(self,):
        """Resets the replay."""
        self._data = None

    @property
    def size(self) -> int:
        return min(self._capacity, self._num_added)

    @property
    def fraction_filled(self) -> float:
        return self.size / self._capacity

    def _preallocate(self, items: Sequence[Any]):
        """Assume flat structure of items."""
        as_array = []
        for item in items:
            if item is None:
                raise ValueError('Cannot store `None` objects in replay.')
            as_array.append(np.asarray(item))

        self._data = [np.zeros(dtype=x.dtype, shape=(self._capacity,) + x.shape)
                        for x in as_array]
    
    def to_mdpdataset(self, coeff: NDArray[np.float64], size: int = None) -> MDPDataset:
        rewards: NDArray[np.float64] = np.multiply(coeff[None,:], self._data[2]).sum(-1)
        size = rewards.shape[0] if size is None else size
        return MDPDataset(
            observations=self._data[0][-size:].copy(),
            actions=self._data[1][-size:].copy(),
            rewards=rewards[-size:].copy(),
            terminals=self._data[3][-size:].copy()
        )