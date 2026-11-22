

import collections
from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from physarum.params import SimParams


class Change(TypedDict):
    frame: int
    key: str
    value: float | int


class Recorder:
    def __init__(
        self, 
        params: SimParams, 
        vmax: float = 0.1, 
        buffer_len: int = 100, 
        out_dir: str | Path = "captures", 
        upscale: int = 1,
    ) -> None:
        self.start_params: SimParams = deepcopy(params)
        self.vmax = vmax
        self.buffer_len = buffer_len
        self.out_dir = out_dir
        self.upscale = upscale

        self._deque: collections.deque = collections.deque(maxlen=self.buffer_len)
        self._changes: list[Change] = []

    def push(
            self,
            frame: int,
            grid: npt.NDArray[np.float32],
    ) -> None:
        self._deque.append(
            (frame, (np.clip(grid / self.vmax, 0, 1) * 255).round().astype(np.uint8))
        )

    @property
    def frames(self) -> list[tuple[int, npt.NDArray[np.uint8]]]:
        return list(self._deque)

    def note_change(
            self,
            frame: int,
            key: str,
            value: float,
    ) -> None:
        self._changes.append(Change(frame=frame, key=key, value=value))

    @property
    def changes(self) -> list[Change]:
        return self._changes