

import collections
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import tomli_w
from PIL import Image

from physarum.colormaps import lut
from physarum.params import SimParams


class Change(TypedDict):
    frame: int
    key: str
    value: float | int


def upscale_grid(grid: npt.NDArray, upscale: int = 1) -> npt.NDArray:
    return grid.repeat(repeats=upscale, axis=0).repeat(repeats=upscale, axis=1)


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
        self.out_dir: Path = Path(out_dir)
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

    def save_png(
            self,
            colormap: str,
    ) -> Path | None:
        if not self.frames:
            return None
        
        # take the final frame
        frame, grid = self.frames[-1]

        # upscale in both dimensions
        if self.upscale > 1:
            grid = upscale_grid(grid, self.upscale)

        # apply colormap
        img = Image.fromarray(grid, mode="P")
        img.putpalette(lut(colormap).flatten().tolist())

        # save
        fn: Path = self._make_filename(start=frame, end=frame)
        fn_png: Path = fn.with_suffix(".png")
        fn_png.parent.mkdir(parents=True, exist_ok=True)

        img.save(fn_png)
        self._write_sidecar(
            fn.with_suffix(".toml"),
            kind="png",
            start=frame,
            end=frame,
            colormap=colormap,
        )
        return fn_png

    def save_gif(
            self,
            colormap: str,
            fps: int = 20,
    ) -> Path | None:
        if not self.frames:
            return None

        # get all the frames in the buffer
        frames, grids = zip(*self.frames)

        # upscale in both directions
        if self.upscale > 1:
            grids = [upscale_grid(grid, self.upscale) for grid in grids]

        # apply colormap
        imgs = [
            Image.fromarray(grid, mode="P") for grid in grids
        ]
        [img.putpalette(lut(colormap).flatten().tolist()) for img in imgs]

        # save
        fn: Path = self._make_filename(start=frames[0], end=frames[-1])
        fn_gif: Path = fn.with_suffix(".gif")
        fn_gif.parent.mkdir(parents=True, exist_ok=True)

        imgs[0].save(fn_gif, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0)
        self._write_sidecar(
            fn.with_suffix(".toml"),
            kind="gif",
            start=frames[0],
            end=frames[-1],
            colormap=colormap,
            fps=fps,
        )
        return fn_gif

    def _make_filename(self, start: int, end: int) -> Path:
        stamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        fn_base = f"{stamp}_f{start}-{end}"

        # check if that name exists, and find a safe name
        suffix: int = 0
        fn: str = fn_base
        while any(self.out_dir.glob(f"{fn}.*")):
            suffix += 1
            fn = f"{fn_base}_{suffix}"
        return self.out_dir / fn

    def _write_sidecar(
            self,
            path: str | Path,
            kind: str,
            start: int,
            end: int,
            colormap: str,
            fps=None,
    ) -> None:

        # get the capture metadata
        capture: dict[str, Any] = {
            "kind": kind,
            "frame_start": start,
            "frame_end": end,
            "colormap": colormap,
            "vmax": self.vmax,
            "created": datetime.now().replace(microsecond=0),
            "reproducible": not self.changes,
            "changes": [c for c in self.changes if c["frame"] <= end]
        }
        if fps:
            capture["fps"] = fps
        metadata: dict[str, Any] = self.start_params.to_dict() | {"capture": capture}
        
        with open(path, "wb") as f:
            tomli_w.dump(metadata, f)