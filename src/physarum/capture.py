

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
    """Dict to hold simulation changes in.

    Attributes
    ----------
    frame : int
        Framenumber when change was made.
    key : str
        Parameter key that was changed.
    value : float | int
        New parameter value.
    """
    frame: int
    key: str
    value: float | int


def upscale_grid(grid: npt.NDArray, upscale: int = 1) -> npt.NDArray:
    """Upscale numpy array by whole factors.

    Uses nearest-neighbour interpolation.
    
    Parameters
    ----------
    grid | ndarray
        2-Dimensional numpy array to upscale.
    upscale | int
        Upscaling factor. Whole numbers, default 1.

    Returns
    -------
    ndarray
        Upscaled array.
    """
    return grid.repeat(repeats=upscale, axis=0).repeat(repeats=upscale, axis=1)


class Recorder:
    """Record simualtion frames and save buffer to disk.
    
    Stored images of gifs receive a sidecar with the relevant simulation parameters and 
    changes.

    Parameters
    ----------
    params : SimParams
        Simulation starting parameters.
    vmax : float
        Value at which colormap maxes out. Default 0.1.
    buffer_len : int
        Number of buffered simulation frames. Default 100.
    out_dir : str | Path
        Output directory to save to. Default `captures`. Will be created if it does not 
        exist.
    upscale : int
        Nearest neighbour upscaling factor. Default 1.

    Attributes
    ----------
    frames : list[tuple[int, ndarray[uint8]]]
        The frame buffer, in normalized values with respect to vmax. Contains tuples of 
        (frame number, frame array).
    changes : list[Change]
        List of recorded changes.
    """

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

        # buffer is a deque. attribute is inaccessible to prevent issues when it is 
        # edited while the sim is running
        self._deque: collections.deque = collections.deque(maxlen=self.buffer_len)
        self._changes: list[Change] = []

    def push(
            self,
            frame: int,
            grid: npt.NDArray[np.float32],
    ) -> None:
        """Add frame to the buffer.

        Grid is normalized to vmax and cast to uint8.

        Parameters
        ----------
        frame : int
            Frame number.
        grid : ndarray[float32]
            Grid at that frame.
        """
        self._deque.append(
            (frame, (np.clip(grid / self.vmax, 0, 1) * 255).round().astype(np.uint8))
        )

    @property
    def frames(self) -> list[tuple[int, npt.NDArray[np.uint8]]]:
        """Framebuffer as list[tuple[int, ndarray[uint8]]]."""
        return list(self._deque)

    def note_change(
            self,
            frame: int,
            key: str,
            value: float | int,
    ) -> None:
        """Record a change in simulation parameters.
        
        Parameters
        ----------
        frame : int
            Framenumber when change was made.
        key : str
            Simulation parameter that was changed.
        value : float | int
            New value for parameter.
        """
        self._changes.append(Change(frame=frame, key=key, value=value))

    @property
    def changes(self) -> list[Change]:
        """Record of made changes as list[Change]."""
        return self._changes

    def save_png(
            self,
            colormap: str,
    ) -> Path | None:
        """Save the newest recorded frame in the buffer to a PNG.

        Saves TOML sidecar alongside image with capture parameters.

        Parameters
        ----------
        colormap : str
            Matplotlib registered colormap to use.

        Returns
        -------
        Path | None
            Path of saved image, or none when the buffer was empty.
        """
        # early exit on empty buffer
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

        # create filepaths
        fn: Path = self._make_filename(start=frame, end=frame)
        fn_png: Path = fn.with_suffix(".png")
        fn_png.parent.mkdir(parents=True, exist_ok=True)

        # save image and sidecar
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
        """Save the current buffered frames to disk as a GIF.

        Saves TOML sidecar alongside image with capture parameters.

        Parameters
        ----------
        colormap : str
            Matplotlib registered colormap to use.
        fps : int, optional
            Framerate of saved gif, by default 20

        Returns
        -------
        Path | None
            Path of saved image, or none when the buffer was empty.
        """
        if not self.frames:
            return None

        # get all the frames in the buffer
        frames, grids = zip(*self.frames)

        # upscale in both directions
        if self.upscale > 1:
            grids = [upscale_grid(grid, self.upscale) for grid in grids]

        # convert buffer to images and apply colormap
        imgs = [
            Image.fromarray(grid, mode="P") for grid in grids
        ]
        [img.putpalette(lut(colormap).flatten().tolist()) for img in imgs]

        # create filepaths
        fn: Path = self._make_filename(start=frames[0], end=frames[-1])
        fn_gif: Path = fn.with_suffix(".gif")
        fn_gif.parent.mkdir(parents=True, exist_ok=True)

        # save gif and sidecar
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
        """Find available filename according to template.

        Fills according to template:
        `YYMMDD-HHMMSS_f{start}-{end}`
        Duplicate filenames get suffix `_1`, `_2`, etc.

        Parameters
        ----------
        start : int
            Starting framenumber of image.
        end : int
            End framenumber of image.

        Returns
        -------
        Path
            Filename to store the image and sidecar under.
        """
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
            fps: int | None = None,
    ) -> None:
        """Write sidecar with the capture parameters.

        Parameters
        ----------
        path : str | Path
            Path to store file to. Must have extenstion '.toml'.
        kind : str
            Either 'png' or 'gif'.
        start : int
            Starting frame of buffer.
        end : int
            Ending frame of buffer.
        colormap : str
            Colormap used in stored image.
        fps : int, optional
            Framerate of gif, only used when kind is gif, by default None
        """

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