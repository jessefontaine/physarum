import numpy as np
import numpy.typing as npt
from matplotlib import colormaps, colors

CYCLE: tuple[str, ...] = (
    "magma", "inferno", "twilight_shifted", "cividis", "bone", "physarum"
)
AMBER: str = "#FFBF00"


def register() -> None:
    # only add it when it does not exist yet
    if "physarum" not in colormaps:
        cmap = colors.LinearSegmentedColormap.from_list("physarum", ["black", AMBER, "white"])
        colormaps.register(cmap)


def lut(name: str) -> npt.NDArray[np.uint8]:
    return colormaps[name](np.arange(256), bytes=True)[:,:3]


def to_rgb(frame: npt.NDArray[np.uint8], name: str) -> npt.NDArray[np.uint8]:
    return lut(name=name)[frame]


def next_name(name: str, step: int = 1) -> str:
    return CYCLE[(CYCLE.index(name) + step) % len(CYCLE)]


register()