import numpy as np
import numpy.typing as npt
from matplotlib import colormaps, colors

CYCLE: tuple[str, ...] = (
    "magma", "inferno", "twilight_shifted", "cividis", "bone", "physarum"
)
AMBER: str = "#FFBF00"


def register() -> None:
    """Register physarum colormap to matplotlib."""
    # only add it when it does not exist yet
    if "physarum" not in colormaps:
        cmap = colors.LinearSegmentedColormap.from_list("physarum", ["black", AMBER, "white"])
        colormaps.register(cmap)


def lut(name: str) -> npt.NDArray[np.uint8]:
    """Lookup table for matplotlib colormaps.
    
    Parameters
    ----------
    name : str
        Name of a matplotlib registered colomap.

    Returns
    -------
    ndarray[uint8], shape (256, 3)
        RGB colorvalues of 256 shades.
    """
    return colormaps[name](np.arange(256), bytes=True)[:,:3]


def to_rgb(frame: npt.NDArray[np.uint8], name: str) -> npt.NDArray[np.uint8]:
    """Convert array frame to rgb values by colormap.

    Parameters
    ----------
    frame : ndarray[uint8]
        Frame of values ranging from [0, 256).
    name : str
        Name of colormap to use.

    Returns
    -------
    ndarray[uint8], shape (*frame.shape, 3)
        Frame converted to RGB.
    """
    return lut(name=name)[frame]


def next_name(name: str, step: int = 1) -> str:
    """Cycle to the next colormap name of physarum/matplotlib colormaps.

    Cycles through colormap names in `physarum.colormaps.CYCLE`.
    
    Parameters
    ----------
    name : str
        Name of the current colormap.
    step : int
        Stepsize to take in the cycle, default 1.

    Returns
    -------
    str
        Name of next colormap
    """
    return CYCLE[(CYCLE.index(name) + step) % len(CYCLE)]

# register the new colormap on module import
register()