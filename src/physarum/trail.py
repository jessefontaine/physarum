import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter

from physarum.params import TrailParams


class TrailMap:
    """A 2D pheromone grid that diffuses and decays over time.

    Parameters
    ----------
    size : tuple[int, int]
        Grid shape as (height, width).
    params : TrailParams
        Decay and diffusion settings. Reads every step, so changes take effect 
        immediately.

    Attributes
    ----------
    grid : NDArray[float32]
        Trail intensity, shape (height, width), indexed as grid[y, x].
    """

    def __init__(
        self,
        size: tuple[int, int],
        params: TrailParams,
    ) -> None:
        self.size: tuple[int, int] = size
        self.params: TrailParams = params
        self.grid: npt.NDArray[np.float32] = np.zeros(shape=self.size, dtype=np.float32)

    def step(self) -> None:
        """Advance one frame: diffuse, then decay, in place."""
        self._diffusion()
        self._decay()

    def _diffusion(self) -> None:
        """Blur the grid with a 3x3 box filter."""
        blurred: npt.NDArray[np.float32] = uniform_filter(self.grid, size=3, mode='nearest')
        diffused: npt.NDArray[np.float32] = self.grid + self.params.diffusion_rate * (blurred - self.grid)
        self.grid = diffused

    def _decay(self) -> None:
        """Multiply the grid by a single scalar, in place."""
        self.grid *= self.params.decay_rate