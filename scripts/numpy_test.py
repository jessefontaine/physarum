from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import uniform_filter


@dataclass
class TrailParams:
    """Settings for how the trail map evolves each frame.

    Attributes
    ----------
    decay_rate : float
        Multiplier applied to every cell per frame [0, 1]. Lower values make trails fade
        faster. Typical: 0.9-0.99.
    diffusion_rate : float
        How much the blurred grid is mixed per frame [0, 1]. 
        0 means no spreading, 1 means a full 3x3 blur.
    """

    decay_rate: float
    diffusion_rate: float


class TrailMap:
    """A 2D pheromone grid that diffuses and decays over time.

    Parameters
    ----------
    size : tuple[int, int]
        Grid shape as (height, width).
    params : TrialParams
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
        """Blur the grid with a 3x3 box filter, in place."""
        blurred: npt.NDArray[np.float32] = uniform_filter(self.grid, size=3, mode='nearest')
        diffused: npt.NDArray[np.float32] = self.grid + self.params.diffusion_rate * (blurred - self.grid)
        self.grid = diffused

    def _decay(self) -> None:
        """Multiply the grid by a single scalar, in place."""
        self.grid *= self.params.decay_rate


@dataclass
class SimParams:
    """All settings for one simulation run.

    Attributes
    ----------
    grid_height : int
        Trail map rows, in cells. Fixed once the simulation is built.
    grid_width : int
        Trail map columns, in cells. Fixed once the simulation is built.
    trail : TrailParams
        Diffusion and decay settings, adjustable while running.
    """

    grid_height: int
    window_width: int
    trail: TrailParams

    @property
    def size(self) -> tuple[int, int]:
        """Grid shape as (height, width)"""
        return self.grid_height, self.window_width


class Simulation:
    """Orchestrate the Physarum simulation.

    Parameters
    ----------
    params : SimParams
        Settings for running the simulation.

    Attributes
    ----------
    trail_map : TrailMap
        2D pheromone grid that diffuses and decays over time.
    """

    def __init__(self, params: SimParams) -> None:

        self.params: SimParams = params

        # create trail map
        self.trail_map: TrailMap = TrailMap(
            size=self.params.size,
            params=self.params.trail
        )

    def step(self) -> None:
        """Advance one frame."""
        self.trail_map.step()


def run_matplotlib(sim: Simulation, vmax: float = 0.1) -> None:
    """Show the simulation live in a matplotlib window until it is closed.

    Parameters
    ----------
    sim : Simulation
        Stepped once per frame.
    vmax : float
        Trail value shown as full brightness. Default 0.1.
    """

    # create figure
    fig, axes = plt.subplots()
    img = axes.imshow(sim.trail_map.grid, vmax=vmax)

    # create animation function
    def animation_func(frame: int) -> None:
        sim.step()
        img.set_data(sim.trail_map.grid)

    # setup the animation
    _animation = FuncAnimation(
        fig,
        animation_func,
        cache_frame_data=False,
        interval=50,
    )

    plt.show()


if __name__ == "__main__":
    params = SimParams(
        grid_height=400,
        window_width=400,
        trail=TrailParams(
            decay_rate=0.95,
            diffusion_rate=0.,
        ),
    )

    sim = Simulation(params)

    sim.trail_map.grid[95:105, 95:105] = 1.

    run_matplotlib(sim)