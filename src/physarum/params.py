from dataclasses import dataclass


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
    grid_width: int
    trail: TrailParams

    @property
    def size(self) -> tuple[int, int]:
        """Grid shape as (height, width)"""
        return self.grid_height, self.grid_width