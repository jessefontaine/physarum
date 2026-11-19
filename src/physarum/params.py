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
class AgentParams:
    """Settings for how agents sense, turn, move and deposit.

    Attributes
    ----------
    count : int
        Number of agents. Fixed once the simulation is built.
    step_size : float
        Distance each agent moves per frame, in cells. Typical: 1.
    deposit_amount : float
        Pheromone added to the agent's cell after each move.
    sensor_distance : float
        How far ahead the sensors sit, in cells. Larger values give
        coarser networks. Typical: 5-15.
    sensor_angle : float
        Angle between the front sensor and each side sensor, in radians.
        Typical: pi/8 to pi/4 (22.5-45 degrees).
    turn_speed : float
        How far an agent turns per frame when steering, in radians.
        Typical: pi/8 to pi/4.
    """

    count: int
    step_size: float
    deposit_amount: float
    sensor_distance: float
    sensor_angle: float
    turn_speed: float


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
    agent : AgentParams
        Sensing, movement and deposit settings.
    seed : int
        Seed for the random generator, so runs are reproducible.
    """

    grid_height: int
    grid_width: int
    trail: TrailParams
    agent: AgentParams
    seed: int

    @property
    def size(self) -> tuple[int, int]:
        """Grid shape as (height, width)"""
        return self.grid_height, self.grid_width
