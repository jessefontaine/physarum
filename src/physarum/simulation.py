from pathlib import Path
from typing import Self

from numpy.random import Generator, default_rng

from physarum.agents import Agents
from physarum.params import SimParams
from physarum.trail import TrailMap


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
        self.rng: Generator = default_rng(self.params.seed)

        # create trail map
        self.trail_map = TrailMap(
            size=self.params.size,
            params=self.params.trail
        )

        # create agents
        self.agents = Agents(
            size=self.params.size,
            rng=self.rng,
            params=self.params.agent,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls(SimParams.from_toml(path))

    def step(self) -> None:
        """Advance one frame."""
        readings = self.agents.sense(self.trail_map.grid)
        self.agents.turn(readings)
        self.agents.move()
        self.agents.deposit(self.trail_map.grid)
        self.trail_map.step()