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

        # create trail map
        self.trail_map: TrailMap = TrailMap(
            size=self.params.size,
            params=self.params.trail
        )

    def step(self) -> None:
        """Advance one frame."""
        self.trail_map.step()