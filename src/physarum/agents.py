import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from physarum.params import AgentParams


class Agents:
    """A population of agents that follow and put down pheromone trails.

    Positions of agents are floats, so they keep their sub-cell position between frames. 
    They are truncated to grid cells when sensing or depositing.
    Agents wrap around grid edges.

    Parameters
    ----------
    size : tuple[int, int]
        Grid shape as (height, width), in cells.
    rng : Generator
        Random generator for start positions and random turns.
    params : AgentParams
        Sensing, movement and deposit settings. Can be updated live.

    Attributes
    ----------
    ys, xs : ndarray of float32, shape (N,)
        Agent positions in cells, within [0, height) and [0, width).
    heading : ndarray of float32, shape(N,)
        Direction of travel in radians, within [0, 2*pi).
    """

    def __init__(
        self, 
        size: tuple[int, int],
        rng: Generator,
        params: AgentParams,
    ) -> None:

        self.size: tuple[int, int] = size
        self.params: AgentParams = params
        self.rng = rng

        # create random agents
        self.ys: npt.NDArray[np.float32] = self.size[0] * rng.random(self.params.count, dtype=np.float32)
        self.xs: npt.NDArray[np.float32] = self.size[1] * rng.random(self.params.count, dtype=np.float32)
        self.heading: npt.NDArray[np.float32] = 2* np.pi * rng.random(self.params.count, dtype=np.float32)

    def move(self) -> None:
        """Step every agent forward along its heading, wrap at edges."""

        # get unit vectors in each direction
        dy: npt.NDArray[np.float32] = np.sin(self.heading)
        dx: npt.NDArray[np.float32] = np.cos(self.heading)

        # add the step to the agents position
        self.ys += self.params.step_size * dy
        self.xs += self.params.step_size * dx

        # wrap at the edges
        self.ys %= self.size[0]
        self.xs %= self.size[1]

    def deposit(self, grid: npt.NDArray[np.float32]) -> None:
        """Add pheromone to the cell under each agent, in place.

        Agents that share a cell each add a full deposit.

        Parameters
        ----------
        grid : ndarray of float32, shape (height, width)
            Trail map to deposit on. Modified in place.
        """

        # cast position to ints, and wrap them to prevent out of bounds issues
        ys: npt.NDArray[np.intp] = self.ys.astype(np.intp) % self.size[0]
        xs: npt.NDArray[np.intp] = self.xs.astype(np.intp) % self.size[1]

        # deposit on a view of grid
        np.add.at(grid, (ys, xs), self.params.deposit_amount)

    def sense(self, grid: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Read the trail map at each agent's three sensors.

        Parameters
        ----------
        grid : ndarray of float32, shape (height, width)
            Trail map to sample.

        Returns
        -------
        ndarray of float32, shape (3, N)
            Readings at the left, front and right sensors, in that order.
        """

        # keep as a shorthand to use in the loop
        d: float = self.params.sensor_distance

        readings: list[npt.NDArray[np.float32]] = []
        for offset in (self.params.sensor_angle, 0., -self.params.sensor_angle):

            # check at current angle
            angle: npt.NDArray[np.float32] = self.heading + offset

            # get sensing locations
            # since we are sensing at indices on the grid, we cast to int, and wrap to 
            # prevent out of bounds errors
            sy: npt.NDArray[np.intp] = (self.ys + d * np.sin(angle)).astype(np.intp) % self.size[0]
            sx: npt.NDArray[np.intp] = (self.xs + d * np.cos(angle)).astype(np.intp) % self.size[1]

            # now we sample the grid
            readings.append(grid[sy, sx])
        return np.stack(readings)


    def turn(self, readings: npt.NDArray[np.float32]) -> None:
        """Steer each agent toward the strongest trail reading.

        An agent turns toward the stronger side sensor, or goes straight
        if the front is strongest. If the front is weakest and the sides
        tie, it turns in a random direction.

        Parameters
        ----------
        readings : ndarray of float32, shape (3, N)
            Left, front and right readings, as returned by `sense`.
        """

        # get the readings at the three directions
        left, front, right = readings

        # random turn direction per agent, only used if the sides tie
        random_dir: npt.NDArray[np.float32] = self.rng.choice(np.array([-1., 1.], dtype=np.float32), size=front.shape)

        # steering directions for all agents
        # we steer towards where the scent is strongest, and go in a random direction
        # when front is weakest and left and right are equal
        # when all are equal, we keep going straight
        steer: npt.NDArray[np.float32] = np.zeros_like(front)
        steer = np.where((front < left) & (front < right), random_dir, steer)
        steer = np.where((left > front) & (left > right), 1.0, steer)
        steer = np.where((right > front) & (right > left), -1.0, steer)

        # change heading based on direction, wrap at 2 pi
        self.heading += steer * self.params.turn_speed
        self.heading %= 2 * np.pi
