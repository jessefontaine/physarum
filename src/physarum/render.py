import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from physarum.simulation import Simulation


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