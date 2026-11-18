from physarum.params import SimParams, TrailParams
from physarum.simulation import Simulation
from physarum.render import run_matplotlib

def main() -> None:
    params = SimParams(
        grid_height=400,
        grid_width=400,
        trail=TrailParams(
            decay_rate=0.95,
            diffusion_rate=1.,
        ),
    )

    sim = Simulation(params)

    sim.trail_map.grid[95:105, 95:105] = 1.

    run_matplotlib(sim)
