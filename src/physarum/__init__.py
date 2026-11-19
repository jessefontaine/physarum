from physarum.params import AgentParams, SimParams, TrailParams
from physarum.render import run_matplotlib
from physarum.simulation import Simulation


def main() -> None:
    params = SimParams(
        grid_height=200,
        grid_width=200,
        trail=TrailParams(
            decay_rate=0.875,
            diffusion_rate=0.075,
        ),
        agent=AgentParams(
            count=400,
            step_size=1,
            deposit_amount=1,
            sensor_distance=10.,
            sensor_angle=1.,
            turn_speed=0.07
        ),
        seed=1,
    )

    sim = Simulation(params)

    run_matplotlib(sim)
