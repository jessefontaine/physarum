from physarum.render import run_matplotlib
from physarum.simulation import Simulation


def main() -> None:
    
    sim = Simulation.from_file("presets/default_parameters.toml")
    run_matplotlib(sim)
