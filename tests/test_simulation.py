import numpy as np
import pytest
from numpy.testing import assert_allclose

from physarum.agents import Agents
from physarum.params import AgentParams, SimParams, TrailParams
from physarum.simulation import Simulation
from physarum.trail import TrailMap


def _sim_params(count: int, seed: int = 0) -> SimParams:
    """Non-square settings, so a swapped height and width shows up."""
    return SimParams(
        grid_height=4,
        grid_width=6,
        trail=TrailParams(decay_rate=0.5, diffusion_rate=1.0),
        agent=AgentParams(
            count=count,
            step_size=1.0,
            deposit_amount=1.0,
            sensor_distance=2.0,
            sensor_angle=np.pi / 4,
            turn_speed=0.3,
        ),
        seed=seed,
    )


@pytest.fixture
def params():
    """No agents, so a step only runs the trail map and results stay exact."""
    return _sim_params(count=0)


@pytest.fixture
def agent_params():
    """A few agents on the same small grid."""
    return _sim_params(count=10)


# --- params ----------------------------------------------------------------


def test_size_is_height_then_width(params):
    assert params.size == (4, 6)


# --- construction ----------------------------------------------------------


def test_trail_map_has_grid_size_from_params(params):
    sim = Simulation(params)

    assert isinstance(sim.trail_map, TrailMap)
    assert sim.trail_map.grid.shape == (4, 6)


def test_new_simulation_starts_with_empty_trail(params):
    sim = Simulation(params)

    assert not sim.trail_map.grid.any()


def test_trail_map_shares_the_trail_params_object(params):
    """Sharing the object is what lets sliders change settings while running."""
    sim = Simulation(params)

    assert sim.trail_map.params is params.trail


# --- step ------------------------------------------------------------------


def test_step_advances_trail_like_a_standalone_trail_map(params):
    start = np.zeros(params.size, dtype=np.float32)
    start[1, 2] = 9.0

    sim = Simulation(params)
    sim.trail_map.grid = start.copy()
    reference = TrailMap(params.size, params.trail)
    reference.grid = start.copy()

    sim.step()
    reference.step()

    assert_allclose(sim.trail_map.grid, reference.grid)


def test_step_on_empty_simulation_stays_empty(params):
    sim = Simulation(params)

    sim.step()

    assert not sim.trail_map.grid.any()


def test_step_picks_up_params_changed_while_running(params):
    sim = Simulation(params)
    sim.trail_map.grid[1, 2] = 8.0

    params.trail.diffusion_rate = 0.0
    params.trail.decay_rate = 0.25
    sim.step()

    assert sim.trail_map.grid[1, 2] == pytest.approx(2.0)
    assert sim.trail_map.grid.sum() == pytest.approx(2.0)


def test_many_steps_keep_shape_dtype_and_finite_values(params):
    sim = Simulation(params)
    sim.trail_map.grid[:] = np.random.default_rng(0).random(params.size)

    for _ in range(50):
        sim.step()

    assert sim.trail_map.grid.shape == (4, 6)
    assert sim.trail_map.grid.dtype == np.float32
    assert np.isfinite(sim.trail_map.grid).all()
    assert (sim.trail_map.grid >= 0).all()


# --- agents ----------------------------------------------------------------


def test_agents_are_built_from_params(agent_params):
    sim = Simulation(agent_params)

    assert isinstance(sim.agents, Agents)
    assert sim.agents.ys.shape == (10,)
    assert sim.agents.size == (4, 6)


def test_agents_share_the_agent_params_object(agent_params):
    """Sharing the object is what lets sliders change settings while running."""
    sim = Simulation(agent_params)

    assert sim.agents.params is agent_params.agent


def test_one_step_deposits_then_decays(agent_params):
    """Diffusion conserves the total, so only deposit and decay change it."""
    sim = Simulation(agent_params)

    sim.step()

    expected = 10 * agent_params.agent.deposit_amount * agent_params.trail.decay_rate
    assert sim.trail_map.grid.sum() == pytest.approx(expected, rel=1e-5)


def test_same_seed_gives_identical_runs():
    first = Simulation(_sim_params(count=10, seed=3))
    second = Simulation(_sim_params(count=10, seed=3))

    for _ in range(20):
        first.step()
        second.step()

    assert_allclose(first.trail_map.grid, second.trail_map.grid)
    assert_allclose(first.agents.heading, second.agents.heading)


def test_different_seeds_place_agents_differently():
    first = Simulation(_sim_params(count=10, seed=3))
    second = Simulation(_sim_params(count=10, seed=4))

    assert not np.allclose(first.agents.xs, second.agents.xs)


def test_many_steps_with_agents_stay_finite_and_in_bounds(agent_params):
    sim = Simulation(agent_params)

    for _ in range(50):
        sim.step()

    assert np.isfinite(sim.trail_map.grid).all()
    assert (sim.trail_map.grid >= 0).all()
    assert ((sim.agents.ys >= 0) & (sim.agents.ys < 4)).all()
    assert ((sim.agents.xs >= 0) & (sim.agents.xs < 6)).all()
