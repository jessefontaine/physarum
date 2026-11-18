import numpy as np
import pytest
from numpy.testing import assert_allclose

from physarum.params import SimParams, TrailParams
from physarum.simulation import Simulation
from physarum.trail import TrailMap


@pytest.fixture
def params():
    """Non-square settings, so a swapped height and width shows up."""
    return SimParams(
        grid_height=4,
        grid_width=6,
        trail=TrailParams(decay_rate=0.5, diffusion_rate=1.0),
    )


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
