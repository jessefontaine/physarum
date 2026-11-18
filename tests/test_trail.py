import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from physarum.params import TrailParams
from physarum.trail import TrailMap


@pytest.fixture
def make_trail():
    """Return a builder for a TrailMap with a chosen starting grid and rates.

    Rates default to 1.0, which means full blur and no decay, so each test
    switches on only the behaviour it checks.
    """

    def _make(grid, diffusion_rate=1.0, decay_rate=1.0):
        grid = np.asarray(grid, dtype=np.float32)
        params = TrailParams(decay_rate=decay_rate, diffusion_rate=diffusion_rate)
        trail = TrailMap(grid.shape, params)
        trail.grid = grid.copy()
        return trail

    return _make


@pytest.fixture
def centre_spot():
    """5x5 zeros with 9.0 in the middle, so a full 3x3 blur gives exactly 1.0 per cell."""
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[2, 2] = 9.0
    return grid


@pytest.fixture
def blurred_centre_spot():
    """What a full 3x3 blur turns centre_spot into."""
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[1:4, 1:4] = 1.0
    return grid


# --- construction ---------------------------------------------------------


def test_new_trail_map_is_all_zeros_float32_with_given_shape():
    trail = TrailMap((4, 6), TrailParams(decay_rate=0.9, diffusion_rate=0.5))

    assert trail.grid.shape == (4, 6)
    assert trail.grid.dtype == np.float32
    assert not trail.grid.any()


# --- diffusion -------------------------------------------------------------


def test_full_diffusion_spreads_centre_spot_evenly(
    make_trail, centre_spot, blurred_centre_spot
):
    trail = make_trail(centre_spot, diffusion_rate=1.0)

    trail._diffusion()

    assert_allclose(trail.grid, blurred_centre_spot, atol=1e-6)


def test_zero_diffusion_rate_leaves_grid_unchanged(make_trail, centre_spot):
    trail = make_trail(centre_spot, diffusion_rate=0.0)

    trail._diffusion()

    assert_allclose(trail.grid, centre_spot)


def test_half_diffusion_rate_is_halfway_between_original_and_blur(
    make_trail, centre_spot, blurred_centre_spot
):
    trail = make_trail(centre_spot, diffusion_rate=0.5)

    trail._diffusion()

    expected = (centre_spot + blurred_centre_spot) / 2
    assert_allclose(trail.grid, expected, atol=1e-6)


def test_diffusion_leaves_uniform_grid_unchanged(make_trail):
    grid = np.full((5, 7), 3.0, dtype=np.float32)
    trail = make_trail(grid, diffusion_rate=1.0)

    trail._diffusion()

    assert_allclose(trail.grid, grid, atol=1e-6)


@pytest.mark.parametrize("row, col", [(0, 0), (0, 3), (4, 6), (2, 0)])
def test_diffusion_conserves_total_at_edges_and_corners(make_trail, row, col):
    grid = np.zeros((5, 7), dtype=np.float32)
    grid[row, col] = 9.0
    trail = make_trail(grid, diffusion_rate=1.0)

    trail._diffusion()

    assert trail.grid.sum() == pytest.approx(9.0, rel=1e-5)


def test_corner_spot_spreads_only_into_corner_block(make_trail):
    """Edges use mode='nearest', so nothing wraps to the opposite side."""
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[0, 0] = 9.0
    trail = make_trail(grid, diffusion_rate=1.0)

    trail._diffusion()

    assert (trail.grid[:2, :2] > 0).all()
    outside = trail.grid.copy()
    outside[:2, :2] = 0.0
    assert not outside.any()


def test_diffusion_keeps_shape_and_dtype(make_trail, centre_spot):
    trail = make_trail(centre_spot, diffusion_rate=0.7)

    trail._diffusion()

    assert trail.grid.shape == (5, 5)
    assert trail.grid.dtype == np.float32


# --- decay -----------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, 0.5, 0.95, 1.0])
def test_decay_multiplies_every_cell_by_rate(make_trail, centre_spot, rate):
    trail = make_trail(centre_spot, decay_rate=rate)

    trail._decay()

    assert_allclose(trail.grid, centre_spot * rate, atol=1e-6)


def test_decay_keeps_zero_grid_at_zero(make_trail):
    trail = make_trail(np.zeros((3, 4)), decay_rate=0.5)

    trail._decay()

    assert not trail.grid.any()


def test_decay_keeps_dtype(make_trail, centre_spot):
    trail = make_trail(centre_spot, decay_rate=0.9)

    trail._decay()

    assert trail.grid.dtype == np.float32


# --- full step -------------------------------------------------------------


def test_step_diffuses_then_decays(make_trail, centre_spot, blurred_centre_spot):
    trail = make_trail(centre_spot, diffusion_rate=1.0, decay_rate=0.5)

    trail.step()

    assert_allclose(trail.grid, blurred_centre_spot * 0.5, atol=1e-6)


def test_step_uses_params_changed_after_construction(make_trail, centre_spot):
    trail = make_trail(centre_spot, diffusion_rate=1.0, decay_rate=1.0)

    trail.params.diffusion_rate = 0.0
    trail.params.decay_rate = 0.5
    trail.step()

    assert_allclose(trail.grid, centre_spot * 0.5, atol=1e-6)


def test_total_shrinks_each_step_and_fades_towards_zero(make_trail, centre_spot):
    trail = make_trail(centre_spot, diffusion_rate=0.5, decay_rate=0.9)

    totals = []
    for _ in range(200):
        trail.step()
        totals.append(trail.grid.sum())

    assert all(later < earlier for earlier, later in itertools.pairwise(totals))
    assert totals[-1] < 1e-6


def test_many_steps_stay_finite_and_non_negative(make_trail):
    rng = np.random.default_rng(0)
    grid = rng.random((16, 24), dtype=np.float32)
    trail = make_trail(grid, diffusion_rate=0.3, decay_rate=0.95)

    for _ in range(100):
        trail.step()

    assert np.isfinite(trail.grid).all()
    assert (trail.grid >= 0).all()
    assert trail.grid.shape == (16, 24)
    assert trail.grid.dtype == np.float32
