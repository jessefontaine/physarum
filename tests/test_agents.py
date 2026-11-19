import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from physarum.agents import Agents
from physarum.params import AgentParams

# Non-square, so a swapped height and width shows up.
SIZE = (20, 30)


def _params(count: int = 1, **overrides) -> AgentParams:
    settings = {
        "count": count,
        "step_size": 1.0,
        "deposit_amount": 1.0,
        "sensor_distance": 3.0,
        "sensor_angle": np.pi / 2,
        "turn_speed": 0.1,
    }
    settings.update(overrides)
    return AgentParams(**settings)


@pytest.fixture
def make_agents():
    """Return a builder for Agents placed at chosen positions and headings.

    Positions are given as (ys, xs, heading) lists. Any AgentParams field can
    be overridden by keyword.
    """

    def _make(ys, xs, heading, size=SIZE, seed=0, **overrides):
        params = _params(count=len(ys), **overrides)
        agents = Agents(size, default_rng(seed), params)
        agents.ys = np.asarray(ys, dtype=np.float32)
        agents.xs = np.asarray(xs, dtype=np.float32)
        agents.heading = np.asarray(heading, dtype=np.float32)
        return agents

    return _make


# --- construction ----------------------------------------------------------


def test_new_agents_have_one_float32_entry_per_agent():
    agents = Agents(SIZE, default_rng(0), _params(count=50))

    for values in (agents.ys, agents.xs, agents.heading):
        assert values.shape == (50,)
        assert values.dtype == np.float32


def test_new_agents_start_inside_grid_with_valid_headings():
    agents = Agents(SIZE, default_rng(0), _params(count=1000))

    assert ((agents.ys >= 0) & (agents.ys < SIZE[0])).all()
    assert ((agents.xs >= 0) & (agents.xs < SIZE[1])).all()
    assert ((agents.heading >= 0) & (agents.heading < 2 * np.pi)).all()


def test_new_agents_spread_over_the_whole_grid():
    """Catches height and width swapped when scaling the random positions."""
    agents = Agents(SIZE, default_rng(0), _params(count=1000))

    assert agents.ys.max() > SIZE[0] * 0.9
    assert agents.xs.max() > SIZE[1] * 0.9


def test_same_seed_gives_same_agents():
    first = Agents(SIZE, default_rng(7), _params(count=10))
    second = Agents(SIZE, default_rng(7), _params(count=10))

    assert_allclose(first.ys, second.ys)
    assert_allclose(first.heading, second.heading)


# --- move ------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading, dy, dx",
    [(0.0, 0.0, 1.0), (np.pi / 2, 1.0, 0.0), (np.pi, 0.0, -1.0), (3 * np.pi / 2, -1.0, 0.0)],
)
def test_move_steps_along_heading(make_agents, heading, dy, dx):
    """Heading 0 points along +x (columns), pi/2 along +y (rows)."""
    agents = make_agents([10.0], [10.0], [heading], step_size=2.0)

    agents.move()

    assert_allclose(agents.ys, [10.0 + 2 * dy], atol=1e-5)
    assert_allclose(agents.xs, [10.0 + 2 * dx], atol=1e-5)


def test_move_wraps_past_far_edges(make_agents):
    agents = make_agents([10.0, 19.5], [29.5, 10.0], [0.0, np.pi / 2])

    agents.move()

    assert_allclose(agents.xs[0], 0.5, atol=1e-5)
    assert_allclose(agents.ys[1], 0.5, atol=1e-5)


def test_move_wraps_past_zero_edges(make_agents):
    agents = make_agents([10.0, 0.25], [0.25, 10.0], [np.pi, 3 * np.pi / 2])

    agents.move()

    assert_allclose(agents.xs[0], 29.25, atol=1e-5)
    assert_allclose(agents.ys[1], 19.25, atol=1e-5)


def test_move_uses_step_size_changed_while_running(make_agents):
    agents = make_agents([10.0], [10.0], [0.0], step_size=1.0)

    agents.params.step_size = 3.0
    agents.move()

    assert_allclose(agents.xs, [13.0], atol=1e-5)


def test_many_moves_stay_in_bounds_and_float32():
    agents = Agents(SIZE, default_rng(0), _params(count=500, step_size=2.7))

    for _ in range(100):
        agents.move()

    assert ((agents.ys >= 0) & (agents.ys < SIZE[0])).all()
    assert ((agents.xs >= 0) & (agents.xs < SIZE[1])).all()
    assert agents.ys.dtype == np.float32
    assert agents.xs.dtype == np.float32


# --- deposit ---------------------------------------------------------------


def test_deposit_adds_to_the_cell_under_the_agent(make_agents):
    """Positions are truncated, so (3.9, 5.1) lands in cell [3, 5]."""
    agents = make_agents([3.9], [5.1], [0.0], deposit_amount=2.5)
    grid = np.zeros(SIZE, dtype=np.float32)

    agents.deposit(grid)

    assert grid[3, 5] == pytest.approx(2.5)
    assert grid.sum() == pytest.approx(2.5)


def test_deposit_adds_to_existing_trail(make_agents):
    agents = make_agents([3.0], [5.0], [0.0], deposit_amount=1.0)
    grid = np.zeros(SIZE, dtype=np.float32)
    grid[3, 5] = 4.0

    agents.deposit(grid)

    assert grid[3, 5] == pytest.approx(5.0)


def test_agents_sharing_a_cell_each_deposit(make_agents):
    """The reason for np.add.at: grid[ys, xs] += amount would count this once."""
    agents = make_agents([3.2, 3.7, 3.5], [5.1, 5.9, 5.5], [0.0, 0.0, 0.0])
    grid = np.zeros(SIZE, dtype=np.float32)

    agents.deposit(grid)

    assert grid[3, 5] == pytest.approx(3.0)


def test_deposit_at_exactly_the_grid_size_wraps_to_zero(make_agents):
    """Float rounding in move can leave a position at exactly the size."""
    agents = make_agents([20.0], [30.0], [0.0])
    grid = np.zeros(SIZE, dtype=np.float32)

    agents.deposit(grid)

    assert grid[0, 0] == pytest.approx(1.0)


def test_total_deposit_is_count_times_amount():
    agents = Agents(SIZE, default_rng(0), _params(count=200, deposit_amount=0.5))
    grid = np.zeros(SIZE, dtype=np.float32)

    agents.deposit(grid)

    assert grid.sum() == pytest.approx(100.0)
    assert grid.dtype == np.float32


# --- sense -----------------------------------------------------------------


@pytest.fixture
def marked_grid():
    """Distinct values around (10, 10): left 1, front 2, right 3.

    For an agent at (10.5, 10.5) with heading 0, sensor_distance 3 and
    sensor_angle pi/2: front is +x, left (+angle) is +y, right is -y.
    The half-cell offsets keep float error in sin/cos from changing the cell.
    """
    grid = np.zeros(SIZE, dtype=np.float32)
    grid[13, 10] = 1.0
    grid[10, 13] = 2.0
    grid[7, 10] = 3.0
    return grid


def test_sense_reads_left_front_right_in_order(make_agents, marked_grid):
    agents = make_agents([10.5], [10.5], [0.0])

    readings = agents.sense(marked_grid)

    assert_allclose(readings[:, 0], [1.0, 2.0, 3.0])


def test_sense_returns_float32_with_shape_3_by_n(make_agents, marked_grid):
    agents = make_agents([10.5, 2.5, 7.5, 15.5], [10.5, 3.5, 20.5, 25.5], [0.0, 1.0, 2.0, 3.0])

    readings = agents.sense(marked_grid)

    assert readings.shape == (3, 4)
    assert readings.dtype == np.float32


def test_sense_rotates_with_heading(make_agents, marked_grid):
    """Facing +y, the front sensor sees what was the left one."""
    agents = make_agents([10.5], [10.5], [np.pi / 2])

    readings = agents.sense(marked_grid)

    assert readings[1, 0] == pytest.approx(1.0)


def test_sense_wraps_around_the_edges(make_agents):
    grid = np.zeros(SIZE, dtype=np.float32)
    grid[5, 1] = 7.0
    agents = make_agents([5.5], [28.5], [0.0], sensor_distance=3.0)

    readings = agents.sense(grid)

    assert readings[1, 0] == pytest.approx(7.0)


def test_sense_does_not_change_grid_or_agents(make_agents, marked_grid):
    agents = make_agents([10.5], [10.5], [0.0])
    grid_before = marked_grid.copy()

    agents.sense(marked_grid)

    assert_allclose(marked_grid, grid_before)
    assert_allclose(agents.xs, [10.5])
    assert_allclose(agents.heading, [0.0])


# --- turn ------------------------------------------------------------------


def _readings(left, front, right):
    return np.array([[left], [front], [right]], dtype=np.float32)


@pytest.mark.parametrize(
    "left, front, right, change",
    [
        (5.0, 1.0, 2.0, +0.1),  # left strongest
        (2.0, 1.0, 5.0, -0.1),  # right strongest
        (1.0, 5.0, 2.0, 0.0),  # front strongest
        (3.0, 3.0, 3.0, 0.0),  # all equal
        (3.0, 3.0, 1.0, 0.0),  # left ties front
    ],
)
def test_turn_steers_toward_strongest_reading(make_agents, left, front, right, change):
    agents = make_agents([10.0], [10.0], [1.0], turn_speed=0.1)

    agents.turn(_readings(left, front, right))

    assert_allclose(agents.heading, [1.0 + change], atol=1e-6)


def test_turn_is_random_when_front_is_weakest_and_sides_tie(make_agents):
    count = 100
    agents = make_agents([10.0] * count, [10.0] * count, [1.0] * count, turn_speed=0.1)
    readings = np.tile(_readings(5.0, 1.0, 5.0), (1, count))

    agents.turn(readings)

    changes = agents.heading - 1.0
    assert_allclose(np.abs(changes), 0.1, atol=1e-6)
    assert (changes > 0).any()
    assert (changes < 0).any()


def test_turn_wraps_heading_into_zero_to_two_pi(make_agents):
    agents = make_agents([10.0, 10.0], [10.0, 10.0], [2 * np.pi - 0.05, 0.05], turn_speed=0.1)
    readings = np.array([[5.0, 1.0], [1.0, 1.0], [1.0, 5.0]], dtype=np.float32)

    agents.turn(readings)

    assert_allclose(agents.heading, [0.05, 2 * np.pi - 0.05], atol=1e-5)
    assert agents.heading.dtype == np.float32


def test_sense_then_turn_steers_toward_the_trail(make_agents):
    """sense and turn must agree on which side is 'left'."""
    grid = np.zeros(SIZE, dtype=np.float32)
    grid[13, 10] = 1.0  # left sensor of an agent at (10.5, 10.5) facing +x
    agents = make_agents([10.5], [10.5], [0.0], turn_speed=0.2)

    agents.turn(agents.sense(grid))
    agents.move()

    assert agents.ys[0] > 10.5
