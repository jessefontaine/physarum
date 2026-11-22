import numpy as np
import pytest

from physarum.capture import Recorder
from physarum.params import AgentParams, SimParams, TrailParams


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


def test_buffer_keeps_newest(params):
    rec = Recorder(params, buffer_len=3)
    for f in range(1, 6):
        rec.push(f, np.zeros(params.size, np.float32))
    assert [f for f, _ in rec.frames] == [3, 4, 5]

def test_push_clips_to_vmax(params):
    rec = Recorder(params, vmax=0.1)
    grid = np.full(params.size, 0.05, np.float32)
    grid[0, 0], grid[0, 1] = 5.0, -1.0
    rec.push(1, grid)
    u8 = rec.frames[-1][1]
    assert u8.dtype == np.uint8 and u8.shape == params.size
    assert u8[0, 0] == 255 and u8[0, 1] == 0 and u8[1, 1] in (127, 128)

def test_push_copies_grid(params):
    rec = Recorder(params); grid = np.zeros(params.size, np.float32)
    rec.push(1, grid); grid[:] = 1.0
    assert rec.frames[-1][1].max() == 0

def test_note_change_is_recorded(params):
    rec = Recorder(params)
    rec.note_change(20, "agent.sensor_angle", 0.6)
    assert rec.changes == [{"frame": 20, "key": "agent.sensor_angle", "value": 0.6}]

