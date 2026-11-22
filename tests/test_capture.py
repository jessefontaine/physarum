import tomllib

import numpy as np
import pytest
from PIL import Image

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

def _filled(params, tmp_path, n=5, **kw):
    rec = Recorder(params, out_dir=tmp_path, **kw)
    for f in range(1, n + 1):
        rec.push(f, np.full(params.size, f / 10, np.float32))
    return rec

def test_save_with_empty_buffer_returns_none(params, tmp_path):
    assert Recorder(params, out_dir=tmp_path).save_png("magma") is None
    assert list(tmp_path.iterdir()) == []

def test_png_size_and_sidecar(params, tmp_path):
    path = _filled(params, tmp_path, upscale=2).save_png("magma")
    assert Image.open(path).size == (params.grid_width * 2, params.grid_height * 2)
    assert path.stem.endswith("_f5-5")
    side = tomllib.loads(path.with_suffix(".toml").read_text())
    assert side["capture"]["kind"] == "png" and side["capture"]["colormap"] == "magma"
    assert side["capture"]["reproducible"] is True
    assert SimParams.from_toml(path.with_suffix(".toml")) == params

def test_gif_has_all_buffered_frames(params, tmp_path):
    path = _filled(params, tmp_path, n=5, vmax=1.0).save_gif("bone", fps=10)
    img = Image.open(path)
    assert img.n_frames == 5 and img.size == (params.grid_width, params.grid_height)
    cap = tomllib.loads(path.with_suffix(".toml").read_text())["capture"]
    assert (cap["frame_start"], cap["frame_end"], cap["fps"]) == (1, 5, 10)

def test_sidecar_keeps_frame0_params(params, tmp_path):
    rec = _filled(params, tmp_path)
    params.agent.sensor_angle = 0.123          # live change after construction
    loaded = SimParams.from_toml(rec.save_png("magma").with_suffix(".toml"))
    assert loaded.agent.sensor_angle != 0.123

def test_changes_mark_not_reproducible(params, tmp_path):
    rec = _filled(params, tmp_path)
    rec.note_change(3, "trail.decay_rate", 0.9)
    cap = tomllib.loads(rec.save_png("magma").with_suffix(".toml").read_text())["capture"]
    assert cap["reproducible"] is False
    assert cap["changes"] == [{"frame": 3, "key": "trail.decay_rate", "value": 0.9}]

def test_same_name_gets_suffix(params, tmp_path):
    rec = _filled(params, tmp_path)
    a, b = rec.save_png("magma"), rec.save_png("magma")
    assert a != b and a.exists() and b.exists()