import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from physarum.params import AgentParams, SimParams, TrailParams, _Validated, bounded
from physarum.simulation import Simulation

PRESETS = Path(__file__).parent.parent / "presets"


@pytest.fixture
def params() -> SimParams:
    """Non-square grid and distinct values, so swapped fields show up."""
    return SimParams(
        grid_height=4,
        grid_width=6,
        trail=TrailParams(decay_rate=0.5, diffusion_rate=0.25),
        agent=AgentParams(
            count=10,
            step_size=1.5,
            deposit_amount=2.0,
            sensor_distance=3.0,
            sensor_angle=0.7,
            turn_speed=0.3,
        ),
        seed=42,
    )


@pytest.fixture
def params_dict(params) -> dict:
    return params.to_dict()


# --- round trips -------------------------------------------------------------


def test_dict_round_trip(params):
    assert SimParams.from_dict(params.to_dict()) == params


def test_to_dict_nests_sections(params):
    d = params.to_dict()
    assert d["trail"] == {"decay_rate": 0.5, "diffusion_rate": 0.25}
    assert d["agent"]["count"] == 10
    assert d["grid_height"] == 4


def test_toml_round_trip(params, tmp_path):
    path = tmp_path / "preset.toml"
    params.to_toml(path)
    assert SimParams.from_toml(path) == params


def test_to_toml_writes_sections(params, tmp_path):
    path = tmp_path / "preset.toml"
    params.to_toml(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert set(data) == {"grid_height", "grid_width", "trail", "agent", "seed"}
    assert data["trail"]["decay_rate"] == 0.5


def test_to_toml_accepts_str_path(params, tmp_path):
    path = str(tmp_path / "preset.toml")
    params.to_toml(path)
    assert SimParams.from_toml(path) == params


def test_to_toml_overwrites_existing_file(params, tmp_path):
    path = tmp_path / "preset.toml"
    path.write_text("junk = 1\n")
    params.to_toml(path)
    assert SimParams.from_toml(path) == params


# --- the shipped preset ------------------------------------------------------


def test_default_preset_loads():
    params = SimParams.from_toml(PRESETS / "default_parameters.toml")
    assert isinstance(params.trail, TrailParams)
    assert isinstance(params.agent, AgentParams)


def test_simulation_from_file(params, tmp_path):
    path = tmp_path / "preset.toml"
    params.to_toml(path)
    sim = Simulation.from_file(path)
    assert sim.trail_map.grid.shape == params.size


# --- hand-edited files -------------------------------------------------------


def test_int_accepted_for_float(tmp_path):
    """A hand-written `step_size = 1` should load as 1.0."""
    path = tmp_path / "preset.toml"
    path.write_text(
        "grid_height = 4\ngrid_width = 6\nseed = 0\n"
        "[trail]\ndecay_rate = 1\ndiffusion_rate = 0\n"
        "[agent]\ncount = 1\nstep_size = 1\ndeposit_amount = 1\n"
        "sensor_distance = 2\nsensor_angle = 1\nturn_speed = 1\n"
    )
    params = SimParams.from_toml(path)
    assert params.agent.step_size == 1.0
    assert type(params.agent.step_size) is float
    assert type(params.trail.decay_rate) is float


def test_float_rejected_for_int(params_dict):
    params_dict["agent"]["count"] = 10.0
    with pytest.raises(TypeError, match="count"):
        SimParams.from_dict(params_dict)


def test_bool_rejected(params_dict):
    params_dict["seed"] = True
    with pytest.raises(TypeError, match="seed"):
        SimParams.from_dict(params_dict)


def test_string_rejected(params_dict):
    params_dict["trail"]["decay_rate"] = "0.5"
    with pytest.raises(TypeError, match="decay_rate"):
        SimParams.from_dict(params_dict)


def test_invalid_toml_syntax(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("grid_height = \n")
    with pytest.raises(tomllib.TOMLDecodeError):
        SimParams.from_toml(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SimParams.from_toml(tmp_path / "nope.toml")


# --- key checks --------------------------------------------------------------


def test_from_dict_ignores_capture_table(params):
    d = params.to_dict() | {"capture": {"kind": "png", "frame_start": 3}}
    assert SimParams.from_dict(d) == params

def test_from_dict_still_rejects_other_unknown_keys(params):
    with pytest.raises(ValueError, match="unknown keys"):
        SimParams.from_dict(params.to_dict() | {"extra": 1})


def test_unknown_top_level_key(params_dict):
    params_dict["colour"] = "red"
    with pytest.raises(ValueError, match=r"SimParams: unknown keys \['colour'\]"):
        SimParams.from_dict(params_dict)


def test_missing_top_level_key(params_dict):
    del params_dict["seed"]
    with pytest.raises(ValueError, match=r"SimParams: missing keys \['seed'\]"):
        SimParams.from_dict(params_dict)


def test_unknown_and_missing_reported_together(params_dict):
    params_dict["sead"] = params_dict.pop("seed")
    with pytest.raises(
        ValueError, match=r"unknown keys \['sead'\], missing keys \['seed'\]"
    ):
        SimParams.from_dict(params_dict)


@pytest.mark.parametrize(
    ("section", "cls"), [("trail", "TrailParams"), ("agent", "AgentParams")]
)
def test_unknown_key_in_section(params_dict, section, cls):
    params_dict[section]["typo"] = 1.0
    with pytest.raises(ValueError, match=rf"{cls}: unknown keys \['typo'\]"):
        SimParams.from_dict(params_dict)


@pytest.mark.parametrize(
    ("section", "key", "cls"),
    [("trail", "decay_rate", "TrailParams"), ("agent", "turn_speed", "AgentParams")],
)
def test_missing_key_in_section(params_dict, section, key, cls):
    del params_dict[section][key]
    with pytest.raises(ValueError, match=rf"{cls}: missing keys \['{key}'\]"):
        SimParams.from_dict(params_dict)


@pytest.mark.parametrize("section", ["trail", "agent"])
def test_section_must_be_a_table(params_dict, section):
    params_dict[section] = 0.5
    with pytest.raises(TypeError, match=f"'{section}' must be a dict"):
        SimParams.from_dict(params_dict)


# --- bounds ------------------------------------------------------------------


@dataclass
class _Bounded(_Validated):
    rate: float = bounded(ge=0.0, le=1.0)
    size: int = bounded(gt=0)


@pytest.mark.parametrize(("rate", "size"), [(0.0, 1), (1.0, 1), (0.5, 100)])
def test_bounds_accept_edges(rate, size):
    _Bounded(rate=rate, size=size)


@pytest.mark.parametrize(
    ("rate", "size", "name"), [(-0.1, 1, "rate"), (1.1, 1, "rate"), (0.5, 0, "size")]
)
def test_bounds_reject_out_of_range(rate, size, name):
    with pytest.raises(ValueError, match=f"_Bounded.{name} = .* out of range"):
        _Bounded(rate=rate, size=size)
