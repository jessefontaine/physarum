import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Self

import tomli_w


def _check_keys(dcls: type, d: dict[str, Any]) -> None:
    """Raise ValueError unless the keys of `d` match the fields of `dcls` exactly."""
    expected = {f.name for f in fields(dcls)}
    unknown = sorted(d.keys() - expected)
    missing = sorted(expected - d.keys())

    problems = []
    if unknown:
        problems.append(f"unknown keys {unknown}")
    if missing:
        problems.append(f"missing keys {missing}")
    if problems:
        raise ValueError(f"{dcls.__name__}: " + ", ".join(problems))


def bounded(*, ge=None, gt=None, le=None):
    """A dataclass field with optional bounds: ge (>=), gt (>) and le (<=)."""
    return field(metadata={"ge": ge, "gt": gt, "le": le})


class _Validated:
    """Base for parameter dataclasses.

    Checks for types and bounds.

    """

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)

            # toml writes 1 for 1.0, so we can accept an int where a float is expected
            if f.type is float and type(value) is int:
                value = float(value)
                setattr(self, f.name, value)

            if isinstance(value, bool) or not isinstance(value, f.type):
                raise TypeError(f"{type(self).__name__}.{f.name} must be {f.type.__name__}, got {type(value).__name__}")

            ge, gt, le = (f.metadata.get(k) for k in ("ge", "gt", "le"))
            if (ge is not None and value < ge) or (gt is not None and value <= gt) or (le is not None and value > le):
                raise ValueError(f"{type(self).__name__}.{f.name} = {value} is out of range (ge={ge}, gt={gt}, le={le})")


@dataclass
class TrailParams(_Validated):
    """Settings for how the trail map evolves each frame.

    Attributes
    ----------
    decay_rate : float
        Multiplier applied to every cell per frame [0, 1]. Lower values make trails fade
        faster. Typical: 0.9-0.99.
    diffusion_rate : float
        How much the blurred grid is mixed per frame [0, 1]. 
        0 means no spreading, 1 means a full 3x3 blur.
    """

    decay_rate: float
    diffusion_rate: float


@dataclass
class AgentParams(_Validated):
    """Settings for how agents sense, turn, move and deposit.

    Attributes
    ----------
    count : int
        Number of agents. Fixed once the simulation is built.
    step_size : float
        Distance each agent moves per frame, in cells. Typical: 1.
    deposit_amount : float
        Pheromone added to the agent's cell after each move.
    sensor_distance : float
        How far ahead the sensors sit, in cells. Larger values give
        coarser networks. Typical: 5-15.
    sensor_angle : float
        Angle between the front sensor and each side sensor, in radians.
        Typical: pi/8 to pi/4 (22.5-45 degrees).
    turn_speed : float
        How far an agent turns per frame when steering, in radians.
        Typical: pi/8 to pi/4.
    """

    count: int
    step_size: float
    deposit_amount: float
    sensor_distance: float
    sensor_angle: float
    turn_speed: float


@dataclass
class SimParams(_Validated):
    """All settings for one simulation run.

    Attributes
    ----------
    grid_height : int
        Trail map rows, in cells. Fixed once the simulation is built.
    grid_width : int
        Trail map columns, in cells. Fixed once the simulation is built.
    trail : TrailParams
        Diffusion and decay settings, adjustable while running.
    agent : AgentParams
        Sensing, movement and deposit settings.
    seed : int
        Seed for the random generator, so runs are reproducible.
    """

    grid_height: int
    grid_width: int
    trail: TrailParams
    agent: AgentParams
    seed: int

    @property
    def size(self) -> tuple[int, int]:
        """Grid shape as (height, width)"""
        return self.grid_height, self.grid_width

    def to_dict(self) -> dict[str, dict[str, int | float] | int]:
        return asdict(self)

    def to_toml(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            tomli_w.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:

        # remove any capture keys when loading
        d = {k: v for k, v in d.items() if k != "capture"}

        _check_keys(cls, d)

        # check whether sub dictionaries are formatted directly
        trail, agent = d["trail"], d["agent"]
        for name, part in (("trail", trail), ("agent", agent)):
            if not isinstance(part, dict):
                raise TypeError(f"'{name}' must be a dict, got {type(part).__name__}")
        _check_keys(TrailParams, trail)
        _check_keys(AgentParams, agent)

        # create classes
        top = {k: v for k, v in d.items() if k not in ("trail", "agent")}
        return cls(**top, trail=TrailParams(**trail), agent=AgentParams(**agent))

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))
