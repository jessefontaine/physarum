import matplotlib
import numpy as np

from physarum.colormaps import CYCLE, lut, next_name, register, to_rgb


def test_cycle_order():
    assert CYCLE == ("magma", "inferno", "twilight_shifted", "cividis", "bone", "physarum")

def test_register_is_idempotent():
    register(); register()
    assert "physarum" in matplotlib.colormaps

def test_physarum_runs_black_to_white():
    t = lut("physarum")
    assert t.shape == (256, 3) and t.dtype == np.uint8
    assert tuple(t[0]) == (0, 0, 0) and tuple(t[-1]) == (255, 255, 255)

def test_to_rgb_uses_lut():
    frame = np.array([[0, 255]], dtype=np.uint8)
    rgb = to_rgb(frame, "magma")
    assert rgb.shape == (1, 2, 3)
    assert (rgb[0, 1] == lut("magma")[255]).all()

def test_next_name_wraps_both_ways():
    assert next_name("physarum") == "magma"
    assert next_name("magma", -1) == "physarum"