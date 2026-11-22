import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from physarum.simulation import Simulation
from physarum.capture import Recorder

from physarum.colormaps import CYCLE, next_name


def run_matplotlib(
        sim: Simulation, 
        vmax: float = 0.1,
        window_px: int = 900,
        dpi: int = 300,
        recorder: Recorder | None = None
    ) -> None:
    """Show the simulation live in a matplotlib window until it is closed.

    Parameters
    ----------
    sim : Simulation
        Stepped once per frame.
    vmax : float
        Trail value shown as full brightness. Default 0.1.
    """

    # create recorder
    if not recorder:
        recorder = Recorder(sim.params, vmax=vmax)

    # change look and interactivity
    plt.rcParams["toolbar"] = "None"
    remove_keystrokes: list[str] = ["s", "g", "c", "C"]
    for k, v in plt.rcParams.items():
        if not "keymap" in k:
            continue
        for ks in remove_keystrokes:
            if ks in v:
                v.remove(ks)

    # create figure
    fig, axes = plt.subplots(dpi=dpi, facecolor="black")
    scale = window_px / max(sim.params.grid_height, sim.params.grid_width)
    fig.set_figheight(sim.params.grid_height * scale / dpi)
    fig.set_figwidth(sim.params.grid_width * scale / dpi)
    axes.set_axis_off()

    # plot the grid
    img = axes.imshow(sim.trail_map.grid, vmin=0, vmax=vmax, cmap=CYCLE[0], interpolation="nearest")

    # create keypress handler
    def on_press(event):
        sys.stdout.flush()

        if event.key == "s":
            p = recorder.save_png(img.get_cmap().name)
            print(p or "Nothing buffered yet")
        if event.key == "g":
            p = recorder.save_gif(img.get_cmap().name)
            print(p or "Nothing buffered yet")
        if event.key in ["c", "C"]:
            img.set_cmap(next_name(img.get_cmap().name, 1))

    fig.canvas.mpl_connect("key_press_event", on_press)

    # create animation function
    def animation_func(frame: int) -> None:
        if frame % 10 == 0:
            fig.canvas.manager.set_window_title(
                f"physarum · {img.get_cmap().name} · "
                + f"frame {frame} · {sim.agents.params.count} agents"
            )

        sim.step()
        recorder.push(sim.frame, sim.trail_map.grid)
        img.set_data(sim.trail_map.grid)

    # setup the animation
    _animation = FuncAnimation(
        fig,
        animation_func,
        cache_frame_data=False,
        interval=50,
    )

    plt.show()