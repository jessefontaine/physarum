import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from physarum.capture import Recorder
from physarum.colormaps import CYCLE, next_name
from physarum.simulation import Simulation


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
    window_px : int
        Maximum side size in pixels of the the simulation window, default 900.
    dpi : int
        Simulation figure dots per inch, default 300.
    recorder : Recorder | None
        Frame buffer recored, default None

    Notes
    -----
    The default Recorder that is instantiated by default has default parameters. 
    Passing a custom one allows setting a longer buffer length, output directory, or 
    upscaling.
    """

    # create recorder
    if not recorder:
        recorder = Recorder(sim.params, vmax=vmax)

    # remove default keystroke actions
    remove_keystrokes: list[str] = ["s", "g", "c", "C"]
    for k, v in plt.rcParams.items():
        if not "keymap" in k:
            continue
        for ks in remove_keystrokes:
            if ks in v:
                v.remove(ks)

    # create figure
    plt.rcParams["toolbar"] = "None"
    fig, axes = plt.subplots(dpi=dpi, facecolor="black")

    # give the window the maximum pixel size
    scale = window_px / max(sim.params.grid_height, sim.params.grid_width)
    fig.set_figheight(sim.params.grid_height * scale / dpi)
    fig.set_figwidth(sim.params.grid_width * scale / dpi)

    # turn off axis
    axes.set_axis_off()

    # plot the grid
    img = axes.imshow(sim.trail_map.grid, vmin=0, vmax=vmax, cmap=CYCLE[0], interpolation="nearest")

    # create keypress handler
    def on_press(event):
        """Add custom keypress actions."""
        sys.stdout.flush()

        # current cmap
        cmap: str = img.get_cmap().name

        # save image
        if event.key == "s":
            p = recorder.save_png(cmap)
            print(p or "Nothing buffered yet")

        # save gif
        if event.key == "g":
            p = recorder.save_gif(cmap)
            print(p or "Nothing buffered yet")

        # cycle colormap
        if event.key in ["c", "C"]:
            img.set_cmap(next_name(cmap, 1))

    # connect the new handler
    fig.canvas.mpl_connect("key_press_event", on_press)

    # create animation function
    def animation_func(frame: int) -> None:
        """Run these actions every frame of the simulation."""

        # update window title every 10 frames
        if frame % 10 == 0:
            fig.canvas.manager.set_window_title(
                f"physarum · {img.get_cmap().name} · "
                + f"frame {frame} · {sim.agents.params.count} agents"
            )

        # update simulation
        sim.step()

        # buffer frame and display
        recorder.push(sim.frame, sim.trail_map.grid)
        img.set_data(sim.trail_map.grid)

    # setup the animation
    _animation = FuncAnimation(
        fig,
        animation_func,
        cache_frame_data=False,
        interval=50,
    )

    # stop timer when the animation closes, so no frames run after teardown
    fig.canvas.mpl_connect("close_event", lambda _event: _animation.event_source.stop())

    plt.show()