# Physarum slime-mold simulation

## Pitch

Millions of simple agents leave pheromone trails and follow each other's scent. Out of that come living, branching networks that look like slime mold, veins or mycelium. The idea came from a Coding Adventures video. The goal here is to build it from scratch in Python and make it run live at 60 fps on the GPU.

## Portfolio angle

- **GPU programming from Python:** real compute kernels, not just a library call.
- **Performance engineering:** the same algorithm at four speeds, with a benchmark table in the README.
- **Visual hook:** the README's top GIF does the selling.
- **Simple to understand, deep to tune:** a reader gets the algorithm in 30 seconds, then sees how parameters shape emergent structure.

## Core algorithm

Based on Jeff Jones (2010), *Characteristics of pattern formation and evolution in approximations of Physarum transport networks*. Each frame:

1. **Agents** (each has a position and a heading):
   - Sense the trail map at three points: front-left, front and front-right, at `sensor_distance` and ±`sensor_angle`.
   - Turn toward the strongest reading by `turn_speed`. If the sides tie, turn randomly.
   - Move forward by `step_size`. Bounce or wrap at the edges.
   - Deposit `deposit_amount` of pheromone at the new position.
2. **Trail map** (a 2D float grid):
   - **Diffuse:** a 3×3 box blur, mixed with the old value by `diffuse_rate`.
   - **Decay:** multiply by `decay` (about 0.95).
3. **Render:** map trail intensity through a color ramp.

The core is about 50 lines. The look comes from four knobs: sensor angle, sensor distance, turn speed and decay.

## Build path

1. **M1: NumPy prototype.** About 100k agents, vectorized sensing with fancy indexing, and `scipy.ndimage.uniform_filter` for diffusion. Show it with matplotlib animation or plain image dumps. *Done when:* the network structures appear and you understand why.
2. **M2: Taichi live version.** Agents and trail as Taichi fields, kernels for move/deposit and diffuse/decay, and `ti.ui.Window` showing the trail texture with no CPU copy-back. *Done when:* 1M+ agents run at 60 fps on the desktop.
3. **M3: parameter UI and presets.** Live sliders (Taichi GUI widgets or imgui) plus a few named presets ("veins", "mycelium", "honeycomb") saved as TOML. *Done when:* you can find a new look in under a minute.
4. **M4: ModernGL compute-shader port.** GLSL compute shaders for agents and diffusion, drawn to a full-screen quad through `moderngl-window`. This is the "real" GPU-programming version, as in the video. *Done when:* it matches Taichi's output and frame rate.
5. **M5: benchmarks and README.** A table of agents vs. fps for NumPy, Numba, Taichi and ModernGL on the RTX 2060 SUPER desktop and the 680M laptop, plus a top GIF and a short "how it works" section.

M2 is already a portfolio-worthy stopping point. M4 is optional depth.

## Stretch twists

- **Multiple species:** each species follows its own trail and avoids the others, giving competing colored networks.
- **Food sources and obstacles:** place attractors or a map image and watch it solve shortest-path or "Tokyo rail network" layouts (Tero et al., 2010).
- **Video export:** a headless render to MP4 for the README and socials.
- **Interaction:** paint pheromone or repellent with the mouse.
- **Neuro twist:** treat the trail map as a "neural field" and agents as axon growth cones. This is a loose link to [[ideas/boids-gpu|GPU boids]]'s spiking brains.

## Tools likely needed

NumPy, SciPy, Taichi, ModernGL and moderngl-window, imgui (optional), and imageio or ffmpeg for export. These get pages in `tools/` once a project actually uses them.

## Open questions

- **Taichi only, or Taichi then ModernGL?** Taichi gets to impressive results fastest. ModernGL teaches GLSL. Proposal: do Taichi first and decide on M4 after M3.
- **Browser demo?** WebGPU would mean porting the shaders to WGSL in JS. Probably out of scope, and a video is enough.
- **Laptop support:** check Taichi's Vulkan backend on the 680M early.

## Related ideas

- [[ideas/boids-gpu|GPU boids]] uses the same stack (GPU agent simulation in Taichi or ModernGL). Build this one first, since it's simpler, and reuse the kernel, window and benchmark scaffolding. Keep them as separate repos: two polished projects read better than one combined one.

## Source

- Side chat: "btw: I saw a video by Coding Adventures…" (claude://claude.ai/epitaxy/local_e3ba7cbd-366a-4dee-8bf2-379e5908c6b8), forked from "Python project ideas for portfolio".
- Jones, J. (2010). *Characteristics of pattern formation and evolution in approximations of Physarum transport networks.* Artificial Life 16(2).
- Coding Adventures (Sebastian Lague), "Ant and Slime Simulations" (YouTube).
