# Devlog

Working notes, newest entries first. The overall plan is in [design-notes.md](design-notes.md).

---

## 2026-11-18: Create prototype in numpy and move it into modules

**Goal:** Create a prototype of the world grid and experiment with how it can be displayed in matplotlib

**Done:**
- Created a prototype script with classes for the trailmap and simulation, a function to run them, and dataclasses for parameters.
- Added documentation for them.
- Moved to separate modules in src and deleted script.
- Let Claude write the tests for simulation and trailmap.

**Decisions:**
- The separation of responsibilities in the classes to keep simplicity.

**Next:**
- Create agents class that will deposit pheromones and follow trails.

**Commits:**
- `8f9408f0aa81ba57a776c0c3500a2453c49f5204` Prototype of creating, running and displaying the map that the agents will later live in.
- `a95ad7f748beebfe0ef500b30aa0a9e218708bb8` Moved prototype script into modules


## 2026-11-17: Repo scaffolding and plan for the first step

**Goal:** Set up the repo and make a numpy prototype.

**Done:**
- Created repo with `uv init --package --vcs git --python 3.12 --description "Physarum slime-mold simulation on the GPU" physarum`
- Added numpy, scipy and matplotlib, and pytest and ruff as devtools
- Setup the repo with README, CLAUDE, AI-USE and LICENSE.
- Placed an idea note that was written from a chat with Claude: [design-notes.md](design-notes.md).
- Setup placeholder test.

**Decisions:**
- **Python 3.12, not newer:** Taichi that will be used later does not reliably support newer versions.
- **Matplotlib for displaying:** `imshow` to open a window and then update it with `set_data` each frame. Speedier solutions for later.

**Next:**
- Step 1 would be to build a prototype in python of how the world is gonna behave. See if the diffuse and decay work well.

**Commits:** `598f803266cae7024851e9f846858bf7fc2b5ca4` Initial project scaffold