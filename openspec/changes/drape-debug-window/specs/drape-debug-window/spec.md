## Purpose

An interactive, code-configured debug window that lets testers watch a drape
simulation live, drag the cloth by hand, and immediately see when the
simulation blows up — without Blender and without editing pytest code.

## ADDED Requirements

### Requirement: Launch with a registered debug scene

The debug window SHALL be launched from the repository test tree and SHALL
resolve the `Qianyi_DP` module through the same mechanism as the pytest
harness (env var, build-output scan, then import), failing with the same
rebuild hint when no module is found. The scene to load SHALL be selected by
name from a scene registry that includes at least one procedural harness
scene and GarmentCodeData element loading.

#### Scenario: List available scenes

- **WHEN** the tester requests the scene list from the command line
- **THEN** every registered scene name is printed with its source kind
  (procedural or GarmentCodeData)

#### Scenario: Unknown scene name

- **WHEN** the window is launched with a scene name that is not registered
- **THEN** startup fails with an error listing the available scene names

#### Scenario: GarmentCodeData environment missing

- **WHEN** a GarmentCodeData scene is selected and `QYDP_GCD_ROOT` is not set
- **THEN** startup fails with setup instructions naming the required
  environment variables, consistent with the data-driven test suite

### Requirement: Live simulation with interactive pacing

The window SHALL advance the simulation live with interactive pacing by
default (two `update(0.003)` steps per rendered frame; a single `0.01` step
tunnels through the thin collision layer and collapses the garment off the
body), targeting interactive frame cost; harness-equivalent pacing (24 fps
frame time, `dt = 0.001`, `ceil(frame_time / dt)` substeps per frame) SHALL
remain available as a code configuration. The solver and parameter block
SHALL be applied through the existing preset registry before `input_data`.
The tester SHALL be able to pause, resume, advance exactly one frame, and
reset the simulation to its initial state without restarting the process.

#### Scenario: Default stepping is interactive

- **WHEN** the window runs uninterrupted with default configuration
- **THEN** each rendered frame advances the simulation by two `update(0.003)`
  steps while keeping the garment on the body

#### Scenario: Harness pacing available

- **WHEN** the tester switches the configuration to harness semantics
- **THEN** each rendered frame runs the same substep count and dt as
  `tests/harness/driver.py`

#### Scenario: Pause, single step, reset

- **WHEN** the tester pauses, requests a single frame, and then resets
- **THEN** single stepping advances exactly one frame while paused, and reset
  restores the initial cloth state and zeroes the frame counter

### Requirement: Interactive cloth dragging

The window SHALL let the tester grab the cloth with the mouse and pull it,
using the engine's triangle pick constraint (press picks the triangle under
the cursor, drag moves the pick target, release removes it), without
disabling the renderer's built-in camera controls (orbit, zoom, pan) when no
pick is active.

#### Scenario: Grab and release

- **WHEN** the tester presses the pick button on the cloth, drags, and
  releases
- **THEN** the cloth follows the cursor while the button is held and resumes
  free simulation under engine constraints on release

#### Scenario: Camera unaffected

- **WHEN** no pick is active
- **THEN** mouse orbit, scroll zoom, and key pan behave exactly as the
  renderer's defaults

### Requirement: Blow-up detection and reporting

While stepping, the window SHALL check every frame for non-finite vertex
positions and for per-frame maximum displacement beyond a configured
threshold. On detection it SHALL pause the simulation automatically, mark the
cloth visually as failed, and print the frame index with the offending
statistics to the console.

#### Scenario: Non-finite vertices

- **WHEN** any simulated vertex becomes NaN or infinite
- **THEN** the simulation pauses automatically and the console reports the
  frame index and the non-finite count

#### Scenario: Displacement threshold exceeded

- **WHEN** the maximum vertex displacement in one frame exceeds the
  configured threshold
- **THEN** the simulation pauses automatically and the console reports the
  frame index and the measured displacement

### Requirement: Screenshot capture

The window SHALL capture the rendered framebuffer to a PNG file. A hotkey
SHALL save the current frame to the gitignored test-artifacts tree under the
scene name, stamped with the frame index. A scripted non-interactive run
mode SHALL advance a configured number of frames from the initial state with
a fixed camera, optionally perform one scripted drag, write the screenshot
file(s), print every written path to the console, and exit — so automated
consumers (including multimodal agents) can obtain the visual result of a
run without interacting with the window.

#### Scenario: Hotkey capture

- **WHEN** the tester presses the screenshot hotkey while the window runs
- **THEN** a PNG of the current frame is written under the scene's artifact
  directory with the frame index in the filename, and the console prints the
  written path

#### Scenario: Scripted capture for an automated consumer

- **WHEN** the window is launched in scripted mode with a frame count and a
  screenshot target
- **THEN** it runs unattended from the initial state, writes the requested
  screenshot file(s), prints their paths, and exits without waiting for user
  input

### Requirement: Programmatic pick testing

The drag machinery (cursor-to-ray unprojection, ray-triangle hit testing,
drag-plane intersection, and the pick press/drag/release lifecycle) SHALL be
importable from a module that does not import the renderer. The repository
SHALL provide renderer-free unit tests for the geometry math and a
GPU-marked integration test that drives a full pick cycle headlessly —
without creating a window — and asserts the observable effect on the cloth.

#### Scenario: Headless drag cycle moves the cloth

- **WHEN** the integration test presses a synthetic pick at a known cloth
  location, drags the target, steps the simulation, and releases
- **THEN** the picked triangle's vertices displace measurably toward the
  drag target while the pick is held, and relax back after release, compared
  against an unpicked baseline run

#### Scenario: Ray math verified without a GPU

- **WHEN** the renderer-free unit tests run on a machine without a GPU
- **THEN** cursor unprojection, ray-triangle intersection, and drag-plane
  intersection match analytic expectations on synthetic data

### Requirement: Seam-aware panel rendering

For multi-panel scenes with sewings, the window SHALL render the full cloth
topology (seam-bridging triangles are kept and drawn as fabric), draw the
sewing chains as thin lines so seam boundaries stay visible, and cycle a
color palette per panel. This is display-only - the engine input keeps the
full topology.

#### Scenario: Sewing chains are drawn on moving cloth

- **WHEN** a multi-panel scene is displayed
- **THEN** the seam-chain lines are recomputed every frame from the current
  panel vertices and follow the moving panels

#### Scenario: Full topology stays in the engine

- **WHEN** a multi-panel scene is displayed
- **THEN** the rendered cloth uses the full panel topology, and the engine
  `input_data` and picking keep the same (unculled) triangles

### Requirement: Code-only configuration

All tunable settings (scene name, solver, parameter overrides, fps, dt,
substeps, blow-up thresholds) SHALL be defined as constants in the entry
module. The window SHALL NOT provide runtime settings dialogs or panels.

#### Scenario: Change a setting

- **WHEN** the tester wants a different solver or threshold
- **THEN** the value is changed by editing the entry module's configuration
  block and relaunching

### Requirement: Registered scenes stay loadable

Every scene registered in the registry SHALL be covered by a CPU-only
regression test (no GPU, no window) that builds its `input_data` and passes
the existing mesh-list validation, so adding debug scenes cannot break the
registry silently.

#### Scenario: Newly registered scene

- **WHEN** a scene is added to the registry
- **THEN** the CPU-only registry test exercises its loader and fails if the
  produced `input_data` is invalid or its environment preconditions are
  misdeclared
