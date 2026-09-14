# Proposal: frontend-scene-capture

## Why

Every simulation problem reported from the Blender frontend has to be
reconstructed by hand before it can be tested: the harness only builds
procedural grid scenes, so a real garment (multiple panels, sealing chains, a
body obstacle, per-panel fabric properties, a world transform per object) cannot
be reproduced from the .blend the user is looking at. Today the only way to get
any signal from such a scene is to drive Blender interactively and read the
viewport, which the addon cannot do in a background session either: the pattern
rebuild path needs a GPU context for its gizmo renderers and the addressing
needed by `setup_sewings_for_simulation` is only filled while the UI runs.

That gap is what made the current "the cloth keeps shaking" investigation
expensive, and it will keep costing time on every frontend report.

## What Changes

- Add a scene capture to the Blender addon: one operator that packages the
  current scene's simulation inputs (per-object mesh, rest and simulated
  positions, edges, triangles, world matrix, fabric properties, collision
  layer, fixed/attached weights, sewing stitch pairs and the solver parameter
  block) into a portable directory that the repository's test system can read
  without Blender.
- Make the same capture work in a background Blender session, so a captured
  scene can be produced and re-captured from a script: the pattern/sewing
  rebuild must not require a GPU context, and the addressing that
  `setup_sewings_for_simulation` needs must be buildable without the UI.
- Add the matching loader to the test harness, modeled on the existing
  data-driven dataset loader: the captured package becomes an `input_data`
  payload plus the metadata the debug window needs, registered as a scene by
  name so harness cases, the debug window and ad-hoc scripts can all use it.
- Define the package format (arrays plus a JSON sidecar, relative paths only,
  documented units) and the round-trip requirement: loading a capture must
  reproduce the payload the frontend sent, byte-for-byte for the integer arrays
  and within float tolerance for positions.

## Capabilities

### New Capabilities

- `frontend-scene-capture`: capture the simulation inputs of a live Blender
  scene (including in a background session) into a portable package, and define
  what that package must contain so a captured scene is reproducible outside
  Blender.

### Modified Capabilities

- `testing-harness`: the harness must be able to run the scenarios it already
  supports against a captured frontend scene, not only against procedurally
  generated ones, while keeping the no-Blender-dependency property at test
  time.

## Impact

- Frontend addon: a new operator plus the non-UI capture entry point, and the
  rebuild path used by `setup_sewings_for_simulation` must tolerate a session
  without a GPU context.
- Engine repository: a harness loader and a scene-registry entry; no change to
  the extension itself and no change to the public Python API.
- Verification: the reported garment scene becomes a reusable regression case,
  which is a precondition for measuring the solver-damping work.
- Portability: a captured package is data, so no machine-specific paths, build
  outputs or environment details enter the repository; captures live under the
  gitignored artifact tree and only the format contract is committed.
