# Design: frontend-scene-capture

## Context

The harness already has a data-driven precedent: the dataset loader maps an
external element into the `input_data` contract (one cloth mesh per panel,
identity-position sewing pairs, optional body obstacle) and the batch runner
drives it like any other case. The capture path is the same shape of problem
with a different source, so the design reuses that structure instead of
inventing a new one.

What a capture needs is exactly what the frontend already assembles in
`SimulationManager._initialize_object_simulation` plus
`setup_sewings_for_simulation`:

per object: `vertices` (rest / pattern placement), `vertices_sim` (state the
solver starts from), `edges`, `triangles`, `world_matrix`, `object_type`,
`collision_layer`, `normals` for non-cloth, and for cloth `mass` (g/m^2),
`granularity` (mm), `thickness` (mm), `friction`, `stretch`, `bending`,
`grain_dir`, `fixed_vertices`, `attached_vertices`;

per sewing: `stitches` (N x 2 panel-local vertex ids) and `patterns` (the two
panel indices), plus the optional `angle` / `compress` fields the engine reads;

plus the solver selection and the parameter block that was active, which the
frontend does not own today (the notebook or the UI sets it on the extension
singleton).

## Package format

A capture is a directory containing:

- `scene.json`: format version, source identification (scene name, object
  names, capture time, addon version), the ordered object metadata, the sewing
  entries, the solver name and the parameter block, and a summary block (panel
  count, stitch count, vertex/edge/triangle counts, edge-length statistics,
  bounding boxes, per-panel material properties).
- `scene.npz`: the arrays, keyed by object index and field name, plus the
  concatenated sewing stitch array and its offsets. All integers are int32,
  all floats float32, positions in metres, `granularity` and `thickness` in
  millimetres, `mass` in grams per square metre - the same units the frontend
  sends, so no conversion happens on either side.

Relative paths only: the capture never records an absolute path, and the loader
resolves everything relative to the package directory. A `source` field may
carry the `.blend` basename for traceability, never its directory.

## Capture side (frontend addon)

- One operator, exposed in the simulation panel, that captures the active
  scene; it must not write the `.blend`, and it must be undo-safe (no scene
  mutation).
- The same logic is reachable without the operator, so a background session can
  run it: `blender -b <file> --python <script>` with a call into the capture
  entry point.
- The rebuild path must work without a GPU context. Today the pattern refresh
  creates gizmo renderers (`Edge2D.update` and friends build `CurveRenderer`,
  which builds a GPU shader) and `setup_sewings_for_simulation` depends on a
  uuid map that the UI fills while it draws. The rebuild must be split so that
  the data path (sections, geometry, addressing, mesh regeneration) does not
  construct renderers, and the addressing must be buildable explicitly rather
  than as a side effect of drawing. A capture that needs a UI is not usable as
  regression material.
- Objects are ordered the way the engine expects: cloth first (the extension
  derives the cloth/non-cloth boundary from the object order and the
  `object_type` field), then obstacles.

## Consumer side (engine repository)

- A loader module under the test harness reads a package and returns the
  `input_data` payload plus a scene description (panels, seam chains, obstacle,
  parameter block) for the debug window and for reporting.
- A scene-registry entry (`frontend:<name>`) resolves a package by name, so
  existing cases, the batch runner and the debug window can select it without
  new code paths. It skips with setup instructions when the package is absent,
  matching the dataset-backed cases.
- The loader reuses the harness mesh-contract validation (triangle index range,
  edge integrity, per-object counts) so a corrupt capture fails with a named
  field instead of running.

## Round-trip contract

The capture is only useful if the reconstructed payload is what the frontend
sent. The verification is a fixture: a small synthetic scene is built in
Blender, captured, loaded in the harness without Blender, and compared against
the payload the frontend produced in the same session - integer arrays
identical, float arrays within tolerance, and the concatenation offsets
identical.

## Risks

- The capture is a snapshot: if the frontend changes its contract, the package
  format has to follow. A format version field plus the fixture above keep that
  honest.
- Captures are large (a garment scene is tens of thousands of vertices per
  panel); they live in the gitignored artifact tree and are never committed.
- The background-capture refactor touches code the UI depends on; the split must
  keep the interactive path identical (same addressing, same mesh regeneration)
  and be covered by the existing CPU-only registry tests.

## Open questions

- Whether the capture should also store the trailing simulation history (frames)
  for reference-metric comparisons, or only the initial state.
- Whether the operator should capture the parameter block from the running
  extension singleton or from an explicitly passed mapping; the notebook today
  sets parameters on the singleton, so the capture needs a documented rule.
