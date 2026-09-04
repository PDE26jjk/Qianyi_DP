## Context

The harness currently drives procedural grid scenes only. GarmentCodeData elements
were inspected to ground this design: box meshes are in centimetres
(`units_in_meter: 100`), carry UV-duplicate vertices (a 29.8k-vertex element maps
to 28.8k original IDs), label every vertex with a panel or `stitch_N` seam label,
and ship a reference draped mesh (`sim.ply`). A sampled 60-element scan shows
triangle counts from 3.4k to 88k (median 21.7k). The neutral body mesh matches
the element body measurements exactly, so one body mesh serves all elements in
the downloaded set. See proposal.md for motivation and the spec for requirements.

## Goals / Non-Goals

**Goals:**
- A repeatable loader + sampler that turns GarmentCodeData elements into the
  existing `input_data` contract with zero manual steps.
- Two-tier verification (invariants with gravity off when no body is present;
  loose reference comparison when the body is configured) and a performance
  record per size bucket.
- An interactive Warp viewer for inspecting individual cases and batch failures.

**Non-Goals:**
- Bit-exact reproduction of the dataset drapes (different solver/materials).
- Matching the dataset fabric parameters to our material model.
- Fixing solver determinism or SDF integration.
- Hard performance thresholds in this phase (record-only).

## Decisions

### D1. Dataset access is environment-configured
Dataset root and neutral body path come from `QYDP_GCD_ROOT` / `QYDP_GCD_BODY`
environment variables; concrete paths live in `LOCAL_DEV.md` only. The pytest
fixture skips data tests with setup instructions when either is missing.

### D2. Loader: trimesh for PLY parsing, our own topology mapping
Use `trimesh` (`process=False`) to read box mesh vertices/faces/UVs, then apply
the documented duplicate-vertex map (`v_id_map`: consecutive identical positions
share an original ID) ourselves so segmentation aligns with original IDs.
Alternative considered: hand-rolled binary-PLY reader (no dependency) - rejected
because trimesh is the dataset's documented tooling and provides reference-metric
utilities later.

### D3. Units and axis: dataset is centimetres, Y-up
All dataset meshes are converted to metres (×0.01) before entering
`input_data`, matching the engine's metre contract. The dataset is stored with
the garment height along **Y** while the engine's gravity is fixed along -Z
(`make_float3(0, 0, gravity_z)` in `geometry.cu`), so both the garment and the
body are rotated to Z-up at load time (`(x, y, z) -> (x, -z, y)`). The
reference drape (`sim.ply`) receives the same rotation for the reference
metrics. Scale/alignment is verified per element by checking the box mesh
bounding box against the body bounding box (calibration task before mass
runs; the neutral body is already in metres).

### D4. One mesh per panel
Each segmentation panel becomes one cloth mesh object in `mesh_list` (edges
derived from that panel's triangles), matching the frontend convention of one
pattern = one object and keeping per-panel fabric settings possible later.
Alternative: one merged mesh (simpler offsets) - rejected because it loses
per-object material/collision settings.

### D5. Stitch pairing from seam chains, cross-checked with the specification
A `stitch_N` label marks an entire seam chain (verified: up to 73 vertices on one
label), not a single vertex pair. The loader orders each chain as a single open
path via mesh-edge adjacency among stitch-labeled vertices. Because the box
mesh is a **closed garment**, the two panels' boundary chains coincide along
each seam curve (every seam vertex has one UV-duplicate copy per side), so the
engine's sewing constraint (a vertex-pair spring, `compute_stitch_constraint`)
is fed identity-position cross-panel pairs `(panel A : p, panel B : p)`, with
each panel traversing the full chain so the engine's init path (which requires
consecutive stitches to be edge-adjacent on each side) stays consistent.
Same-panel seams (darts) emit one internal entry. The reference drape confirms
the pair structure (consecutive chain positions coincide at distance ~0).
The specification JSON's panel-edge stitches are used as a cross-check;
mismatches are reported per element and non-simple chains (branched or
multi-component) are a classified loader error.

### D5a. Degenerate triangles and seam edges (verified implementation detail)
After UV-duplicate collapse, seam-bridging triangles can reference one original
id twice (zero area) or form near-degenerate slivers (verified aspect ratios
above 1e6 on seam-adjacent triangles). Such triangles are dropped from the
panel meshes (relative area threshold `max(1e-8, 1e-3 * panel median)`);
otherwise the zero-area triangles give vertices zero mass and the linear solve
diverges (`PCG NaN`, 1/0 inverse mass). Dropping them can remove a seam-chain
edge, which is then re-added to the owning panels' edge lists so the engine's
sewing init finds every consecutive-stitch edge.

### D5b. Frame analysis uses local-space output (verified engine quirk)
`get_simulation_data(world_space=True)` leaves stale positions for
proxy-merged seam vertices (the world buffer is not fully remapped after the
sewing proxy merge), while the local buffer stays consistent. With identity
world matrices the two agree everywhere else, so all invariant/reference
analysis uses local-space frames.

### D6. Attachments are optional
Semantic vertex labels (waist/collar/armhole) are exposed as `attached_vertices`
but not required by either tier: the reference tier relies on body collision, and
the no-body tier runs with gravity disabled (D7), so the garment stays near its
initial box mesh.

### D7. No-body tier disables gravity
Without a body mesh, `gravity` is set to 0 in the run preset so the box mesh does
not collapse to the ground; seam closure and invariants remain meaningful.

### D8. Body obstacle
The neutral body OBJ is loaded as an `object_type=1` obstacle with per-face
normals computed from its triangles (matching the frontend's obstacle contract).
Body and garment scales are aligned in the same calibration task as D3.

### D9. Stratified sampling manifest
The sampler reads `design_params.yaml` (garment type from the upper/wb/bottom
template tree) and box-mesh face counts, buckets elements (S < 5k, M 5-25k,
L 25-50k, XL > 50k faces), and writes a reproducible manifest (fixed seed) with
the selected element ids. The fabric-batch axis is intentionally omitted: our
material model does not correspond to the dataset materials.

### D10. Isolated batch execution
Each element runs in a fresh subprocess (reusing the determinism-subprocess
pattern) with a per-case timeout and fd-level output capture, producing the
standard per-case artifacts plus a failure class. This isolates crashes and the
singleton simulator state, which matters at this scale.

### D11. Reference metrics are loose and record-only until calibrated
Metrics: per-panel area ratio, z-extent, horizontal cross-section quantiles
(bust/waist/hip heights), and mean surface distance. A calibration run over a
small hand-checked subset establishes default tolerances; until then the default
mode records metrics without hard-failing. The spec's failure behavior activates
once tolerances are configured.

### D12. Warp viewer follows the frontend rendering pattern
The notebook uses `wp.render.OpenGLRenderer` with a frame loop rendering the box
mesh, simulated frames (`frames.npz`), reference drape, and body, with keyboard
frame-scrubbing and overlay toggles; failing cases are listed from the batch
`results.json` and can be opened directly. The viewer assumes a CUDA-capable
Warp runtime, which is a given in this project's test environment; no non-GPU
fallback is provided. This replaces the sparse `warp_scenes.ipynb`.

## Risks / Trade-offs

- [Seam chains with T-junctions or unequal side counts may not pair cleanly] →
  cross-check against the specification; unpaired seams are classified loader
  errors and a manual override list can be added.
- [Reference comparison may fail systematically because our draping differs] →
  metrics chosen to be geometry-insensitive; record-only default until a
  calibrated baseline exists.
- [XL elements (up to 88k faces) are slow (seconds per substep today) and may
  exceed VRAM] → size caps for default runs, per-case timeouts, XL excluded from
  the default manifest.
- [Box/body scale mismatch from centimetre data] → calibration task verifies
  bounding-box alignment before mass runs.

## Migration Plan

Additive: new `tests/data/` cases, `tests/harness/gcd/` loader/sampler, notebook
replacement, `requirements-dev.txt` (`trimesh`), `AGENTS.md` entries, marker
registration in `pyproject.toml`. Rollback: remove those additions; no `src/`
changes and no existing behavior is modified.
