## Context

See `proposal.md` for the motivation and the solver-damping change (tasks
2.28-2.30) for the audit that produced the gap list. The relevant current state:

- The shared physics lives outside the solvers: membranes in
  `src/simulation/dynamics/planar.cuh` (spring lattice `accumulate_spring_forces`,
  FEM `compute_BW_FEM`), bending in `src/simulation/dynamics/bending.cuh`
  (`compute_quadratic_bending_IBM`, `compute_dihedral_bending_GN`,
  `compute_dihedral_bending_AOGS`), stitching in `src/simulation/sewing.cu`
  (`accumulate_sewing_force` -> `compute_stitch_constraint`), contacts in
  `src/simulation/collision.cu` (`refit_bvh_with_target`,
  `accumulate_contact_force`, `collision_detect_broad_phase_stated`).
- Those kernels take an optional off-diagonal row buffer and an optional
  per-vertex `Jx_diag` block buffer, so the same kernel can serve a global
  system assembly (PDNewton) and a per-vertex block solve (VBD). PDNewton passes
  both; VBD needs the `Jx_diag` + force form only.
- VBD solves a colored per-vertex block descent: every iteration memsets
  `f`/`f_elastic`/`Jx_diag`, loops the color groups calling
  `solve_elasticity_springs_kernel` + `vbd_self_contact_kernel` +
  `apply_force_color_kernel`, then updates the contact duals. Its graph coloring
  already covers stitches (`Geometry::color_graph` builds `valid_edges` from
  edges *and* stitches), so stitch constraints can run inside the same loop.
- XPBD solves per-constraint projections with an optional multiplier array, then
  applies `vf`/`ee` contact projections, and its `step_end_kernel` computes
  velocities. Its broad phase currently comes from the frame-level
  `Geometry::collision_detect()` call in `Simulator::update`, which happens
  before the substep loop.
- Explicit integrates forces explicitly and then clamps velocity and ground.

## Goals / Non-Goals

**Goals:**

- Each experimental solver models membrane (both `constitutive_model_planar`
  options), bending (the `bending_model` options it supports, documented),
  stitch constraints and contacts, with the same `input_data` contract as
  PDNewton.
- Damping and collision-structure freshness become parameter/step driven and
  measurable, per the `experimental-solvers` spec.
- The harness ships a working parameter block per solver and runs their smoke
  test (including a seam check), so a regression in the wiring is caught.
- Every change is measured and recorded with before/after numbers, in the style
  of the solver-damping change.

**Non-Goals:**

- Drape-quality or cost parity with PDNewton. These stay test solvers.
- Tearing, plasticity or CUDA-graph capture for the three solvers (PDNewton-only
  features today). Seam projection *is* in scope: the maintainer's call was
  "geo's is generic anyway" - the projection is a geometry-side operation that
  belongs to every solver, so all three call it (see D9).
- Any change to the Blender frontend repository.
- Making Explicit real-time, or XPBD deterministic.

## Decisions

**D1 - Reuse the shared kernels through their `Jx_diag` + force interface
instead of writing solver-specific physics.** The bending and stitch kernels
already accept a per-vertex 3x3 block buffer plus a force buffer, which is
exactly what VBD's block solve consumes; the FEM kernel accepts a null
off-diagonal buffer. Alternatives considered: duplicating the math as
VBD/XPBD-only kernels (two copies of the same constitutive code, guaranteed
drift) or routing the experimental solvers through the global assembly and
PCG (defeats the point of having different solvers). Consequence: bending and
stitching stay identical to PDNewton's force/Hessian definitions wherever the
formulation allows it.

**D2 - VBD gets stitching and bending once per iteration (before the color
loop), and a FEM branch for the membrane.** Implementation note that corrected
this decision: a hinge spans four vertices in four different color groups and a
stitch pair is an edge of the coloring, so the element-parallel shared kernels
cannot run "per color" the way the spring kernel does. They run once per VBD
iteration, after the iteration's memset and before the color loop, accumulating
their per-vertex blocks into VBD's `Jx_diag` and their forces into `f`
(bending) / `f_elastic` (stitches). The consequence is explicit: springs stay
Gauss-Seidel (a vertex is solved against its neighbours' latest positions)
while bending and stitches are Jacobi terms evaluated at the iteration-start
positions. That is the cost of reusing the shared kernels instead of duplicating
their math; the probes in task 7.1 decide whether it is acceptable, and the
fallback is a per-vertex hinge/stitch incidence table (`edge_lookup` shape)
with a per-vertex gather kernel. The membrane reads `constitutive_model_planar`
and dispatches to the spring kernel or to `compute_BW_FEM` (with the same PSD
clamp / shear flags PDNewton uses).

**D3 - XPBD gets bending and stitching as compliance constraints, not as
penalty projections of the force kernels.** XPBD's contract is per-constraint
projection with compliance `alpha = 1/k` and optional multipliers, so bending is
implemented as a dihedral constraint over the existing `bend_points` /
`bend_rest_theta` / `bend_factor` tables, and stitching as a zero-rest-length
distance constraint over `stitches` (skipping torn stitches). Alternatives:
projecting the PDNewton bending force (mixes two formulations in one iteration
and needs the Hessian assembly), or skipping bending for XPBD (leaves the
skipped smoke test justified). The existing unused `xpbd_use_lambdas` path is
either completed with these constraints or removed, so no dead switch is left
in the shipped parameter block.

**D4 - Damping becomes one parameter name for all solvers.** Each experimental
solver reads `velocity_damping` (same semantics as PDNewton: 1/s, applied as
`v * exp(-h * damping)` per substep) and stops applying the fixed
`v * exp(-h * 0.5)` decay. The legacy `vbd_damping` / `xpbd_damping` keys stay
readable as aliases so existing scenes keep working: the solver reads
`velocity_damping` and falls back to its legacy key when it is not set
(negative-default sentinel). The presets are updated to set `velocity_damping`
explicitly.

**D5 - XPBD and Explicit refresh the collision structure per substep.** Both
call the same per-substep refresh PDNewton and VBD use (`refit_bvh_with_target`
for the frame's target positions, or the substep's own prediction where the
solver has one) before the contact kernels, instead of relying on the
frame-level broad phase. Alternatives: refreshing every N substeps for cost
(cheaper but step-count dependent, and the spec asks for per-substep freshness)
or leaving it (the current behaviour, which is what the spec change exists to
fix). Cost is measured; if the refresh dominates, the measurement is recorded
and the fallback is a documented `collision_refresh_every` knob, not silence.

**D6 - Explicit keeps its explicit integrator.** It gains the FEM membrane path
(its call is commented out), the AOGS bending model, per-substep collision
refresh and parameter-driven damping, but no implicit term: its stable
`step_h` and its cost stay what they are, documented as such. Alternatives:
converting Explicit to semi-implicit (it would stop being an Explicit
reference).

**D7 - The harness carries the acceptance, not the engine.** The re-enabled
smoke test covers finite data, pinned drift, free motion and seam closure with
the solver's shipped block. A new procedural two-panel stitch scene is built in
the harness (`MeshSpec` output plus `sewings` with index pairs) so the seam
check needs no Blender and no dataset. Per `AGENTS.md`, adding or re-enabling
tests needs the maintainer's explicit approval, so the change records that as
its first task rather than assuming it.

**D8 - Measurements use the existing headless probes.** The two-panel stitch
probe and the garment probe built for the audit (gitignored, `build/`) are the
before/after instruments; their numbers go into the change's tasks file, and any
change that turns out ineffective is removed and recorded instead of kept.

**D9 - The seam projection is wired into every solver.** `Geometry::project_stitches`
(cluster centroid projection, gated by `sewing_forced_connect_frame`, the ramping
`sewing_snap_dist` gate and `sewing_snap_max_dist`) plus its matching
`average_stitch_cluster_velocities` pass are solver-independent, and PDNewton
already called them; VBD, XPBD and Explicit now call the same pair at the end of
each substep. Placement differs by solver and is deliberate: the position-based
solvers (VBD, XPBD) project after their position update so the welded position is
what the next substep reads, and Explicit projects after `step_end_kernel`
because its integrator overwrites the position from the velocity and would
discard an earlier projection. The soft stitch constraint stays in all three
(it is what holds a seam together *before* the gate opens, and what the seam test
measures); the projection is what welds it shut afterwards. Measured: with the
gate lowered to frame 10, all four solvers report a 0.000 mm seam gap at 60 and
at 200 frames, while the `sewing_k = 0` control still separates by metres.

## Risks / Trade-offs

- [XPBD's instability may not be a wiring problem] -> The audit measured 456 mm
  of run-to-run difference on a 10-frame drape. Wiring bending/stitching may not
  fix it; the acceptance bar is the smoke scene plus seam closure, and if
  repeatability stays unusable the finding is recorded and XPBD stays a known
  failure with a measured failure mode rather than a green test.
- [Bending destabilizes the block solve] -> The dihedral Hessian is not
  guaranteed positive definite. Mitigation: reuse PDNewton's PSD clamp path,
  start with a small `bending_k` in the preset, and fall back to the
  diagonal-only (regularized) block when a block is indefinite.
- [Per-substep collision refresh multiplies XPBD's cost] -> XPBD runs 42
  substeps per frame at `step_h = 1 ms`; the refresh is O(substeps). Measured
  before/after; if it dominates, D5's fallback knob applies and the number is
  recorded.
- [Test-suite time grows] -> The experimental smoke cases are 60 frames each;
  they stay outside the `quick` marker if their runtime makes the quick suite
  slow, with the marker decision recorded.
- [Silent partial wiring, the failure this change exists to fix] -> Each wiring
  step has a scenario in the `experimental-solvers` spec, and the seam check
  fails loudly when a solver ignores `sewings` (the audit's XPBD result was a
  5.8 m free fall, which no tolerance hides).

## Migration Plan

No API or data migration: the solver names, the `input_data` contract and the
PDNewton path are unchanged. Rollback is a revert of the solver files plus the
preset values; the parameter aliases in D4 keep old scenes loadable either way.
