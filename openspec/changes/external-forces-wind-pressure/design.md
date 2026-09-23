## Context

See proposal.md - Why. The relevant current state:

- The PDNewton inertia kernel already takes an `external_force` pointer and adds
  it to the external acceleration, but the solver passes `nullptr`. The explicit
  solver has an equivalent `other_forces` parameter. In other words the plumbing
  exists and is unused.
- PDNewton already works with an approximated system: a block-diagonal element
  Hessian plus a scalar per-vertex diagonal built from the inertia term
  (`mass / h^2`), the projective lattice row sum, and the mask stiffness. The
  element Hessian is refreshed on a configurable cadence, so the solver is
  already a lagged quasi-Newton method.
- The warm start heuristic projects the previous acceleration onto the current
  external acceleration direction, which today is essentially gravity. A
  per-vertex, position-dependent force breaks the assumption behind that
  projection.
- Sewing is a permanent zero-rest-length stitch spring, not a topological weld.
  Panels therefore stay separate vertex sets joined by constraints, and a stitch
  cluster subsystem already exists for snapping and velocity averaging.
- The planar element energy is area-weighted (an energy density times triangle
  area), while the bending element and the stitch spring carry a per-element
  stiffness. This matters when asking how an inflated shape depends on mesh
  density.

## Goals / Non-Goals

Goals:

- Add wind and internal pressure with no new global state, no new
  synchronization point, and no change to the linear-system structure.
- Keep the model parameter set small enough to tune by hand.
- Keep the disabled path byte-for-byte equivalent in behavior to today's
  simulation.

Non-Goals (design level, in addition to the proposal's list):

- No watertightness requirement. Pressure is a distributed load, not a gas state,
  so open panels and holes are legal inputs.
- No seam-specific handling in this change. The seam stays what it is today
  (a stitch spring); pressure is applied as a per-face load and nothing about
  the stitching contract changes.
- No attempt to make the inflated size independent of mesh density beyond what
  the existing material parameterization already gives.

## Decisions

### Decision: pressure is a constant distributed normal load, not a gas state

Force per face is `p * A * n`, split equally to the three vertices, where `p` is
the per-object pressure value, `A` the current triangle area, and `n` its unit
normal. Positive pushes along the normal, negative pulls inward. This is the
degenerate case of the standard gas-pressure term in which the pressure is a
user parameter instead of a state variable.

The mathematical reason the volume never has to be computed: for a uniform
pressure the force is `F = -p * dV/dx`, and the gradient `dV/dx_i` is the local
area-weighted vertex normal (one third of the sum of the incident face area
vectors). The volume only appears through its *gradient*, and that gradient is
local; only the second derivative would need volume machinery. Uniform pressure
on a closed surface therefore produces zero net force while still inflating the
shell, which is exactly the balloon behavior we want.

Which side a panel inflates towards is decided by its **3D placement**, not by
the winding in the input data. At load, `init_triangle_data` re-orders every
cloth triangle in the pattern plane (`vertices`) so its pattern-space normal has
a `+Z` component, and the cloth normal array is zero-initialised with `Z = 1`;
the input winding is therefore normalized away. The force itself is evaluated on
the simulated 3D positions, so the world-space normal is the placed panel's
outward normal (`vertices_sim` plus the object world matrix). A back-to-back
pair must be placed with opposite outward directions, exactly as pattern pieces
are placed in 3D by the frontend; a winding flip in the input has no effect.

Alternatives considered:

- Ideal-gas pressure `p = nRT / V`: requires a per-substep global reduction, a
  watertightness contract, and volume clamping to avoid a blow-up as `V -> 0`.
  Rejected for this change.
- A PBD/XPBD volume constraint: the stiffest and most accurate option, and the
  upgrade path if hand-tuning ever stops being enough. Deferred, not rejected;
  the cheapest form of it is a volume servo that reads one scalar reduction per
  frame and holds it constant across substeps.

### Decision: wind uses the Frozen (SIGGRAPH 2014) quadratic lift/drag model

Per face, with `v = v_wind(x, t) - v_vertex` (air velocity relative to the
surface, so a positive normal projection means the air pushes the surface):

```
F = 0.5 * rho * A * [ (C_D - C_L) * (v . n) * v + C_L * |v|^2 * n ]
```

which is the separated form of "drag along the relative velocity, lift along the
normal component perpendicular to it". `rho` is air density, `C_D` and `C_L` are
user coefficients, `n` is the face normal. Setting `C_L = 0` reduces to the
classic quadratic normal drag used by Bridson-style cloth aerodynamics; this
model was chosen over the linear normal-projection variant because the published
Frozen work switched from Stokes drag to the quadratic form precisely for
high-Reynolds-number air and richer turbulent response.

The convention is pinned here so it cannot drift: `F` is the force the air
applies to the surface, `v` is air minus cloth, and the result is a force per
unit area that is multiplied by the face area and distributed to its vertices.

Alternatives considered:

- Linear normal-only pressure (the Blender formulation): cheaper and stable, but
  produces no force for grazing flow and no lift, which is visible on flags and
  loose garments. Kept as a possible fallback if the quadratic term proves too
  stiff at the shipped step size.
- Two-way fluid coupling: out of scope for a real-time garment tool.

### Decision: turbulence perturbs the wind velocity field, never the force

Gusts are a low-frequency temporal signal plus spatial noise (curl noise is
preferred because it is divergence-free and reads as eddies rather than jitter).
Adding noise to the force instead would inject impulses at the noise frequency
and read as impacts rather than airflow.

### Decision: external forces contribute to the right-hand side only

No Hessian or diagonal contribution is added for pressure or wind in this
change. Rationale:

- Equilibrium is set by force balance; the Hessian only affects convergence rate
  and preconditioning quality.
- Gravity is already handled this way, and the solver already lags and
  approximates its element Hessian, so this is a continuation of the existing
  approximation rather than a new one.
- The pressure Hessian is `-p * d2V/dx2`; skipping it is the direct consequence
  of choosing not to evaluate volume, so the cost is paid once.

Criterion to revisit, evaluated from recorded data rather than by guesswork: the
pressure contribution scales like `p * A / h` against an elastic scale of
`k * A`, so the dimensionless ratio `p / (k * h)` decides whether the omitted
term is negligible. During implementation, sweep pressure on the reference
balloon and compare the reported per-substep iteration count and residual
against the no-external-force run; if the convergence degrades materially, add a
diagonal approximation first (`p * A / h`-scale) before considering a full
treatment.

Measured on the reference balloon at 600 Pa: the linear solve's relative
residual moved from 0.0053 (no pressure) to 0.0063 and the Newton relative
residual improved, so the omitted term is not material at the reference
magnitude and no diagonal approximation was added. The criterion stays recorded
for larger pressure values.

The one term that is not a Hessian question: the drag's dependence on velocity.
If the quadratic drag is too stiff at the shipped step size, treat drag
semi-implicitly in velocity (a damping-like diagonal update) rather than moving
position-dependent terms into the matrix. The Frozen talk's reported trick -
substituting `v -> (v . n) n` after differentiation to keep the force gradient
symmetric positive definite - is the reference approach for that.

### Decision: the warm-start path is untouched

The existing projection assumes a globally aligned external acceleration, and a
per-vertex, position-dependent pressure or wind load does not fit that
assumption. It is nevertheless left exactly as it was, with no new parameter:
the projection only runs for `warm_start = 1` (the VBD-style predictor), while
the shipped default is `warm_start = 2` (full inertia prediction), which never
reads the projected direction. Adding a switch for a path the default does not
take was judged not worth the surface area. If a scene ever ships with
`warm_start = 1` and external forces, revisit this: the projection would then
be steering on a load it was not designed for.

### Decision: external forces are PDNewton-only

PDNewton is the only production-ready solver (VBD / XPBD / Explicit are
experimental and research-only, see the `testing-harness` spec's "Experimental
solver governance") and the only one with the external-force hook wired. Other
solvers simply ignore external forces: their scenes still run, without an
external-force term. This keeps the experimental-solver comparison work
meaningful and avoids a divergent behavior matrix.

### Decision: parameters live in two places with inert defaults

Per-object values (pressure and optional wind coefficient overrides) belong to
the per-object data that `input_data` already carries alongside stretch,
bending, and mass density. Global wind values go through the existing scalar
parameter map; because that map is string-to-float, the wind velocity is exposed
as components (`wind_x`, `wind_y`, `wind_z`) and a magnitude is derived, rather
than introducing a new parameter type in this change.

### Decision: no orientation check for pressure-enabled objects

An earlier draft reported pressure-enabled objects whose input triangle winding
was inconsistent. It was removed after review: pressure is a cloth-only input,
and cloth triangles are re-oriented in the pattern plane to the same `+Z` normal
at load, so the input winding cannot reach the force at all. The check could
only ever fire on input the engine normalizes anyway, which made it dead weight
rather than a diagnostic.

## Risks / Trade-offs

[Seam-region load imbalance] → Accepted as an observation item, not a design
item, per the maintainer's direction. The load near a stitch ring is not that of
a closed surface because the two panels remain separate vertex sets joined by a
spring. Watch the reference balloon scene for a pinched or pulled seam; the
available remedies if it shows (a per-vertex pressure weight, or accumulating
per-face contributions through the existing stitch clusters) are both local and
additive, so deferring them costs nothing structural. One guard is kept from the
start: pressure load must not feed the stitch tearing criterion.

[Inflated size varies with mesh density] → Partly already handled: the planar
element is area-weighted, so the in-plane response is close to
resolution-independent. The bending element and the stitch spring are
per-element, so their effective stiffness does change with subdivision. Verify
with one comparison (same pressure, two mesh densities) on the balloon scene
before doing anything; if the drift is visible, the fix is to normalize those
per-element stiffnesses for pressure-enabled objects, not to change the pressure
model.

[Compressed shell feels soft] → Accepted and documented. With no volume
feedback, squeezing reduces the projected area and therefore the force, which is
the opposite of a real gas. This is the intended trade-off for a balloon-like
model; the name "constant normal pressure" is used everywhere so it is not
mistaken for a gas model.

[Coefficient convention drift] → Mitigated by pinning the formula, the sign of
`v`, and the units in this document, and by calibrating once against the
reference scenes rather than re-tuning per scene.

[Frame budget] → One O(T) pass per substep for pressure and one O(T) pass for
wind (reusing the face normals and areas already computed for the element
terms), with no new global synchronization. Measure against the recorded
per-frame budget rather than assuming it is free.

## Verification results (local, gitignored script)

Measured against the Release build with a band-limited contact radius so the
scenes isolate the force models (test code is not in the repository, per the
maintainer's instruction):

- Inert by default: a corner-pinned scene with no external-force inputs and the
  same scene with explicit zero wind differ by 2.98e-07 m, inside the 4.17e-07 m
  run-to-run spread of the engine itself.
- Pressure: a two-panel sewn shell (back panel placed with a 180-degree X
  world-matrix rotation) inflates monotonically with pressure - mean panel gap
  0.0006 m at 0 Pa, 0.0122 m at 100 Pa, 0.0172 m at 300 Pa, 0.0232 m at 600 Pa,
  all frames finite. Zero pressure leaves the shell flat; negative pressure
  pulls the panel inward (-7.6 mm at -200 Pa on a rim-pinned panel).
- Wind: a 1 m2 hanging panel at 6 m/s deflects 13 mm downwind after 12 frames
  and 31 mm at 12 m/s, i.e. monotone in wind speed in the early window; the
  gust + turbulence run stays finite with a 2 mm maximum per-frame step. A
  grazing wind (along the panel plane) produces almost no force, which is
  expected for a normal-projection model.
- Mesh-density drift: the same 300 Pa on a 9x9 and a 17x17 balloon differs by
  17.6% in peak inflated gap. Real, and consistent with the per-element
  stiffness of the bending element and the stitch springs.
- Frame cost: +0.14 ms/frame on a 129x129 cloth (~16.6k vertices, ~32k
  triangles) with base wind + gust + turbulence, measured after warm-up.
- Regression: the quick group has one pre-existing failure
  (`sim/test_smoke.py::test_standard_scene_smoke`, the sheet hovers ~11.25 mm
  above the ground instead of sagging). Rebuilding the same test from the
  unmodified HEAD reproduces it identically (0.000297 m / 11.2541 mm against
  0.000299 m / 11.2453 mm with the change), so it is unrelated to this work.

## Migration Plan

Additive only. New per-object fields default to inert values and new global
parameters are unset by default, so existing scenes, parameter blocks, and the
frontend driver keep working unchanged. Rollback is unsetting the parameters;
there is no persisted state or data migration.

## Open Questions

- The shipped defaults for air density and the lift/drag coefficients are
  decided during calibration against the reference scenes. This affects tuning
  values only, not the spec, the approach, or the task breakdown.
- Whether the wind velocity should eventually become a first-class vector
  parameter is deferred; the component form is sufficient for this change and
  does not constrain the internal representation.
