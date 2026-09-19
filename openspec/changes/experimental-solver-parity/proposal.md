## Why

VBD, XPBD and Explicit are the only alternatives to PDNewton, they are selectable
from the Blender frontend, and the cross-solver audit (solver-damping task 2.30)
showed each of them is missing pieces of the shared pipeline rather than being
wrong in its own method: VBD ignores the constitutive-model switch and never
applies the bending or stitch terms, XPBD and Explicit never apply bending or
stitching either and only refresh their broad phase once per frame, and both
hard-code a velocity decay instead of reading the damping parameter. Their smoke
tests are skipped, so nothing detects this.

They are wanted as *test* solvers: cheap-to-switch references for convergence,
stability and performance comparisons against PDNewton. That use needs correct
physics wiring and a green test, not production drape quality.

## What Changes

- **VBD**: solve the membrane according to `constitutive_model_planar` (spring
  lattice and FEM_BW, as PDNewton and XPBD already do), add the bending dispatch
  (IBM quadratic / DiscreteShells GN / DiscreteShells AOGS) and the stitch
  constraint to the colored block solve, keeping the existing per-substep stated
  contact path.
- **XPBD**: add bending constraints and the stitch constraint, refresh the
  broad phase per substep instead of once per frame, read a damping parameter
  instead of the hard-coded `exp(-h*0.5)`, and either finish or remove the
  unused multiplier (`xpbd_use_lambdas`) path so the compliance formulation is
  unambiguous.
- **Explicit**: enable the FEM_BW membrane path (its call is commented out
  today), add the AOGS bending model, refresh collisions per substep, and read
  the damping parameter instead of the hard-coded `exp(-h*0.5)`.
- **Harness**: update each experimental solver's parameter block to what the
  solver actually implements, and re-enable the experimental solver smoke test
  (currently a skipped strict xfail) with a stiffness-independent seam check so
  the stitch wiring is covered.
- Record the measured cost and behaviour of each solver in the change so
  "test solver" keeps a defined meaning.

Not in scope: the Blender frontend's solver list (another agent owns that
repository), PDNewton's own convergence work, and any new production-quality
claim for these three solvers.

**BREAKING**: none. The public API keeps the same four solver names; the
changes are engine-side wiring plus harness parameter values.

## Capabilities

### New Capabilities

- `experimental-solvers`: what VBD, XPBD and Explicit must model to be usable as
  test solvers (membrane model dispatch, bending, stitching, collision refresh
  cadence, damping wiring) and what they are explicitly not required to do.

### Modified Capabilities

- `testing-harness`: the experimental solver governance requirement changes from
  "mark VBD / XPBD / Explicit tests as known failures" to "run their smoke test
  as a normal test, including a seam-closure check", once they are wired up.

## Impact

- Engine: `src/simulation/solver_VBD.cu`, `solver_XPBD.cu`, `solver_explicit.cu`,
  plus shared kernels in `src/simulation/dynamics/{planar,bending}.cuh`,
  `src/simulation/sewing.cu` and the contact path in `src/simulation/collision.cu`
  where a solver needs a different entry point (for example a per-substep
  broad-phase refresh for XPBD / Explicit).
- Harness: `tests/harness/presets.py` (experimental parameter blocks) and
  `tests/sim/test_experimental_solvers.py` (re-enabled, plus the seam check).
- No change to `qydp.simulator` signatures, to `get_all_solver()`, or to the
  PDNewton path.
- Verification uses the existing headless probes and the quick suite; no new
  external dependency.
