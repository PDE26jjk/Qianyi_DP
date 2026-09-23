## Why

The solver currently accelerates cloth with one global constant (gravity). Two
product targets cannot be expressed at all: garments that react to wind, and
closed shells (pillows, plush toys, balloons) that hold a puffed shape. The
inertia kernel already accepts a per-vertex `external_force` pointer that every
caller fills with `nullptr`, so what is missing is the force model and its
parameter surface, not a solver redesign.

## What Changes

- Add a per-object **constant normal pressure** model: the user sets one pressure
  value per panel/object; the force is applied along the face normal and scales
  with face area. No volume is tracked, no gas state is stored, and no global
  reduction or per-substep synchronization is introduced.
- Add a **wind** model: a world-space wind velocity field (constant base plus
  gusts/turbulence) evaluated at each vertex, driving an aerodynamic force with
  the quadratic lift/drag form published for Disney's Frozen.
- Apply both forces through the existing per-vertex external-force path, and only
  in the PDNewton solver. Other solvers ignore external forces and must say so
  instead of silently producing different results.
- Treat external forces as right-hand-side terms only: they contribute no
  Hessian and no diagonal entry to the linear system in this change. The design
  records the criterion for revisiting that decision.
- Extend the parameter surface: per-object `pressure` (with optional per-object
  wind coefficient overrides) in `input_data`, global wind parameters through
  `set_parameter` / `set_parameters`.
- Apply wind turbulence to the wind **velocity field**, never directly to the
  force, so gusts stay temporally continuous.

Non-goals for this change: volume constraints or any gas/air-mass model, filling
media (fiber, foam, granular particles), wind occlusion/shadowing between
surfaces, external-force support in the VBD/XPBD/explicit solvers, and a
pressure volume servo.

Disabling both features (pressure 0, wind off) must leave today's simulation
behavior unchanged.

## Capabilities

### New Capabilities

- `external-forces`: per-object constant normal pressure and a global wind field
  with quadratic lift/drag and noise, applied through the PDNewton external-force
  path with a documented force/Hessian policy.

### Modified Capabilities

None. Existing capabilities (`testing-harness`, `drape-debug-window`,
`real-time-performance`, `data-driven-drape-tests`) keep their requirements;
new debug scenes enter the registry under their existing rules.

## Impact

- Solver: `src/simulation/geometry.cu` (inertia kernel external-force wiring),
  `src/simulation/geometry.cuh` (force buffers), `src/simulation/solver_PDNewton.cu`
  (kernel arguments and warm-start handling), plus new force kernels.
- API and data model: `src/simulation/simulator.h` (`ObjectDataInput` per-object
  fields), `src/simulator_interface.cpp` and `src/main.cpp` (`input_data` keys and
  parameter plumbing).
- Debug tooling: new frontend scenes (balloon, windy cloth) registered under the
  existing scene-registry rules.
- Performance: one O(T) pass per substep for pressure and one O(T)/O(V) pass for
  wind; no new global synchronization and no change to the linear-system
  structure.
- References: the wind model follows the Disney Frozen SIGGRAPH 2014 talk
  ("Simulating Wind Effects on Cloth and Hair in Disney's Frozen"); pressure is
  the constant-load degenerate case of the standard gas-pressure force term.
