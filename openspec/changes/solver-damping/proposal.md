# Proposal: solver-damping

## Why

The PDNewton path has no effective energy dissipation. A settled garment keeps
moving forever instead of coming to rest, which the frontend reports as "the
cloth cannot stabilize, it keeps shaking like rubber". Reproduced headlessly on
a sewn garment draped over a body (two cloth panels, ~26k cloth vertices,
3.63 mm median edge, 2 mm thickness, 116 stitch pairs already closed, no pinned
vertices): over 27 s of simulated time the tail-window median per-substep
displacement stays at 0.21-0.25 mm per 4.5 ms substep and never decays, with the
per-vertex maximum at 1.1-2.0 mm and a displacement-coherence of 0.1-0.5 (so it
is per-vertex chatter, not a coherent swing). Removing gravity makes the same
scene decay. The motion is insensitive to membrane stiffness (x5), mass (x1000),
PD/PCG iteration counts (x2/x5) and contact stiffness, which points at missing
dissipation rather than at an unconverged solve; the earlier contact-stability
investigation already measured that a 100x global velocity multiplier does not
change its ringing either.

The only dissipation in the engine today is a hard-coded `v *= exp(-h * 0.5)`
in the PDNewton end-of-step kernel and a frame-level Laplacian smoothing of the
velocity field. Neither can act on the elastic modes: the first is a fixed
0.5/s floor that is not even exposed as a parameter, and the second only
low-passes a velocity that the next substep re-derives from positions.

## What Changes

- Add implicit damping to the PDNewton solve: a stiffness-proportional
  (Rayleigh `beta`) term folded into the assembled tangent used by the linear
  solve, covering the membrane, bending, stitch and contact blocks, and a
  mass-proportional (`alpha`) term for the low-frequency and rigid-body modes.
  Both are exposed through the existing parameter map and both are applied
  inside the iteration, so they cannot introduce a new step-size stability
  limit.
- Add relative-velocity damping to the contact response: a semi-implicit normal
  dashpot that dissipates the relative normal velocity of an active pair. The
  positional penalty alone cannot do this, and Coulomb friction is tangential,
  load-capped and slip-based, so neither can break a contact force limit cycle.
- Make the seam cluster merge velocity-consistent. The merge is a hard position
  projection by design (the trajectory envelope had to go: box-layout stitch
  pairs start tens of centimetres apart and never became eligible with it), but
  the following velocity update currently reads the teleport as a velocity and
  injects kinetic energy. Reuse the momentum-preserving cluster velocity average
  that is already implemented but not called, or fold the projection into the
  velocity update.
- Turn the hard-coded end-of-step velocity damping into a documented parameter
  (default unchanged) and document it as a floor, not the primary mechanism.
- Keep the frame-end velocity Laplacian smoothing and its anti-blow-up
  semantics, but fix its launch geometry, which currently launches
  `num_vertices + block - 1` blocks instead of `ceil(num_vertices / block)` for
  a kernel that does 258x redundant work per pass on the reported scene (26,211
  blocks of 256 threads for 25,956 vertices), each thread exiting on a bounds
  check. Measured cost of one pass, three same-batch A/B pairs: 0.02, 0.14 and
  0.24 ms, i.e. at most about 1.2 ms of a 19-32 ms substep for the five passes
  of a frame; the measurement is noisy because the GPU is shared with an
  interactive session.
- Verification: add a resting-stability metric (tail-window per-substep motion
  at rest), a ring-down metric (decay with the drive removed), a substep-size
  invariance check, and an "equilibrium is unchanged" check to the simulation
  verification path.

## Capabilities

### New Capabilities

- `solver-damping`: dissipation guarantees for the cloth solvers - a resting
  configuration settles, an undriven configuration loses kinetic energy at a
  bounded rate, damping is time-step consistent, damping does not move the
  static equilibrium, and the dissipation mechanisms (operator-level Rayleigh
  damping, contact relative-velocity damping, velocity-consistent seam merge)
  are controllable through the parameter map without regressing the existing
  quality invariants or the frame budget.

### Modified Capabilities

None. The existing `testing-harness`, `real-time-performance`,
`data-driven-drape-tests` and `drape-debug-window` requirements stay as they
are; the new measurements are additional cases in the harness, not new
requirements on it.

## Impact

- Engine: `src/simulation/solver_PDNewton.cu` (iteration assembly, end-of-step
  kernel, the CUDA-graph capture key), `src/simulation/collision.cu` and the
  contact narrow phase (normal force and Hessian), `src/simulation/sewing.cu`
  (cluster merge and the cluster velocity pass), `src/simulation/geometry.cu`
  (frame-end smoothing launch geometry).
- API: no Python API change. New parameter names only, so a scene can A/B the
  mechanisms in one binary; `0` reproduces today's behavior.
- Behavior: damping changes drape dynamics by design (that is the point). The
  static equilibrium must not move, and the drape acceptance invariants (seam
  closure, area, penetration, finite frames) must not regress.
- Verification: the harness gains resting-stability and ring-down cases; the
  reported frontend scene is the reference case.
