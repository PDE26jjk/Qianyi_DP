## 1. Reproduce and measure the contact-side jitter

- [x] 1.1 Two flat layers, frontend mesh and parameters (`step_h` 0.0045,
      `pd_iters` 5, `linear_iters` 2, `query_radius` 2 mm): 0.80 mm/substep
      with `constitutive_model_planar=1`, 1.04 mm with `0`
- [x] 1.2 Single 3 mm layer: 0.59 mm (planar 1), 3.04 mm (planar 0) - the
      residual is model independent, so it is the contact path
- [x] 1.3 Operator-row budget and broad-phase occupancy per scene
      (`contact_budget.py`): rows 16.6k / 29.1k / 33.3k against 3.5k / 71.3k /
      167.6k vf+ee candidates
- [x] 1.4 Marginal matvec cost from a `linear_iters` 2 -> 20 sweep: 1.45 ns per
      row plus 29 us fixed overhead per PCG iteration
- [x] 1.5 Classify the candidate set (`self_contact_probe.py`): the two-layer
      stack has ~1 valid cross-layer contact per vertex with a 7.9 um median
      penetration and no spurious self-contacts; the 3 mm flat layer has 13.8
      candidates per vertex and zero valid contacts
- [x] 1.6 Contact-force ablations: no contact forces -> 1.1e-4 mm (3 mm layer),
      2.5e-5 mm (two layers); stiffness 0.5 -> 0.1 -> 0.12 mm; `linear_iters`
      and `pd_iters` sweeps and 100x velocity damping do not change the jitter

## 2. Implement and measure the projection (not adopted)

- [x] 2.1 `Contact::project_contacts` with vf and ee narrow-phase sweeps on the
      existing candidate lists (`project_vf_kernel`, `project_ee_kernel`,
      `apply_contact_projection_kernel`)
- [x] 2.2 Safety bounds: `lambda >= 0`, under-relaxation, per-contact and
      accumulated per-vertex caps, zero weight for obstacles and pinned
      vertices
- [x] 2.3 Call it after the PD loop and before `step_end_kernel`; switches
      `contact_projection`, `contact_projection_iters`,
      `contact_projection_relax`, `contact_projection_max_step`
- [x] 2.4 Measured: no configuration reduces the jitter; settings below
      `relax` 0.1 are a no-op against the 7.9 um penetrations, settings above
      it make the jitter worse, and the candidate pass costs 2.5-3.5x the
      frame. The engine changes were reverted; only this record remains.

## 3. Verification (of the rejection)

- [x] 3.1 Reported scenes: jitter per substep (3 mm single, 8 mm two layers)
      with the projection on/off (`jitter_probe.py`, `jit_*.json`)
- [x] 3.2 Deep-penetration stress: two layers starting 2 mm inside each other
      (10x the contact thickness) with projection on/off: finite in both cases,
      but the projection makes the pile move more (0.80 -> 7.67 mm), so a
      "safety net" role is not demonstrated either
- [ ] 3.3 Quality invariants: not run, the change was rejected on 3.1/3.2
- [ ] 3.4 Cost: not run, measured inline as 2.5-3.5x per substep on the two
      scenes
- [x] 3.5 Regression groups after the revert: `-m quick`
      `-m algo`

## 4. Contact rows in the linear operator (implemented, measured, not shipped)

- [x] 4.1 Emit one compact row per validated vf/ee contact from the narrow
      phase (4 vertices, 4 coefficients, normal, stiffness); `A_mul_x`
      applies the rank-one term `K J J^T x`
- [x] 4.2 Rows are allocated outside the captured region and the row buffers
      are part of the CUDA-graph key (the first attempt failed with
      `cudaErrorStreamCaptureUnsupported` because the allocation ran inside
      the capture)
- [x] 4.3 Frontend scenes: jitter 0.79 -> 0.66 mm (two layers, planar 1),
      1.55 -> 1.23 mm (planar 0), 0.61 -> 0.57 mm and 3.06 -> 2.77 mm (3 mm
      layer) - a 6-21% reduction at +2% to +17% per step
- [x] 4.4 study-1: RTS 0.466 -> 0.271, `PCG ended with NaN residual` with
      friction on (RTS 0.213 with `friction_on=0`). Root cause: the emitted
      row carries only the normal penalty stiffness while the diagonal tiles
      also carry the friction stiffness
- [x] 4.5 Decision: not shipped; engine changes reverted, `-m quick` passes
      with the reverted tree

## 5. Making the friction term consistent (implemented, measured, not shipped)

- [x] 5.1 Rank-one slip-direction friction in diagonal + row: breaks the
      preconditioner (the per-vertex 3x3 block becomes rank two and
      `diag.inverse()` returns garbage), PCG NaN
- [x] 5.2 Isotropic friction in the diagonal plus two rank-one rows over an
      orthonormal tangent basis: consistent and stable in the small scenes,
      but the jitter benefit disappears (0.775 vs 0.789 mm on the two-layer
      stack; 0.613 vs 0.594 mm on the 3 mm layer) at +2% to +24% per step
- [x] 5.3 study-1: RTS 0.398 -> 0.267 (-33%), still `PCG ended with NaN
      residual` on a repeat run; with `friction_on=0` it runs at 0.280
- [x] 5.4 Decision: not shipped. The earlier benefit came from the
      inconsistent friction diagonal; a consistent operator has no benefit at
      this cost. Next attempt should start from a gather-based contact matvec
