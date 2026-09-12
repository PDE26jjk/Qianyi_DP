## 1. Pin the baseline

- [x] 1.1 Reproduce the jitter scenes headlessly with the shipped parameter
      block and record, for the tail window: median and maximum per-substep
      displacement, the moving-vertex count, and the finite check.
      Reproduced: two flat layers 8 mm mesh 0.78-0.80 mm per substep;
      two layers 3 mm mesh 0.71 mm; a single 3 mm layer 0.61-0.81 mm; a single
      8 mm layer clean (7e-6 mm). All frames finite. Motion is 96% in plane
      (47 um tangential against 2 um normal per substep) and the layer gap is
      stable to 0.3 um, so this is not a normal-direction ring.
- [x] 1.2 Record the baseline real-time speed of the garment scene and the
      invariant results of the drape acceptance set. Garment scene measured at
      ~10-15 ms per 1 ms `update` call (solver only, no render), which is the
      RTS ~0.4 the study notebook reports at `step_h` 4.5 ms; the drape
      acceptance set passes (3 passed, seams closed 98/98).
- [x] 1.3 Record the penetration measure. Two-layer stack: median valid
      penetration 7.9-9.1 um, deepest 24-200 um, minimum inter-layer gap
      105 um where two 0.1 mm shells imply ~190 um - i.e. the layers locally
      squeezed through each other while the ringing was happening.

## 2. Spikes that decide the mechanism

- [x] 2.1 Measure the contact set. Candidates per vertex 1.2 (8 mm layer) to
      13.8 (3 mm layer); 290 of 300 sampled vertices carry a valid cross-layer
      contact in the stack, none in a flat single layer. Recorded.
- [x] 2.2 Measure the per-iteration residual and step norm. The per-substep
      motion stays at a constant level over the whole tail window (first
      quarter 0.83 mm, last quarter 0.82 mm) instead of decaying: a limit
      cycle, not an unconverged fixed point.
- [x] 2.3 Measure how much of the limit cycle friction owns. Disabling
      friction (`friction_on=0`) leaves the motion unchanged (0.80 mm), and
      sweeping `friction_epsilon` over 0.01/0.1/1/10 changes it by under 3%.
      Friction does not own any of it, so task 3.1 is not required for this
      issue.
- [x] 2.4 Compare the barrier contact option. At equal nominal stiffness the
      barrier is much deeper and much less stable (median penetration 58.5 um
      against 8.7 um, tail motion 5.43 mm against 0.80 mm); matching the
      penalty's resting penetration needs ~100x smaller `*_force_k`, and at
      large k it drives the two layers to zero gap. The two force types are
      not parameter-compatible and the barrier is not a drop-in.
- [x] 2.5 Record the spike results and confirm the implementation target. The
      jitter turned out to be a defect in the existing same-layer edge-edge
      contact (see section 8.1) rather than a limitation of the contact model
      or of the step size, so the ranked candidates were not needed. Sections
      4-6 are closed without implementation, with the reasons recorded below.

## 3. Cheap model corrections

- [x] 3.1 Time-step independent positional friction: not required. Measured
      irrelevant (2.3); closed with the measurement rather than implemented.
- [ ] 3.2 Two-sided shell definition with an explicit buffer: still open. The
      measured relationship between the per-substep motion and the contact
      band (8.3) is the part of it that matters for penetration.
- [x] 3.3 Tie the maximum per-substep motion to the contact distance: closed
      as ineffective. The velocity clamp is applied to the end-of-substep
      velocity, so it does not bound the motion inside a substep: at
      `max_vel` = 0.7 m/s (the band/`step_h` value) the folded sheet still
      passes through itself (421/922/325 crossings per quarter).

## 4. Candidate 1: local contact solve with duals

- [x] 4.1 Not implemented: superseded by 8.1. The measured limit cycle was a
      sign defect in the existing contact, and fixing it removes the ringing
      (0.78 mm -> 0.0003 mm per substep) without changing the solver.

## 5. Candidate 2: barrier contact with a consistent Hessian

- [x] 5.1 Not implemented: superseded by 8.1, and the barrier's own
      measurements (2.4) show it is not a drop-in for cloth-cloth stacking.

## 6. Candidate 3: impact-zone contact solve

- [x] 6.1 Not implemented: superseded by 8.1.

## 7. Verification and shipping

- [x] 7.1 Run the full protocol after the fix. Both jitter scenes and both
      constitutive models settle: two layers 8 mm 0.80 -> 0.0000 mm, two
      layers 3 mm 0.71 -> 0.0000 mm, rotated variants 0.36/0.42 -> 0.00 mm,
      overlapping variants 0.78/0.70 -> 0.02/0.00 mm, single layer unchanged;
      resting penetration 8.6-10 um -> 1.0-6.0 um and the minimum gap
      105 um -> 176-199 um. Garment-scene solver speed is unchanged within
      noise (same-session A/B 14.96 ms -> 14.57 ms per 1 ms call, median).
- [x] 7.2 Regression groups. `-m quick` 12 passed; `-m "sim or algo or api or
      data or drape"` 19 passed / 4 skipped / 2 xfailed; the new
      `tests/sim/test_self_contact.py` 4 passed / 1 xfailed (the xfail is the
      documented tunnelling case at the shipped substep).
- [x] 7.3 Shipped as the default. The change is a defect fix rather than a
      parameter-gated mechanism, so the "selectable at runtime" requirement in
      `specs/contact-stability/spec.md` does not apply as written; that spec
      requirement still needs the maintainer's decision (see 8.8).
- [x] 7.4 This file records the final numbers. The parameter names added by the
      follow-up work are `edge_parallel_sin` / `ee_iteration_sign` (experiments,
      removed again) and none are shipped as a new default.

## 8. What the investigation actually found

- [x] 8.1 Root cause of the ringing (fixed, shipped in the previous commit).
      The same-layer edge-edge narrow phase re-derived the tetrahedron
      (A,B,C,D) orientation at force-evaluation time and multiplied it by the
      orientation recorded at detection time. For nearly parallel edges that
      quantity is identically zero (two parallel lines are always coplanar;
      verified numerically for several z offsets) and its sign is float noise,
      so the product flipped at random inside a substep: a separated pair
      became `pen = thickness + |ba|`, the deepest possible penetration, with
      an inverted attractive normal. On the first substep of the two-layer
      stack every accepted edge-edge contact did that (166/166). Keeping the
      closest-point direction repulsive and using the unsigned separation
      removes the ringing (40-70x) and the local squeeze-through, with no
      measured cost.
- [x] 8.2 Recovery from a crossing inside one substep: implemented three ways
      and rejected. Recorded side used verbatim (2.5 mm jitter, penetration
      saturated, layers pressed together); tetrahedron evaluated once with the
      detection-time reference (1.5 mm / 10.2 mm jitter, layers crossed);
      iteration-level reference (a no-op: 0.78 -> 0.80 mm, because a crossing
      pair is usually a new candidate with no previous reference to compare
      against, and the parallel case carries no sign at all).
- [x] 8.3 Root cause of the penetration (measured). The narrow phase only
      accepts a pair while `pen = thickness - dist > 0`; once two surfaces have
      crossed, `dist` grows again on the far side and the pair is dropped
      entirely, so no force can push it back. Whether a pair crosses is decided
      by the per-substep motion against the contact band
      (`query_radius + thickness`, about 1.2 mm in the shipped block, against
      6-14 mm measured per substep on 0.3-1.0 m panels). Penetration depth
      tracks the per-substep motion and is independent of contact stiffness,
      damping, momentum-matched impulse terms and the force clamp - all of
      which were implemented and measured inert.
- [x] 8.4 Root cause of the "dense mesh penetrates and no parameter helps"
      report. The broad-phase candidate row holds 16 entries per vertex/edge
      and the traversal stops as soon as it is full, so for spacing <= 2 mm
      (more edges inside the band than slots) the nearest pairs are never
      tested and no force parameter can matter. Density-normalised contact
      stiffness (`(reference spacing / spacing)^2`) was implemented and
      rejected (about 15% better at 2 mm, nothing at 1 mm). Distance-ordered
      top-K candidate selection keeps the same 16 slots but keeps the nearest
      ones: the 2 mm stack goes from minimum gap -234 um (crossed) to
      +110 um, deepest penetration 434 -> 90 um, while the garment scene costs
      about 1.9% RTS (pruned two-pass traversal, unsorted insert, cached worst
      distance). `cuobjdump --dump-resource-usage` confirms no new local-memory
      spill (STACK 256/288 as before).
- [ ] 8.5 Spacing <= 1 mm is still not fixed at 16 slots: it needs a larger
      row (VRAM cost, rejected by the maintainer for now) or a different
      candidate structure.
- [ ] 8.6 Fine-mesh cost of the top-K selection: the 2 mm stack pays about
      +110% per substep. Enabling the ordered path only for rows that actually
      saturate is still to be evaluated.
- [ ] 8.7 The seam-snap transient in the garment scene reaches the `max_vel`
      clamp (100 m/s) in its first frames and dominates fast-impact
      penetration; it needs its own fix (a slower snap ramp or a rate limit
      tied to the contact band).
- [ ] 8.8 The unsigned separation gives up within-substep recovery of an
      already crossed pair; that recovery is left to the untangling pass. This
      limitation is documented at the code site and covered by the strict
      xfail in `tests/sim/test_self_contact.py`.
