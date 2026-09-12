## 1. Pin the baseline

- [ ] 1.1 Reproduce the two jitter scenes headlessly with the shipped parameter
      block and record, for the tail window: median and maximum per-substep
      displacement, the moving-vertex count, and the finite check. Verify the
      numbers match the proposal (0.78 mm two layers, 0.61 mm single 3 mm
      layer) and keep the artifact in the gitignored scratch area
- [ ] 1.2 Record the baseline real-time speed of the garment scene with the
      interleaved protocol (at least three round-robin rounds, report min and
      median) and the invariant results of the drape acceptance set. Verify by
      re-running both once and confirming the spread is inside the round-to-
      round noise recorded in the round-1 change
- [ ] 1.3 Record the penetration measure for both scenes (median and deepest
      valid contact penetration, inter-layer overlap) as the number the
      bounded-penetration requirement is checked against

## 2. Spikes that decide the mechanism

- [ ] 2.1 Measure the contact set and its overlap graph (valid contacts per
      substep, per-vertex contact degree, largest connected contact zone) on
      both scenes, to size candidates 1-3
- [ ] 2.2 Measure the per-iteration residual and step norm on both scenes:
      whether the iteration approaches a fixed point and how fast, versus
      staying at a constant level. Verify by plotting/inspecting the recorded
      series and stating which of the two it is
- [ ] 2.3 Measure how much of the limit cycle friction owns: both scenes with
      friction disabled and with the friction Hessian removed from the diagonal,
      reported against the baseline metrics of task 1.1
- [ ] 2.4 Compare the existing barrier contact option (its Hessian is a diagonal
      approximation) against a derivative-consistent variant on one scene, and
      record whether the consistent variant removes the ring on its own
- [ ] 2.5 Record the spike results in this change and confirm the first
      implementation target with the maintainer; if the spikes are
      inconclusive, state which measurement is missing before any code is
      written

## 3. Cheap model corrections (prerequisites for any mechanism)

- [ ] 3.1 Replace the explicit tangential friction force with a time-step
      independent positional friction limited by the Coulomb cone. Verify with
      an A/B on both scenes (jitter metric) plus a lateral-drift measurement
      showing the pile still resists sliding
- [ ] 3.2 Give the contact thickness a two-sided shell definition with an
      explicit buffer. Verify that the resting penetration decreases and the
      settled layer height stays within tolerance of the pre-change value
- [ ] 3.3 Tie the maximum per-substep motion to the contact distance instead of
      a global maximum velocity. Verify no tunnelling on the drop scenes with
      the existing invariant tier

## 4. Candidate 1: local contact solve with duals

- [ ] 4.1 Prototype the local contact solve with dual variables behind a
      parameter, leaving the global solve unchanged. Verify that both scenes
      meet the settling requirement (median per-substep motion at most 10% of
      the baseline) at the unchanged step size
- [ ] 4.2 Measure the local iteration count and the frame cost of the prototype
      against the 10% budget on the garment scene. Verify with the interleaved
      protocol, and record the result in the design's decision table
- [ ] 4.3 If 4.1 and 4.2 pass, make the mechanism the shipped default behind
      the parameter and re-run the full verification group; if not, record why
      and move to candidate 2

## 5. Candidate 2: barrier contact with a consistent Hessian (fallback)

- [ ] 5.1 Make the barrier contact's operator contribution consistent with its
      own force (its curvature, not the penalty stiffness) and verify on one
      scene whether the ring disappears before any cost work
- [ ] 5.2 If it does, make the resulting contact rows affordable (a
      gather-based matvec or packed rows) and measure the frame cost against
      the 10% budget; record both numbers in the design table
- [ ] 5.3 If the cost cannot be met, record the measurement and stop; do not
      adopt a mechanism that fails the cost requirement

## 6. Candidate 3: impact-zone contact solve (only if 1 and 2 fail)

- [ ] 6.1 Prototype a grouped (impact-zone) penetration solve on the two scenes
      and measure the jitter metric and the frame cost against the baseline

## 7. Verification and shipping

- [ ] 7.1 Run the full protocol on the chosen mechanism: both jitter scenes and
      both constitutive models, the garment scene for real-time speed, and the
      drape acceptance set for the invariants. Verify every scenario in
      `specs/contact-stability/spec.md`
- [ ] 7.2 Run the regression groups (`-m quick`, `-m sim`, `-m 'data or drape'`)
      with the mechanism enabled and record the results in this change. Adding
      new tests requires the maintainer's explicit approval first
- [ ] 7.3 Ship the new behaviour as the default only if every requirement
      passes; otherwise leave the parameter off by default and record the
      failing requirement and the measurement behind it
- [ ] 7.4 Update the change's documents with the final parameter name, the
      shipped default, and the measured comparison that justifies it
