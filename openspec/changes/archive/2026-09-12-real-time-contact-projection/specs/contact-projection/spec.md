## ADDED Requirements

### Requirement: Contact penetration is corrected at the position level

The PDNewton solver SHALL remove the penetration its penalty contacts leave
behind by applying a bounded, mass-weighted correction to the vertex positions
after the substep's iteration loop and before the velocity step. The
correction SHALL use the same broad-phase candidate lists as the contact force
kernels and SHALL be switchable at runtime so the projected and unprojected
behaviour can be compared in one binary.

#### Scenario: Projection removes the residual penetration

- **WHEN** a cloth rests on the ground or on another cloth and the substep
  leaves a penetration inside the contact radius
- **THEN** the vertices involved are moved along the contact normal by at most
  the penetration depth of that contact, and the penetration measure does not
  grow from substep to substep

#### Scenario: Switchable

- **WHEN** `contact_projection` is 0
- **THEN** no projection kernel is launched and the frame behaves exactly as
  before the change

### Requirement: The projection cannot blow up under deep or unsolvable penetration

The projection SHALL be a contraction towards the non-penetration manifold:
its per-contact multiplier SHALL be non-negative and under-relaxed, its
per-contact and accumulated per-vertex displacement SHALL be capped per
substep, and vertices that are not free cloth vertices (obstacles, pinned or
attached vertices) SHALL receive zero correction. An infeasible contact set
SHALL leave residual penetration instead of producing a large or growing
displacement.

#### Scenario: Deep penetration

- **WHEN** a cloth is initialised deep inside an obstacle, or inside a second
  cloth layer, and stepped with the frontend parameters
- **THEN** all vertices stay finite, the per-substep displacement stays within
  the configured cap, and the penetration measure decreases or stays bounded

#### Scenario: Velocity stays bounded

- **WHEN** the projection corrects a deep penetration in one substep
- **THEN** the substep velocity is recomputed from the corrected positions and
  still passes through the existing maximum-velocity clamp

### Requirement: The projection does not change the quality invariants

Turning the projection on SHALL NOT degrade seam closure, area preservation,
attached-vertex drift or the drape's penetration depth beyond the tolerances
the drape acceptance set already uses, and it SHALL NOT change the simulation
when no contact is active.

#### Scenario: No contact active

- **WHEN** a scene with no valid contact is stepped with the projection enabled
- **THEN** the positions are identical to the unprojected run

#### Scenario: Quality invariants

- **WHEN** the drape acceptance set is run with the projection enabled
- **THEN** seam closure, area ratio, attached-vertex drift and penetration
  depth stay within the tolerances of the unprojected run
