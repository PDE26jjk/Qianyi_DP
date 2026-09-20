## Purpose

Defines how a closed pattern outline with optional holes and internal lines is
sampled and triangulated into a simulation-ready panel mesh, including the
meaning of the requested resolution and the guarantees on triangle quality at
constraint edges.

## ADDED Requirements

### Requirement: Requested resolution is the target edge length

The `radius` argument SHALL be the target triangle edge length. The sampler
SHALL place one jittered interior point per cell whose side is that length, and
the relaxed median interior edge of a panel whose boundary is discretised at or
below the requested length SHALL be within 15% of it.

#### Scenario: Dense legal boundary

- **WHEN** a 1 x 1 panel is sampled with `radius = 0.01` and its boundary is
  discretised at 0.01
- **THEN** the median interior edge is between 0.0085 and 0.0115

#### Scenario: Existing callers are unchanged

- **WHEN** a caller passes only the five required positional arguments
- **THEN** the call succeeds and uses the shipped default sampling parameters

### Requirement: Hole handling is orientation independent

The inside/outside test SHALL be driven by `is_holes`: a point is inside the
domain when it is inside the first (outer) loop and outside every loop flagged
as a hole. Loop winding SHALL NOT affect the result.

#### Scenario: Clockwise outer loop

- **WHEN** the outer loop is passed clockwise and the hole counter-clockwise
- **THEN** the returned mesh covers the same domain as the counter-clockwise
  outer / clockwise hole input

#### Scenario: No triangles inside a hole

- **WHEN** a panel carries a hole loop
- **THEN** no returned triangle centroid lies inside the hole

### Requirement: Interior points keep a clearance from every constraint edge

Generated interior points SHALL stay at least `boundary_margin_cells` times the
requested edge length away from every constraint edge: the outer loop, hole
loops and internal lines. The clearance SHALL be enforced when points are
generated, when relaxation moves them, and when they are validated. The default
SHALL be 0.8 and the value SHALL be exposed to callers.

#### Scenario: Straight boundary

- **WHEN** a panel with a straight, fully discretised seam edge is sampled
- **THEN** every generated interior point is at least the configured clearance
  from that edge

#### Scenario: Seam-adjacent triangle quality

- **WHEN** a legal panel (boundary spacing at or below the requested length) is
  sampled with the default clearance
- **THEN** the minimum quality of triangles touching a constraint edge is at
  least 0.5

### Requirement: No zero-area or collinear triangles

The returned mesh SHALL NOT contain zero-area triangles. Triangles whose
normalised cross product falls below the numerical sliver floor SHALL be
dropped before the mesh is returned.

#### Scenario: Two-millimetre straight edge

- **WHEN** a panel whose boundary is discretised at 2 mm is sampled with a 2 mm
  requested edge length
- **THEN** no returned triangle has an area below 1e-12 and the covered area is
  within 0.1% of the panel area

### Requirement: Only in-domain triangles are returned

The triangulation SHALL be filtered by an exact centroid-inside test on the
same parity mask that classifies the domain, so concave corners, holes and
orientation changes cannot leak triangles across a boundary.

#### Scenario: Concave outline

- **WHEN** an L-shaped outline is sampled
- **THEN** no returned triangle centroid lies in the excluded quadrant

### Requirement: Selectable triangulation backend with fallback

The sampler SHALL accept a `triangulator` selector. The default SHALL be
gDel2D. When gCDT is selected and cannot produce a valid mesh, the sampler
SHALL fall back to gDel2D and return the fallback mesh.

#### Scenario: gCDT rejected mesh

- **WHEN** gCDT is selected on an input where its validation fails
- **THEN** the sampler returns a gDel2D mesh instead of failing or returning
  the invalid result
