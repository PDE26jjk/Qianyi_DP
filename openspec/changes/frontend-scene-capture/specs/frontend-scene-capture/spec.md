## Purpose

Capture the simulation inputs of a live Blender scene into a portable package
so a real garment scene can be reproduced, inspected and regression-tested
outside Blender, including from a background session with no user interface.

## ADDED Requirements

### Requirement: Capture produces a complete simulation input package

The capture SHALL record everything the extension needs to rebuild the scene
through `input_data`, and SHALL be self-contained: a consumer SHALL NOT need the
`.blend` file, the addon, or Blender to rebuild the payload.

#### Scenario: Package contents

- **WHEN** a scene containing cloth panels, sewing entries and a collision body
  is captured
- **THEN** the package contains, for every object, the rest positions and the
  simulated positions, the edge and triangle arrays, the world transform, the
  object type, the collision layer and the cloth material properties, plus every
  sewing entry's stitch pairs and pattern indices and the solver parameter
  block that was active at capture time

#### Scenario: Units and identifiers survive the round trip

- **WHEN** a captured package is loaded by a consumer
- **THEN** the reconstructed payload matches the payload the frontend sent:
  integer arrays identical, float arrays within the documented tolerance, and
  the millimetre and gram-per-square-metre quantities unchanged, not
  pre-converted

### Requirement: Capture works without a user interface

The capture SHALL be usable from a background Blender session driven by a
script, because that is how regression material is produced and re-produced.

#### Scenario: Background capture succeeds

- **WHEN** Blender runs the capture entry point in background mode on a saved
  scene
- **THEN** the capture completes without a GPU context, writes the package, and
  exits with a success status

#### Scenario: Capture does not modify the scene

- **WHEN** the capture runs
- **THEN** the `.blend` file is not written, mesh data, shape keys, vertex
  groups and object transforms are unchanged afterwards, and the operator is
  undo-safe

### Requirement: Capture is reproducible and inspectable

A capture SHALL be reproducible and SHALL carry enough metadata for a consumer
to explain it without opening Blender.

#### Scenario: Repeated capture is stable

- **WHEN** the same scene is captured twice without intervening edits
- **THEN** the two packages describe the same payload (arrays equal within
  tolerance, metadata equal except for an explicitly excluded timestamp field)

#### Scenario: Metadata identifies the scene

- **WHEN** a consumer reads the package
- **THEN** it can report the source scene name, the number of cloth panels, the
  number of stitch pairs, the mesh resolution summary, the per-panel material
  properties and the parameter block, without loading Blender

### Requirement: The harness consumes a captured scene

The test system SHALL be able to drive a captured frontend scene through the
same path it uses for its other scenes, while keeping the property that running
tests does not require Blender.

#### Scenario: Captured scene runs as a harness case

- **WHEN** a harness case is pointed at a captured package
- **THEN** the case runs the standard driver against the reconstructed payload,
  writes the standard per-case artifacts, and applies the same mesh contract
  validation as a procedural scene

#### Scenario: Validation rejects a broken package

- **WHEN** a package is missing a required array, has an inconsistent triangle
  or edge index, or references a pattern index that does not exist
- **THEN** loading fails with a message that names the missing or inconsistent
  field, instead of running a corrupt scene

#### Scenario: No Blender dependency at test time

- **WHEN** the test suite runs a captured scene case in an environment without
  Blender
- **THEN** the case runs to completion, and the capture step is the only part of
  the workflow that requires Blender
