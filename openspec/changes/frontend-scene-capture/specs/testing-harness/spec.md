## ADDED Requirements

### Requirement: Frontend-captured scenes as first-class harness scenes

The system SHALL be able to drive a scene captured from the Blender frontend
through the same driver, artifact and assertion path it uses for procedurally
generated scenes, so a reported frontend scene can become a regression case
without being re-authored by hand.

#### Scenario: Procedural and captured scenes share the driver

- **WHEN** the same harness case is run against a procedural scene and against a
  captured frontend scene
- **THEN** both produce the standard per-case artifacts (per-frame data, traces,
  results summary, log) and both are subject to the same invariant checks

#### Scenario: A captured scene reports its own parameters

- **WHEN** a captured scene carries a solver parameter block
- **THEN** the case uses that block (unless the caller overrides it explicitly)
  and records it in the results, so a result can be attributed to the scene's
  own configuration

#### Scenario: Missing capture is a skip, not a failure

- **WHEN** a case refers to a captured scene that is not present in the
  environment
- **THEN** the case skips with setup instructions instead of failing, matching
  how the dataset-backed cases behave
