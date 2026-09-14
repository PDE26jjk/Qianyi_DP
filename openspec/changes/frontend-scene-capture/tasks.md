# Tasks: frontend-scene-capture

## 1. Define the package format

- [ ] 1.1 Write the format contract (version, `scene.json` fields, `scene.npz`
      array keys, units, ordering rule for cloth before obstacles, relative-path
      rule) next to the loader, in English.
- [ ] 1.2 Add a small synthetic fixture scene and a comparison test that
      captures it and checks the loaded payload against what the frontend sent
      (integers identical, floats within tolerance, offsets identical).

## 2. Make the capture path work without a UI (frontend)

- [ ] 2.1 Split the pattern rebuild so the data path (sections, geometry,
      addressing, mesh regeneration) does not construct gizmo renderers.
- [ ] 2.2 Make the addressing that `setup_sewings_for_simulation` needs
      buildable explicitly, instead of only as a side effect of drawing.
- [ ] 2.3 Add the non-UI capture entry point and confirm it runs in a
      background Blender session on a saved scene and exits with a success
      status.
- [ ] 2.4 Confirm the interactive path is unchanged (pattern edit, sewing,
      simulation start still behave as before) and covered by the existing
      CPU-only registry tests.

## 3. Capture implementation (frontend)

- [ ] 3.1 Gather the per-object payload exactly as
      `_initialize_object_simulation` does, including rest and simulated
      positions, material properties, collision layers, and fixed/attached
      weights.
- [ ] 3.2 Gather the sewing stitch pairs and pattern indices exactly as
      `setup_sewings_for_simulation` does, and assert the pair count matches the
      mesh vertex contract.
- [ ] 3.3 Record the active solver name and parameter block, and the summary
      block (counts, edge-length statistics, bounding boxes, per-panel
      materials).
- [ ] 3.4 Write the package (JSON sidecar plus NPZ arrays), relative paths only,
      and verify the `.blend` is not written and the scene is not mutated.
- [ ] 3.5 Add the operator that runs the same code, exposed in the simulation
      panel, undo-safe, with a user-visible message naming the output directory.

## 4. Loader implementation (engine repository)

- [ ] 4.1 Add the harness loader that turns a package into the `input_data`
      payload plus a scene description.
- [ ] 4.2 Reuse the harness mesh-contract validation and fail with a named field
      on a missing array, an out-of-range index, or an unknown pattern index.
- [ ] 4.3 Register the `frontend:` scene family in the scene registry and make
      an absent package skip with setup instructions.
- [ ] 4.4 Make a harness case able to run a captured scene end to end (standard
      artifacts, invariants, and the scene's own parameter block recorded in the
      results unless overridden).

## 5. Verification

- [ ] 5.1 Capture a real reported scene, load it in an environment without
      Blender, run the standard case, and confirm the round-trip metrics match a
      direct in-Blender run of the same scene within tolerance.
- [ ] 5.2 Confirm the standard artifact set is written and the invariants run.
- [ ] 5.3 Confirm two captures of an unedited scene agree (excluding the
      documented timestamp field).
- [ ] 5.4 Confirm no absolute path appears in either file of the package.

## 6. Record the outcome

- [ ] 6.1 Record the format version, the fixture location, and the capture and
      load commands in this file.
