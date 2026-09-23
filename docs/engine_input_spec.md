# Engine input specification

The authoritative description of what `Qianyi_DP` accepts: the `input_data`
payload, the runtime parameter surface, and the driver order. It mirrors the
parsing code in `src/simulator_interface.cpp` and the consumers in
`src/simulation/`; when the two disagree, the code wins and this document is
wrong.

All geometry is in metres, all indices are per-object (local), and the engine
adds the per-object offsets itself.

## 1. Driver order

```python
sim = qydp.simulator
sim.set_solver("PDNewton")          # 1. pick the solver
sim.set_parameters({...})           # 2. set the runtime parameters
sim.input_data(payload)             # 3. build the scene
for frame in range(frames):
    sim.update(dt)                  # 4. advance one frame
    vertices = sim.get_simulation_data(world_space=True)
```

- Steps 1 and 2 MUST precede `input_data`: the solver object is created from the
  current solver name inside `Simulator::init`, which `input_data` calls.
- `update(dt)` advances one frame. The engine subdivides the frame internally
  into substeps of `step_h` (clamped by the solver's own `max_stable_step_h`),
  so callers pass the frame length once and never loop substeps themselves.
- Accepted solver names: `PDNewton`, `VBD`, `XPBD`, `Explicit`. An unknown name
  throws `Unknown solver type: <name>`.
- **Only `PDNewton` is production-ready.** `VBD`, `XPBD` and `Explicit` are
  experimental: they are not part of the validated configuration, their
  automated tests are recorded as known failures (strict xfail, see the
  `testing-harness` spec's "Experimental solver governance"), and they are for
  research and comparison only. A shipped scene uses PDNewton.
- Only PDNewton applies the external forces (pressure and wind); every other
  solver ignores them, so the same scene run on another solver simply has no
  external-force term.
- `on_exit()` releases the solver.

## 2. Payload

```python
{
    "mesh_list": [mesh_entry, ...],   # at least one
    "sewings":   [sewing_entry, ...], # may be empty
}
```

### Worked example: one hanging cloth panel

```python
import numpy as np

cols, rows, size = 10, 10, 1.0
xs = np.linspace(0.0, size, cols)
ys = np.linspace(0.0, size, rows)
xx, yy = np.meshgrid(xs, ys)
vertices = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1).astype(np.float32)

tris = []
for j in range(rows - 1):
    for i in range(cols - 1):
        a = j * cols + i
        tris += [(a, a + 1, a + cols + 1), (a, a + cols + 1, a + cols)]
tris = np.asarray(tris, dtype=np.int32)
edges = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
                          axis=1), axis=0).astype(np.int32)

pinned = np.zeros(cols * rows, dtype=np.float32)
pinned[vertices[:, 1] > size - 1e-6] = 1.0   # pin the top edge

mesh = {
    "vertices": vertices.reshape(-1),
    "vertices_sim": vertices.reshape(-1).copy(),
    "edges": edges.reshape(-1),
    "triangles": tris.reshape(-1),
    "object_type": 0,
    "collision_layer": 0,
    "world_matrix": np.eye(4, dtype=np.float32),
    "fixed_vertices": pinned,
    "attached_vertices": np.zeros(cols * rows, dtype=np.float32),
    "mass": 100.0,          # g/m^2
    "granularity": 20.0,    # mm
    "thickness": 0.1,       # mm
    "friction": 0.03,
    "stretch": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "bending": np.array([1.0, 1.0, 1.0], dtype=np.float32),
}
```

### Worked example: two panels sewn into a closed shell

Both panels use the same flat pattern; the back panel is placed by its world
matrix (180 degrees about X, translated by the gap) so its outward direction is
the opposite of the front panel's. The stitches join the two boundary rings by
local index, and both panels carry the same `pressure`:

```python
back = dict(front)                       # same flat pattern in vertices
back["vertices_sim"] = mirrored_vertices # local pattern mirrored in Y so the rings align
world = np.eye(4, dtype=np.float32)
world[1, 1] = -1.0
world[2, 2] = -1.0
world[2, 3] = -0.002                     # translation lives in the last column
back["world_matrix"] = world
front["pressure"] = back["pressure"] = 300.0   # Pa

ring = sorted(boundary_vertex_indices(cols, rows))
payload = {
    "mesh_list": [front, back],
    "sewings": [{
        "patterns": [0, 1],
        "stitches": np.asarray([[i, i] for i in ring], dtype=np.int32).reshape(-1),
        "angle": 0.0,
    }],
}
```

### Worked example: a static body next to cloth

Append the body after every cloth object (cloth must be the prefix), give it
`object_type = 1` and its per-triangle-vertex `normals`, and mark the cloth
vertices that should follow it with `attached_vertices`.

### Extra keys are ignored

The engine reads the keys listed above and ignores everything else in a mesh
entry. The Blender addon therefore passes its own bookkeeping along in the same
dicts (`obj`, `vertices_offset`, `faces_offset`, `edges_offset`), and a caller
that needs those must keep them out of the engine's way only in the sense that
they must not collide with a documented key.

## 3. Mesh entry - required for every object

| Key | Type / shape | Meaning |
| --- | --- | --- |
| `vertices` | float32, `(N, 3)` flattened | Pattern-space vertex positions (the flat pattern, `z = 0`). Rest lengths and the material frame are measured here, not on `vertices_sim`. |
| `edges` | int32, `(E, 2)` flattened | Unique undirected edges as local vertex index pairs. |
| `triangles` | int32, `(T, 3)` flattened | Triangles as local vertex indices. |
| `object_type` | int32 scalar | `0` = simulated cloth; non-zero = static obstacle/body (its vertices get zero inverse mass). |
| `collision_layer` | int32 scalar | Selects how a contact normal is oriented. Compared as a difference between the two sides: equal layers use the side sign recorded by the broad phase, a smaller vertex layer uses the smoothed vertex normal, and a larger one leaves the triangle normal unchanged. |
| `world_matrix` | float32, 16 values | Local-to-world transform, column-vector convention (translation in the last column, i.e. `w = M @ [x, y, z, 1]`). Its inverse maps world back to local. |

## 4. Mesh entry - required for cloth (`object_type == 0`)

| Key | Type / shape | Unit | Meaning |
| --- | --- | --- | --- |
| `fixed_vertices` | float32, `(N,)` | - | Per-vertex weight. A value `> 0.01` pins the vertex: inverse mass is forced to zero. |
| `attached_vertices` | float32, `(N,)` | - | Per-vertex weight. A value `> 0.01` attaches the vertex to the nearest obstacle face (it follows that face). Takes precedence over `fixed_vertices`. |
| `mass` | float32 scalar | g/m^2 | Areal density (the fabric weight). Converted to kg/m^2 internally. |
| `granularity` | float32 scalar | mm | Target pattern granularity. Converted to metres internally. |
| `thickness` | float32 scalar | mm | Shell thickness. Converted to metres internally; also the height at which the ground plane carries the cloth. |
| `friction` | float32 scalar | - | Contact friction coefficient. |
| `stretch` | float32, `(3,)` | - | In-plane stiffness `(u, v, shear)`, scaled by the solver's `base_spring_stiffness`. |
| `bending` | float32, `(3,)` | - | Bending stiffness, scaled by the solver's `bending_k`. |

## 5. Mesh entry - required for non-cloth objects (`object_type != 0`)

| Key | Type / shape | Meaning |
| --- | --- | --- |
| `normals` | float32, `(T, 3)` flattened | Per-triangle-vertex normals. Used to orient the triangles at load; cloth objects do not take this key. |

## 6. Mesh entry - optional

| Key | Applies to | Default | Meaning |
| --- | --- | --- | --- |
| `vertices_sim` | any | `vertices` | Initial simulated placement in local space. The solver integrates these and `world_matrix` maps them to world space. Use it to place a flat pattern in 3D. |
| `grain_dir` | cloth | `0` | Pattern grain direction. |
| `pressure` | cloth | `0` | Constant pressure load along the surface normal, in Pa, applied by PDNewton. Positive inflates, negative pulls inward, `0` is inert. Requires no watertight shell and no volume state. |
| `wind_drag` | cloth | `-1` | Per-object drag coefficient `C_D` override. A negative value means "use the global `wind_drag_coefficient`". |
| `wind_lift` | cloth | `-1` | Per-object lift coefficient `C_L` override. A negative value means "use the global `wind_lift_coefficient`". |

## 7. Sewing entry

| Key | Type / shape | Meaning |
| --- | --- | --- |
| `patterns` | list[int], length 2 | The two object indices joined by this seam. |
| `stitches` | int32, `(S, 2)` flattened | Stitch pairs as panel-local vertex indices; entry `k` pairs `stitches[2k]` of the first pattern with `stitches[2k + 1]` of the second. |
| `angle` | optional float32 | Rest dihedral angle of the seam. Default `0`. |
| `compress` | optional float32 | Compression length of the seam. Default `1`. |

A seam is a permanent zero-rest-length constraint. Panels that are sewn
together stay separate vertex sets, so a seam is not a topological weld.

## 8. Index spaces and ordering rules

- Every index inside a mesh entry is local to that object. The engine offsets
  them when it concatenates objects into the global buffers.
- Every edge used by a triangle must exist in that object's `edges` array,
  otherwise the vertex-to-edge lookup fails, the triangle is dropped and the
  engine prints `[Qianyi Error] There are edges that do not exist in the
  triangle!`.
- Cloth objects MUST come first. The cloth vertex/edge/triangle counts are taken
  as the prefix up to the first object with a non-zero `object_type`, and only a
  single cloth-to-non-cloth transition is supported. A scene that puts an
  obstacle before a cloth object makes the engine treat the obstacle as cloth.
- `vertices` is the pattern space: the 2x2 material frame is built from its
  `(x, y)` projection, so a pattern that lies in a plane perpendicular to `Z`
  produces a singular material Jacobian. Author the flat pattern in `vertices`
  and the 3D placement in `vertices_sim` / `world_matrix`.
- Triangle orientation is normalized at load: cloth triangles are re-ordered in
  the pattern plane so their pattern normal has a `+Z` component, and the cloth
  normal array is `(0, 0, 1)`. The winding in the input data is therefore not
  preserved, and the direction a panel inflates towards or receives wind on is
  decided by its 3D placement (`vertices_sim` plus `world_matrix`). Two panels
  sewn back to back need opposite outward placements, not opposite windings.
- Obstacle objects are static: their vertices get zero inverse mass, so they
  must be supplied with `vertices_sim` already in their final place.

## 9. Units and conversions

The engine works in metres: a caller whose source data is in centimetres (for
example a dataset whose box mesh is in centimetres) converts before calling
`input_data`. The loader for the GarmentCodeData tests does exactly that, and
also rotates its Y-up meshes into the engine's Z-up convention.

| Quantity | Caller unit | Engine unit | Conversion |
| --- | --- | --- | --- |
| Vertex positions, world matrix translation | m | m | - |
| `mass` | g/m^2 | kg/m^2 | `x 0.001` |
| `granularity` | mm | m | `x 0.001` |
| `thickness` | mm | m | `x 0.001` |
| `pressure` | Pa | Pa | - |
| `friction`, `wind_drag`, `wind_lift`, `stretch`, `bending`, `grain_dir` | - | - | dimensionless |
| `gravity`, `wind_*` velocities, `query_radius` | - | m/s, m | - |

## 10. Runtime parameters

Set through `set_parameter(key, value)` or `set_parameters(dict)`. Keys that are
never set take the default listed here; the listed default is the one compiled
into the consumer.

This list is the engine's whole parameter surface as read from the source (78
keys). It is deliberately wider than any user interface: the Blender solver
panel exposes a subset and the two are not aligned yet (see "The panel does not
define the engine's surface" below).

### Scene and environment

| Key | Default | Meaning |
| --- | --- | --- |
| `gravity` | `-9.8` | Gravity along `Z`, m/s^2. It is applied only once the sewing assembly is considered closed. |
| `ground` | `1` | `1` enables the ground plane at the cloth thickness. |
| `ground_f` | `1e3` | Ground contact stiffness. |
| `max_vel` | `1000` | Velocity clamp, m/s. |
| `velocity_damping` | `0.5` | Velocity decay rate at the end of each substep, `v *= exp(-h * rate)`. This is the rate used for vertices moving faster than `creep_speed`. |
| `smooth_times` | `5` | Laplacian smoothing passes per frame. |
| `step_h` | `0.001` | Maximum substep length; `update(dt)` subdivides by it. |
| `average_mass_by_cloth` | `0` | `1` averages the areal density across the cloth vertices of an object. |
| `debug_v_id`, `debug_e_id` | `-1` | Debug probes: vertex / edge index to dump. |

### External forces (PDNewton only)

| Key | Default | Meaning |
| --- | --- | --- |
| `wind_x`, `wind_y`, `wind_z` | `0` | Base wind velocity, m/s. Wind is inert while all three are zero and `wind_turbulence` is zero. |
| `wind_gust` | `0` | Gust amplitude as a fraction of the base speed. |
| `wind_gust_frequency` | `0.5` | Gust noise frequency, Hz. |
| `wind_turbulence` | `0` | Turbulence amplitude, m/s. |
| `wind_turbulence_scale` | `0.5` | Turbulence spatial frequency, 1/m. |
| `wind_turbulence_speed` | `1` | Advection speed of the turbulence pattern, m/s. |
| `air_density` | `1.225` | Air density, kg/m^3. |
| `wind_drag_coefficient` | `1.0` | Global lift/drag coefficient `C_D`. |
| `wind_lift_coefficient` | `0.0` | Global lift/drag coefficient `C_L`. |

### PDNewton

| Key | Default | Meaning |
| --- | --- | --- |
| `pd_iters` | `10` | Outer (Newton) iterations per substep. |
| `pd_hessian_every` | `1` | Rebuild the element Hessian every N substeps; larger values reuse a lagged Hessian. |
| `pd_static_diag_scale` | `1` | Scale on the fixed projective diagonal. |
| `pd_static_diag_offdiag` | `1` | `1` gives the fixed diagonal a graph-Laplacian off-diagonal part (rigid mode free, free fall correct). |
| `warm_start` | `2` | Substep start predictor: `0` off, `1` VBD-style projection, `2` full inertia prediction, `3` velocity only. |
| `pd_cuda_graph` | `1` | `1` replays the iteration loop from a captured CUDA graph. |
| `linear_solver_type` | `0` | `0` = PCG, `1` = Jacobi (shown as "Block Jacobi" in the UI panel). |
| `linear_iters` | `10` | Linear solver iterations. |
| `pc_iters` | `2` | Post-contact correction iterations. |
| `creep_damping` | `0` | Damping rate applied *instead of* `velocity_damping` to vertices slower than `creep_speed`. It removes the slow crawl of a garment resting on a body without killing the momentum of a real motion; `0` turns the creep band off. |
| `creep_speed` | `0.03` | Speed threshold (m/s) that selects the creep band; `0` disables it. |
| `mask_stiff` | `1e2` | Stiffness added for masked (pinned/attached) vertices. |
| `base_spring_stiffness` | `4.0e3` | Base in-plane stiffness; `stretch` is relative to it. |
| `strain_stiffen_start` | `5.0e-2` | Strain above which the spring stiffness starts to grow. |
| `strain_stiffen_rate` | `2.0` | Growth rate of that stiffening. |
| `bending_k` | `0.2` | Base bending stiffness; `bending` is relative to it. |
| `bending_model` | `2` | `0` = IBM quadratic, `1` = discrete shells (Gauss-Newton), `2` = discrete shells (AOGS). |
| `constitutive_model_planar` | `0` | `0` = spring-mass, `1` = FEM (BW). |
| `fem_psd_clamp`, `fem_shear_hessian` | `1` | FEM Hessian regularization switches. |
| `gamma_r`, `gamma_min` | `0.9`, `1e-9` | Line-search parameters. |
| `parallel_eps` | `1e-6` | Parallel-edge epsilon. |

### Sewing

| Key | Default | Meaning |
| --- | --- | --- |
| `sewing_k` | `1e5` | Seam spring stiffness. |
| `sewing_max_force` | `1e5` | Seam force clamp. |
| `sewing_close_dist` | `1e-2` | Stitch pairs closer than this count as closed (which enables gravity). |
| `sewing_forced_connect_frame` | `80` | Frame after which seam clusters are snapped together. |
| `sewing_snap_dist` | `3e-3` | Initial snap gate, widened as the assembly settles. |
| `sewing_snap_max_dist` | `1.0` | Ceiling of the widened snap gate. |
| `sewing_force_merge_frames` | `0` | Frames of forced merge after the activation gate. |
| `seam_merge_velocity` | `1` | `1` refreshes cluster velocities after a hard seam projection. |

### Contact

| Key | Default | Meaning |
| --- | --- | --- |
| `query_radius` | `1e-3` (PDNewton and Explicit), `1e-2` (VBD) | Contact / broad-phase query radius, m. |
| `update_collision_substeps` | `20` | Substeps between collision refreshes. |
| `update_pick_substeps` | `10` | Substeps between pick-constraint refreshes. |
| `tight_broad_phase` | `1` | `1` keeps the broad phase tight (needs small substeps), `0` keeps the conservative CCD behavior. |
| `bvh_query_order`, `bvh_streams` | `1` | BVH traversal order and stream count. |
| `collision_collect_tp`, `collision_collect_ee` | `1` | Collect point-triangle / edge-edge candidate pairs. |
| `vf_force_k`, `ee_force_k`, `ef_force_k` | `0.2` (`vf_force_k` is `0.5` in the frontend block) | Contact penalty stiffnesses. |
| `vf_force_type`, `ee_force_type` | `1` | Contact force formulations. |
| `vf_ground_k` | `0.2` | Ground penetration stiffness used by the vertex-face path. |
| `friction_on` | `1` | Enables contact friction. |
| `friction_epsilon` | `1e-2` | Friction regularization. |
| `IPC_k`, `LCP_substeps`, `avbd_contact_beta` | `1500`, `20`, `10000` | Contact parameters of the IPC/LCP and AVBD paths. |

### Other solvers (VBD / XPBD / Explicit)

**Research reference only.** These solvers are experimental and not
production-ready: they are not validated, they are not part of the standard
configuration, and their tests are known failures. The parameters below are
documented so that a comparison run can be reproduced, not as a supported
configuration.

| Key | Default | Meaning |
| --- | --- | --- |
| `vbd_iters` | `10` | VBD iterations per substep. |
| `xpbd_iters`, `xpbd_dynamics_iters` | `10`, `1` | XPBD iterations. |
| `xpbd_relaxation` | `0.9` | XPBD relaxation factor. |
| `xpbd_use_lambdas` | `1` | `1` uses the XPBD Lagrange-multiplier form. |
| `xpbd_max_step_h` | `1e-3` | Largest substep XPBD accepts; the frame is subdivided to match. |
| `explicit_max_step_h` | `2.5e-4` | Largest substep the explicit solver accepts. |

### Three different sets of defaults

The tables above are the values **compiled into the engine**, i.e. what a key
falls back to when nothing sets it. Two other sets exist and they are not the
same numbers, so always say which one is meant:

1. **Engine fallback** - the compiled defaults in this document.
2. **Solver panel defaults** - what the Blender addon applies when a run starts
   from the UI (`Qianyi/model/solver_params.py` `PARAMETERS`;
   `apply_to_engine()` calls `set_solver` and `set_parameters` before
   `input_data`).
3. **Harness block** - the frozen PDNewton block in the repository's own test
   harness (`tests/harness/presets.py`), used by the automated tests.

Where the three disagree:

| Key | Engine fallback | Solver panel | Harness block |
| --- | --- | --- | --- |
| `step_h` | `0.001` | `0.0045` | `0.003` |
| `pd_iters` | `10` | `5` | `5` |
| `linear_iters` | `10` | `2` | `2` |
| `pc_iters` | `2` | not exposed | `0` |
| `mask_stiff` | `1e2` | `2e3` | `2000` |
| `max_vel` | `1000` | `10` | `100` |
| `bending_model` | `2` (AOGS) | `2` (AOGS) | `0` (IBM quadratic) |
| `bending_k` | `0.2` | `1e-3` | `1.0` |
| `constitutive_model_planar` | `0` (spring-mass) | `1` (FEM_BW) | `0` (spring-mass) |
| `strain_stiffen_start` | `5e-2` | `0.0` | - |
| `query_radius` | `1e-3` | `1e-3` | `1e-3` |
| `sewing_k` | `1e5` | `4e4` | `1e5` |
| `sewing_forced_connect_frame` | `80` | `10` | `80` |
| `vf_force_k` | `0.2` | `0.1` | `0.5` |
| `ee_force_k` | `0.2` | `0.1` | `0.2` |
| `ef_force_k` | `0.2` | `50.5` | `0.5` |
| `vf_ground_k` | `0.2` | `4.0` | `20` |
| `vf_force_type`, `ee_force_type` | `1` (IPC) | `1` (IPC) | `0` (Spring) |

The solver panel also carries a *custom parameters* list (arbitrary engine key /
float pairs) which is forwarded to `set_parameters` verbatim; that is the escape
hatch for engine keys the panel does not model explicitly.

### The panel does not define the engine's surface

The authoritative key set is what the engine reads: **78 keys**, all of them
listed in this document. The Blender solver panel currently models **34** of
them, and every key it models is read by the engine (there are no dead panel
knobs), but the two lists are **not aligned yet** - a key that has no panel
control is still a valid engine input, settable through the panel's custom
parameters list or programmatically. Do not read "not in the panel" as "not in
the engine".

The 44 engine keys with no panel control today, by area:

| Area | Keys |
| --- | --- |
| External forces (11) | `wind_x`, `wind_y`, `wind_z`, `wind_gust`, `wind_gust_frequency`, `wind_turbulence`, `wind_turbulence_scale`, `wind_turbulence_speed`, `air_density`, `wind_drag_coefficient`, `wind_lift_coefficient` |
| PDNewton and linear solve (5) | `warm_start`, `pd_cuda_graph`, `pd_static_diag_scale`, `pd_static_diag_offdiag`, `pc_iters` |
| Contact and BVH (13) | `ground_f`, `friction_on`, `friction_epsilon`, `collision_collect_tp`, `collision_collect_ee`, `tight_broad_phase`, `bvh_query_order`, `bvh_streams`, `update_collision_substeps`, `update_pick_substeps`, `IPC_k`, `LCP_substeps`, `avbd_contact_beta` |
| Sewing (3) | `sewing_close_dist`, `sewing_max_force`, `sewing_force_merge_frames` |
| Other solvers (7) | `vbd_iters`, `xpbd_iters`, `xpbd_dynamics_iters`, `xpbd_relaxation`, `xpbd_use_lambdas`, `xpbd_max_step_h`, `explicit_max_step_h` |
| Numerical and membrane Hessian (5) | `gamma_r`, `gamma_min`, `parallel_eps`, `fem_psd_clamp`, `fem_shear_hessian` |

The same applies to the per-mesh keys: the Blender payload builder does not emit
`pressure`, `wind_drag` or `wind_lift` yet, so a scene that wants per-object
pressure needs a frontend change (or an out-of-band payload).

## 11. Runtime API beyond the payload

| Call | Purpose |
| --- | --- |
| `update(dt)` | Advance one frame. |
| `get_simulation_data(world_space=False)` | Vertex positions, `(N, 3)`, for the cloth vertices. |
| `get_debug_colors()` | Per-vertex debug colours, `(N, 3)`. |
| `get_residual_metrics()` | Dict with the last substep's Newton and linear residual triples (`newton_initial/final/relative`, `linear_*`, `linear_plain_*`). |
| `set_solver(name)`, `get_all_solver()` | Select / list the solver. |
| `update_world_matrix(obj, matrix16)`, `update_local_vertices(obj, vertices)` | Move an object or replace its rest vertices at runtime. |
| `add_picker` / `update_picker` / `remove_picker`, `pick_triangle` / `pick_triangle_update` / `pick_triangle_remove` | Interactive drag constraints. |
| `check_point_attributes(i)`, `check_edge_attributes(p0, p1)`, `check_edge_collision_data(p0, p1)` | Per-vertex / per-edge debug state. |
| `on_exit()` | Release the solver and the CUDA resources. |

## 12. Diagnostics emitted for invalid input

| Message | Trigger |
| --- | --- |
| `[Qianyi Error] There are edges that do not exist in the triangle!` | A triangle uses an edge that is not in `edges`. |
| `stitches closed: k/n` | Progress of the seam assembly. |
| `PCG ended with NaN residual.` | The linear solve diverged; `update` throws. |
| `Unknown solver type: <name>` | `set_solver` with an unknown name. |

## 13. Frontend mapping (Blender addon)

Where each engine input comes from in the Blender addon
(`Qianyi/simulation/simulation_manager.py` and
`Qianyi/model/`), so a payload can be traced back to the scene:

This describes the addon as it stands today. The addon and the engine are not
fully aligned yet, so treat the engine-facing sections above as the contract and
this section as the current implementation of it.

### Which objects are sent

Every mesh object whose simulation properties say it is a pattern mesh or that
it participates in the simulation. Before sending, the list is sorted by
`object_type`, which is what guarantees the engine's "cloth first" rule (the
sort is stable, so objects of the same type keep their scene order). The
position in that sorted list is the object index used by
`update_world_matrix`, `update_local_vertices`, picking, and the `patterns`
pairs of a sewing entry.

### Cloth entries

| Engine key | Blender source |
| --- | --- |
| `vertices` | The `QYBasis` shape key: the panel's rest geometry in the object's local space, i.e. the pattern the mesh was built from. |
| `vertices_sim` | The `QYSim` shape key (the simulated placement; the engine's output is written back into it). |
| `edges`, `triangles` | The evaluated pattern mesh (`mesh.edges`, `mesh.loop_triangles`). |
| `world_matrix` | `obj.matrix_world`. |
| `mass`, `thickness`, `friction`, `stretch`, `bending` | The fabric of the object's pattern. |
| `granularity`, `grain_dir`, `collision_layer` | The pattern. |
| `fixed_vertices`, `attached_vertices` | Vertex-group weights of the fix / attach pin groups (continuous weights, not only 0/1). |
| `object_type` | `0` for a pattern mesh. |

### Non-cloth entries

`object_type` is `1`, `mass` is `1`, `normals` come from the evaluated mesh's
`loop_triangles`, and `collision_layer` comes from the object's simulation
properties. The frontend takes `edges` straight from `mesh.edges` when every
face is already a triangle, and otherwise derives them from the triangulated
faces after warning `Not all faces are triangles!`.

### Sewing entries

`Project.setup_sewings_for_simulation()` collects `Sewing.get_stitch_data()`,
which returns exactly `{"patterns": (index1, index2), "stitches": (S, 2) int
array, "angle": 0.0}`. The indices are the sorted payload positions, and the
frontend never sends `compress`. Stitches that run outside their panel are
dropped from both sides so the pairs stay aligned.
