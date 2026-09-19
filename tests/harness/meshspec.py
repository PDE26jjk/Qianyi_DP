"""Procedural mesh specification and scene construction for simulation tests.

The scene contract mirrors the Qianyi Blender frontend
(``simulation_manager._initialize_object_simulation``): meshes are passed to
``simulator.input_data`` as flat float32/int32 arrays with the documented keys.

Unit conversion note: ``granularity`` and ``thickness`` are expressed in
millimetres and converted to metres inside the extension (matching the frontend
contract); the harness stores and passes the millimetre values unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_IDENTITY_WORLD_MATRIX = np.eye(4, dtype=np.float32)


def _derive_edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    """Return the unique undirected edges implied by the triangle list."""
    pairs = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).astype(np.int32)


@dataclass
class MeshSpec:
    """Specification of a procedural grid cloth.

    Defaults match the Blender frontend fabric/pattern defaults: mass=100,
    granularity=20mm, thickness=0.1mm, friction=0.03, stretch/bending=(1,1,1),
    collision_layer=0 and an identity world matrix.
    """

    width_m: float = 1.0
    height_m: float = 1.0
    cols: int = 10
    rows: int = 10
    mass: float = 100.0
    granularity_mm: float = 20.0
    thickness_mm: float = 0.1
    friction: float = 0.03
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bending: tuple[float, float, float] = (1.0, 1.0, 1.0)
    collision_layer: int = 0
    grain_dir: float = 0.0
    fixed_vertex_indices: tuple[int, ...] = ()
    attached_vertex_indices: tuple[int, ...] = ()

    @property
    def num_vertices(self) -> int:
        return self.rows * self.cols

    def grid_vertices(self) -> np.ndarray:
        """Return the local-space grid vertex positions, shape (N, 3)."""
        xs = np.linspace(0.0, self.width_m, self.cols)
        ys = np.linspace(0.0, self.height_m, self.rows)
        xx, yy = np.meshgrid(xs, ys)
        points = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
        return points.astype(np.float32)

    def triangles(self) -> np.ndarray:
        """Return the grid triangulation, shape (T, 3), with consistent winding."""
        tris = []
        for j in range(self.rows - 1):
            for i in range(self.cols - 1):
                a = j * self.cols + i
                b = a + 1
                c = a + self.cols
                d = c + 1
                tris.append((a, b, d))
                tris.append((a, d, c))
        return np.asarray(tris, dtype=np.int32)

    def edges(self) -> np.ndarray:
        """Return the unique undirected edges, shape (E, 2)."""
        return _derive_edges_from_triangles(self.triangles())

    def _pin_weights(self, indices: tuple[int, ...]) -> np.ndarray:
        weights = np.zeros(self.num_vertices, dtype=np.float32)
        for idx in indices:
            weights[idx] = 1.0
        return weights

    def to_dict(self) -> dict:
        """Return the mesh dict exactly matching the input_data contract."""
        self._validate()
        vertices = self.grid_vertices().reshape(-1)
        return {
            "vertices": vertices,
            "vertices_sim": vertices.copy(),
            "edges": self.edges().reshape(-1),
            "triangles": self.triangles().reshape(-1),
            "world_matrix": _IDENTITY_WORLD_MATRIX.copy(),
            "object_type": 0,
            "collision_layer": self.collision_layer,
            "grain_dir": self.grain_dir,
            "mass": self.mass,
            "granularity": self.granularity_mm,
            "thickness": self.thickness_mm,
            "friction": self.friction,
            "stretch": np.asarray(self.stretch, dtype=np.float32),
            "bending": np.asarray(self.bending, dtype=np.float32),
            "fixed_vertices": self._pin_weights(self.fixed_vertex_indices),
            "attached_vertices": self._pin_weights(self.attached_vertex_indices),
        }

    def to_input_data(self) -> dict:
        """Return a single-mesh input_data dict (``mesh_list`` + ``sewings``)."""
        return {"mesh_list": [self.to_dict()], "sewings": []}

    def _validate(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise ValueError("grid cloth requires at least 2 rows and 2 columns")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("width_m and height_m must be positive")
        if self.mass <= 0 or self.granularity_mm <= 0 or self.thickness_mm <= 0:
            raise ValueError("mass, granularity_mm and thickness_mm must be positive")
        if not 0.0 <= self.friction <= 1.0:
            raise ValueError("friction must be in [0, 1]")
        if len(self.stretch) != 3 or len(self.bending) != 3:
            raise ValueError("stretch and bending must have length 3")
        num_vertices = self.num_vertices
        for idx in (*self.fixed_vertex_indices, *self.attached_vertex_indices):
            if not 0 <= idx < num_vertices:
                raise ValueError(f"vertex index {idx} out of range [0, {num_vertices})")


def seamed_panels_input_data(
    cols: int = 5,
    rows: int = 3,
    width_m: float = 0.2,
    height_m: float = 0.1,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Two panels joined only by ``sewings``, plus the seam index pair arrays.

    Panel A sits directly above panel B and is pinned along its top edge; the
    two panels share no mesh edge, so the engine's stitch constraint is the
    only thing that can hold B (this is the arrangement the Blender frontend
    builds from its sewing lines, and the one that distinguishes a solver that
    applies the constraint from one that ignores it).

    Returns ``(input_data, upper_indices, lower_indices)`` where the two index
    arrays are the global vertex ids of each stitch pair, so the closure metric
    is ``norm(world[upper] - world[lower])``.
    """
    upper = _shift_mesh(
        _grid_spec(cols, rows, width_m, height_m,
                   fixed=tuple(range(cols * (rows - 1), cols * rows))).to_dict(),
        dy=height_m,
    )
    lower = _grid_spec(cols, rows, width_m, height_m).to_dict()
    pairs = [(i, cols * (rows - 1) + i) for i in range(cols)]
    sewings = [{
        "patterns": [0, 1],
        "stitches": pairs,
        "angle": 0.0,
    }]
    n_upper = cols * rows
    upper_ids = np.arange(cols, dtype=np.int64)
    lower_ids = n_upper + np.asarray([cols * (rows - 1) + i for i in range(cols)],
                                     dtype=np.int64)
    return {"mesh_list": [upper, lower], "sewings": sewings}, upper_ids, lower_ids


def _grid_spec(cols: int, rows: int, width_m: float, height_m: float,
               fixed: tuple[int, ...] = ()) -> "MeshSpec":
    """Grid spec helper (kept separate so the seam scene reads as a scene)."""
    return MeshSpec(width_m=width_m, height_m=height_m, cols=cols, rows=rows,
                    fixed_vertex_indices=fixed)


def _shift_mesh(mesh: dict, dy: float) -> dict:
    """Translate a mesh dict's vertices along +Y, in place."""
    for key in ("vertices", "vertices_sim"):
        values = np.array(mesh[key], dtype=np.float32).reshape(-1, 3)
        values[:, 1] += dy
        mesh[key] = values.reshape(-1)
    return mesh


def validate_mesh_list(mesh_list: list[dict]) -> None:
    """Validate multi-mesh input ordering and per-mesh triangle-edge integrity.

    Cloth objects (object_type 0) must precede obstacles (object_type > 0), and
    every triangle edge must appear in the mesh's edge list.
    """
    saw_obstacle = False
    for index, mesh in enumerate(mesh_list):
        object_type = int(mesh["object_type"])
        if object_type == 0 and saw_obstacle:
            raise ValueError(
                f"cloth mesh at index {index} follows an obstacle; "
                "cloth objects must precede obstacles"
            )
        if object_type != 0:
            saw_obstacle = True

        edges = np.asarray(mesh["edges"], dtype=np.int32).reshape(-1, 2)
        triangles = np.asarray(mesh["triangles"], dtype=np.int32).reshape(-1, 3)
        edge_set = {tuple(e) for e in edges}
        for triangle in triangles:
            for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                if tuple(sorted((int(a), int(b)))) not in edge_set:
                    raise ValueError(
                        f"triangle edge ({a}, {b}) of triangle {triangle.tolist()} "
                        f"missing from edges (mesh index {index})"
                    )


def build_input_data(*specs: MeshSpec) -> dict:
    """Build an input_data dict from one or more mesh specs, validated."""
    mesh_list = [spec.to_dict() for spec in specs]
    validate_mesh_list(mesh_list)
    return {"mesh_list": mesh_list, "sewings": []}


def square_boundary(size_m: float, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed CCW square outline resampled at ~``spacing_m``.

    This is the pattern-curve representation the Blender frontend hands to the
    sampler: a list of boundary points plus the index pairs of the segments
    connecting them (closing the loop).
    """
    if size_m <= 0 or spacing_m <= 0:
        raise ValueError("size_m and spacing_m must be positive")
    per_edge = max(1, int(round(size_m / spacing_m)))
    points = []
    corners = [(0.0, 0.0), (size_m, 0.0), (size_m, size_m), (0.0, size_m)]
    for index, (ax, ay) in enumerate(corners):
        bx, by = corners[(index + 1) % len(corners)]
        for step in range(per_edge):
            t = step / per_edge
            points.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    boundary = np.asarray(points, dtype=np.float32)
    count = len(boundary)
    segments = np.asarray(
        [[i, (i + 1) % count] for i in range(count)], dtype=np.int32)
    return boundary, segments


def sample_panel(
    qydp,
    boundary: np.ndarray,
    segments: np.ndarray,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a closed pattern outline with the engine's own sampler.

    Uses ``qydp.geometry.sample_points``: stratified (jittered-grid) sampling
    with repulsion relaxation, then constrained Delaunay (gDel2D) - the
    pattern-mesh pipeline the Blender frontend drives, not a structured grid
    and not Poisson-disc sampling (see README.md, "Pattern meshing").
    Returns ``(vertices, triangles)`` with ``vertices`` in the XY plane.
    """
    curve_sizes = np.asarray([len(boundary)], dtype=np.int32)
    is_holes = np.asarray([0], dtype=np.int32)
    points, triangles = qydp.geometry.sample_points(
        boundary, segments, curve_sizes, is_holes, float(spacing_m))
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    vertices = np.hstack(
        [points, np.zeros((len(points), 1), dtype=np.float32)])
    return vertices, triangles


def nearest_vertex(vertices: np.ndarray, target_xy: tuple[float, float]) -> int:
    """Index of the vertex closest to ``target_xy`` in the XY plane."""
    target = np.asarray(target_xy, dtype=np.float64)
    offset = np.asarray(vertices, dtype=np.float64)[:, :2] - target
    return int(np.argmin(np.einsum("ij,ij->i", offset, offset)))


def panel_mesh_dict(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    spacing_m: float,
    mass: float = 100.0,
    thickness_mm: float = 0.1,
    friction: float = 0.03,
    collision_layer: int = 0,
    grain_dir: float = 0.0,
    fixed_vertex_indices: tuple[int, ...] = (),
) -> dict:
    """Wrap sampled geometry in the ``input_data`` mesh dict contract.

    The key set mirrors ``MeshSpec.to_dict`` so a sampled panel and a grid
    panel are interchangeable inputs for the harness and for the engine.
    """
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    num_vertices = vertices.shape[0]
    for idx in fixed_vertex_indices:
        if not 0 <= idx < num_vertices:
            raise ValueError(f"vertex index {idx} out of range [0, {num_vertices})")
    pinned = np.zeros(num_vertices, dtype=np.float32)
    for idx in fixed_vertex_indices:
        pinned[idx] = 1.0
    return {
        "vertices": vertices.reshape(-1),
        "vertices_sim": vertices.reshape(-1).copy(),
        "edges": _derive_edges_from_triangles(triangles).reshape(-1),
        "triangles": triangles.reshape(-1),
        "world_matrix": _IDENTITY_WORLD_MATRIX.copy(),
        "object_type": 0,
        "collision_layer": collision_layer,
        "grain_dir": grain_dir,
        "mass": mass,
        "granularity": spacing_m * 1e3,
        "thickness": thickness_mm,
        "friction": friction,
        "stretch": np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        "bending": np.asarray((1.0, 1.0, 1.0), dtype=np.float32),
        "fixed_vertices": pinned,
        "attached_vertices": np.zeros(num_vertices, dtype=np.float32),
    }
