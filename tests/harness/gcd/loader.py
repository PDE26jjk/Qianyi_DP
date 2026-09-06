"""GarmentCodeData element loader.

Maps one dataset element (box mesh, per-vertex segmentation, specification
JSON, and optionally a neutral body OBJ) to the Qianyi_DP ``input_data``
contract: one cloth mesh per panel, vertex-pair sewings derived from the
stitch labels, and (when configured) a body obstacle mesh.

The box mesh is a closed garment whose connected components are the panels:
trimesh UV-duplicates the seam vertices, which splits the raw mesh into one
island per panel. Each panel's UV island is used to split the garment and to
name the panel. Normal vertices keep their box-mesh positions; seam vertices
are re-projected onto each panel's plane with the non-seam (u,v)->3D affine,
so no seam triangle is dropped and no panel vertex is moved. Attachment
semantics are not used (``attached_vertices`` are all zero).

Dataset facts verified against the local GarmentCodeData download (recorded in
LOCAL_DEV.md, gitignored):

- Box meshes are in centimetres with a Z-up axis; UV-duplicate vertices share
  consecutive identical positions, so the original-id map is derived by
  scanning for consecutive duplicates (the PLY header has no ``v_id_map``
  property).
- ``sim_segmentation.txt`` has one label per *original* vertex; labels are
  comma-separated. Stitch vertices are tagged ``stitch_N``; non-stitch
  vertices carry a panel name.
- ``specification.json`` lists panel-edge stitches; the mesh seam labels
  ``stitch_N`` map by index to that list.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from ..meshspec import validate_mesh_list
from .errors import (
    InvalidElementError,
    MissingFileError,
    ScaleMismatchError,
    SeamPairingError,
)

CM_TO_M = 0.01

# Fabric/pattern defaults mirror the Blender frontend defaults used by the
# procedural harness (see harness/meshspec.py); the dataset has no per-element
# fabric parameters in this phase (design D9).
FABRIC_DEFAULTS: dict = {
    "mass": 100.0,
    "granularity": 20.0,
    "thickness": 0.1,
    "friction": 0.03,
    "stretch": (1.0, 1.0, 1.0),
    "bending": (1.0, 1.0, 1.0),
    "collision_layer": 0,
    "grain_dir": 0.0,
}

_IDENTITY_WORLD_MATRIX = np.eye(4, dtype=np.float32)

# Tolerance for detecting UV-duplicate vertices (consecutive identical rows).
_POSITION_EPS = 1e-8


def _derive_v_id_map(vertices: np.ndarray) -> np.ndarray:
    """Map each mesh vertex to its original ID (consecutive duplicates share one).

    UV-duplicate vertices are stored as consecutive rows with identical
    positions; each new position starts a new original ID.
    """
    assert vertices.size > 0
    is_new = np.ones(len(vertices), dtype=bool)
    if len(vertices) > 1:
        is_new[1:] = (
                np.abs(vertices[1:] - vertices[:-1]).max(axis=1) > _POSITION_EPS
        )
    return np.cumsum(is_new) - 1


def _derive_edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    """Return the unique undirected edges implied by a triangle list."""
    pairs = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).astype(np.int32)


def _read_segmentation(path: Path, n_original: int) -> list[str]:
    """Read the per-original-vertex segmentation labels."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) != n_original:
        raise InvalidElementError(
            f"segmentation has {len(lines)} labels but the box mesh maps to "
            f"{n_original} original vertices"
        )
    return lines


def _parse_labels(
        segmentation: list[str], panel_names: set[str]
) -> tuple[np.ndarray, list[set[str]]]:
    """Split labels into a per-vertex panel name and per-vertex stitch sets."""
    labels = [[p.strip() for p in label.split(",")] for label in segmentation]
    stitch_label_size = 0
    for i, label in enumerate(labels):
        if not label[0].startswith('stitch_'):
            stitch_label_size = i
            # stitch_ always above pattern name
            break
    return labels, stitch_label_size


@dataclass
class PanelMesh:
    """A decomposed cloth panel with its local index maps."""

    name: str
    vertices: np.ndarray  # (N, 3) metres, original positions
    vertices_2d: np.ndarray  # (N, 3) metres, 2d positions
    uv: np.ndarray  # (N, 2) raw PLY UV coordinates
    triangles: np.ndarray  # (T, 3) local int32
    edges: np.ndarray  # (E, 2) local int32
    raw_vertex_ids: np.ndarray  # (N,) source PLY vertex rows


def _split_panels_by_uv_islands(
        vertices_m: np.ndarray,
        faces: np.ndarray,
        raw_to_original: np.ndarray,
        uv: np.ndarray,
        stitch_label_size: int,
        labels: list[set[str]],
        panel_names: set[str],
):
    """Split the box mesh into one PanelMesh per UV island.

    The dataset documents that the box mesh's UV islands correspond to the
    sewing-pattern panels and that this is the way to restore the individual
    panel meshes. Each island is a connected component of the raw mesh and is a
    self-consistent patch: its vertices and triangles are taken verbatim, so no
    seam triangle is dropped and no seam vertex is lost or displaced. The panel
    name is the majority over the island's non-seam (panel-labelled) vertices.
    """
    import collections

    n = len(vertices_m)
    face_rows = faces.tolist()
    adjacency = collections.defaultdict(set)
    for a, b, c in face_rows:
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    seen = [False] * n
    islands: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            vertex = stack.pop()
            comp.append(vertex)
            for neighbour in adjacency.get(vertex, ()):
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        comp.sort()
        islands.append(comp)
    # islands.sort(key=len, reverse=True)

    island_of = np.empty(n, dtype=np.int64)
    island_of[:] = -1
    stitchs = {}
    for index, comp in enumerate(islands):
        for vertex in comp:
            island_of[vertex] = index
            oid = raw_to_original[vertex]
            if oid < stitch_label_size:
                label_set = labels[oid]
                for stitch_label in label_set:
                    stitch_label: str
                    for neighbour in adjacency.get(vertex, ()):
                        if stitch_label in labels[raw_to_original[neighbour]]:
                            stitchs.setdefault(stitch_label, []).append((index, vertex))
                            break

    # Since every triangle lies entirely in one island (verified for the
    # dataset), its owning island is that of its first vertex.
    global_to_local = np.full(n, -1, dtype=np.int32)
    for comp in islands:
        global_to_local[comp] = np.arange(len(comp))

    owner = island_of[faces[:, 0]]
    panels: list[PanelMesh] = []
    for index, comp in enumerate(islands):
        local_tris = faces[owner == index]
        assert len(local_tris) > 0, "Empty island!"
        name: str = labels[raw_to_original[comp[-1]]][0]
        assert not name.startswith('stitch_'), f"name = {name}"

        raw_vertex_ids = np.array(comp)
        remapped = global_to_local[local_tris].astype(np.int32)

        is_seam = raw_to_original[raw_vertex_ids] < stitch_label_size
        seam_locals = np.where(is_seam)[0].tolist()
        non_seam_locals = np.where(~is_seam)[0].tolist()
        panel_vertices = vertices_m[raw_vertex_ids].astype(np.float64)

        if seam_locals and len(non_seam_locals) >= 6:
            uvs_non_seam = uv[raw_vertex_ids[non_seam_locals]].astype(np.float64)
            pos_non_seam = panel_vertices[non_seam_locals]
            design = np.column_stack([uvs_non_seam, np.ones(len(uvs_non_seam))])
            affine, *_ = np.linalg.lstsq(design, pos_non_seam, rcond=None)

            # Recover the seam vertices using affine transformation
            uvs_seam = uv[raw_vertex_ids[seam_locals]].astype(np.float64)
            design_seam = np.column_stack([uvs_seam, np.ones(len(uvs_seam))])
            panel_vertices[seam_locals] = design_seam @ affine

        # Compute vertices_2d: Project all vertices onto the panel plane, then rotate to the XY plane.
        centroid = panel_vertices.mean(axis=0)
        centered = panel_vertices - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        plane_basis = vh[:2].T  # 3x2
        coords_2d = (centered @ plane_basis)        # (n, 2)
        vertices_2d = np.zeros((len(panel_vertices), 3), dtype=np.float32)
        vertices_2d[:, :2] = coords_2d.astype(np.float32)

        edges = _derive_edges_from_triangles(remapped)
        panels.append(PanelMesh(
            name=name,
            vertices=np.ascontiguousarray(panel_vertices.astype(np.float32)),
            vertices_2d=np.ascontiguousarray(vertices_2d),
            uv=np.ascontiguousarray(uv[raw_vertex_ids], dtype=np.float32),
            triangles=np.ascontiguousarray(remapped),
            edges=edges,
            raw_vertex_ids=raw_vertex_ids.astype(np.int32),
        ))

    sewings = []
    for stitch in stitchs.values():
        # patterns = [s[0] for s in stitch]
        # verties = [s[1] for s in stitch]
        vertex_to_pattern = {v: p for p, v in stitch}
        vertex_set = set(vertex_to_pattern.keys())

        adj = {v: adjacency[v] & vertex_set for v in vertex_set}

        endpoints = [v for v in adj if len(adj[v]) == 1]
        chains = []
        remaining = set(endpoints)
        while remaining:
            start = remaining.pop()
            chain = [start]
            prev = None
            cur = start
            while True:
                nxt = next((n for n in adj[cur] if n != prev), None)
                if nxt is None:
                    break
                chain.append(nxt)
                prev, cur = cur, nxt
            remaining.discard(chain[-1])
            chains.append(chain)
        assert len(chains) <= 2, f"len(chains)=={len(chains)}!!!"
        if len(chains) == 1:
            chain = chains[0]
            assert len(chain) % 2 != 0, "Single sewn chain must be an odd number to be evenly divisible."
            mid = len(chain) // 2
            chain1 = list(reversed(chain[:mid + 1]))
            chain2 = chain[mid:]
            chains = [chain1, chain2]
        else:  # len(chains) == 2
            chain1, chain2 = chains
            if raw_to_original[chain1[0]] != raw_to_original[chain2[0]]:
                chain2 = list(reversed(chain2))
            chains = [chain1, chain2]
        assert len(chains[0]) == len(chains[1]), "Inconsistent Stitching Chain Length"
        for v1, v2 in zip(chains[0], chains[1]):
            assert raw_to_original[v1] == raw_to_original[v2], "Sewing vertex mismatch"
        local_pairs = list(zip(global_to_local[chain1], global_to_local[chain2]))

        sewings.append(
            {
                "patterns": [vertex_to_pattern[chains[0][0]], vertex_to_pattern[chains[1][0]]],
                "stitches": local_pairs,
                "angle": 0.0,
            }
        )

    return panels, sewings


def _compute_face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Per-face normals for an obstacle mesh (object_type=1)."""
    tri = vertices[triangles]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return np.ascontiguousarray(normals / lengths, dtype=np.float32)

_YUP_TO_ZUP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

def load_body_mesh(
        body_obj: Path,
        garment_bounds_m: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Load the neutral-body OBJ as an obstacle mesh.

    The body is stored in metres with a Y-up axis; it is rotated to the
    engine's Z-up convention (gravity acts along -Z). Returns the mesh dict
    and a stats record used by the loader report (bounding-box alignment is
    reported, not enforced here; the calibration pass sets the tolerances).
    """
    body = trimesh.load(str(body_obj), process=False)
    vertices = np.asarray(body.vertices, dtype=np.float64) @ _YUP_TO_ZUP.T
    triangles = np.asarray(body.faces, dtype=np.int32)
    normals = _compute_face_normals(vertices, triangles)
    mesh = {
        "vertices": np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1),
        "edges": _derive_edges_from_triangles(triangles).reshape(-1),
        "triangles": triangles.reshape(-1),
        "world_matrix": _IDENTITY_WORLD_MATRIX.copy(),
        "object_type": 1,
        "collision_layer": 0,
        "normals": normals.reshape(-1),
        "mass": 1.0,
    }
    body_bounds = np.asarray(vertices).reshape(-1, 3)
    body_bounds = np.stack(
        [body_bounds.min(axis=0), body_bounds.max(axis=0)], axis=0
    )
    stats: dict = {
        "body_vertices": len(vertices),
        "body_faces": len(triangles),
        "body_bounds": body_bounds.tolist(),
    }
    if garment_bounds_m is not None:
        garment_center = garment_bounds_m.mean(axis=0)
        body_center = body_bounds.mean(axis=0)
        stats["garment_bounds_m"] = garment_bounds_m.tolist()
        stats["garment_center_m"] = garment_center.tolist()
        stats["body_center_m"] = body_center.tolist()
        stats["center_offset_m"] = float(
            np.linalg.norm(garment_center - body_center)
        )
        stats["aligned"] = bool(stats["center_offset_m"] < 1.0)
    return mesh, stats


def _element_files(element_dir: Path) -> dict[str, Path]:
    element_id = element_dir.name
    required = {
        "boxmesh": element_dir / f"{element_id}_boxmesh.ply",
        "segmentation": element_dir / f"{element_id}_sim_segmentation.txt",
        "specification": element_dir / f"{element_id}_specification.json",
        "vertex_labels": element_dir / f"{element_id}_vertex_labels.yaml",
    }
    optional = {
        "sim_ply": element_dir / f"{element_id}_sim.ply",
        "design_params": element_dir / f"{element_id}_design_params.yaml",
        "orig_lens": element_dir / f"{element_id}_orig_lens.pickle",
    }
    for key, path in required.items():
        if not path.is_file():
            raise MissingFileError(f"element {element_id} is missing {path.name}")
    return {**required, **optional}


@dataclass
class LoadedElement:
    """A loaded element: input_data plus the loader report."""

    element_id: str
    element_dir: Path
    input_data: dict
    report: dict
    reference_vertices: np.ndarray | None = None
    reference_faces: np.ndarray | None = None
    panel_triangle_assignment: dict[str, np.ndarray] = field(default_factory=dict)


# Bump this whenever the loader's geometry/restore/fit algorithm changes: the
# on-disk element cache is keyed by it so a stale cache is never returned.
LOADER_CACHE_VERSION = "7"
_GCD_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "gcd_cache"


def _element_input_fingerprint(
        element_dir: Path,
        files: dict[str, Path],
        body_obj: Path | None,
) -> str:
    """Fingerprint the element's inputs plus the loader version.

    The cache is per-element and per-body: any change in an input file's size,
    mtime, the body object, or the loader algorithm invalidates it. This keeps
    the "first load runs the expensive restore/fit, repeat loads read the
    cached geometry" behaviour safe across code and data edits.
    """
    hasher = hashlib.sha1()
    hasher.update(LOADER_CACHE_VERSION.encode("utf-8"))
    hasher.update(element_dir.name.encode("utf-8"))
    if body_obj is not None:
        body_path = Path(body_obj).resolve()
        hasher.update(b"|body|")
        hasher.update(str(body_path).encode("utf-8"))
        if body_path.is_file():
            st = body_path.stat()
            hasher.update(f"|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
    for name in sorted(files):
        path = files[name]
        if not path.is_file():
            continue
        st = path.stat()
        hasher.update(f"|{name}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
    return hasher.hexdigest()[:24]


def _element_cache_path(
        element_dir: Path,
        files: dict[str, Path],
        body_obj: Path | None,
) -> Path:
    cache_dir = _GCD_CACHE_DIR / element_dir.name
    return cache_dir / (
            _element_input_fingerprint(element_dir, files, body_obj) + ".pkl"
    )


def _load_cached_element(
        element_dir: Path,
        body_obj: Path | None,
) -> LoadedElement | None:
    try:
        files = _element_files(element_dir)
        cache_path = _element_cache_path(element_dir, files, body_obj)
        if not cache_path.is_file():
            return None
        payload = pickle.loads(cache_path.read_bytes())
        element = payload.get("element")
        return element if isinstance(element, LoadedElement) else None
    except Exception:
        return None


def _store_cached_element(
        element_dir: Path,
        body_obj: Path | None,
        element: LoadedElement,
) -> None:
    try:
        files = _element_files(element_dir)
        cache_path = _element_cache_path(element_dir, files, body_obj)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps({"element": element}))
    except Exception:
        # Caching is best-effort for debugging/performance; never block a load.
        pass


def _load_element_impl(
        element_dir: Path,
        body_obj: Path | None = None,
) -> LoadedElement:
    """Load one GarmentCodeData element into an input_data dict."""
    element_dir = Path(element_dir)
    files = _element_files(element_dir)
    element_id = element_dir.name

    mesh = trimesh.load(str(files["boxmesh"]), process=False)
    vertices_cm = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    uv = getattr(mesh.visual, "uv", None)

    uv = np.asarray(uv, dtype=np.float64)
    v_id_map = _derive_v_id_map(vertices_cm)
    n_original = int(v_id_map.max()) + 1 if len(v_id_map) else 0

    segmentation = _read_segmentation(files["segmentation"], n_original)
    specification = json.loads(files["specification"].read_text(encoding="utf-8"))
    panel_names = set(specification.get("pattern", {}).get("panels", {}))
    if not panel_names:
        raise InvalidElementError(
            f"element {element_id} has no panels in its specification"
        )

    labels, stitch_label_size = _parse_labels(segmentation, panel_names)
    # The dataset is Y-up; the engine's gravity is fixed along -Z, so the
    # garment is rotated to Z-up before the centimetre-to-metre conversion.
    vertices_m = (vertices_cm @ _YUP_TO_ZUP.T) * CM_TO_M

    panels, sewings = _split_panels_by_uv_islands(
        vertices_m, faces, v_id_map, uv, stitch_label_size, labels, panel_names
    )

    mesh_list: list[dict] = []
    for panel in panels:
        mesh_list.append(
            {
                "vertices": panel.vertices_2d.reshape(-1),
                "vertices_sim": panel.vertices.reshape(-1),
                "edges": panel.edges.reshape(-1),
                "triangles": panel.triangles.reshape(-1),
                "world_matrix": _IDENTITY_WORLD_MATRIX.copy(),
                "object_type": 0,
                "collision_layer": FABRIC_DEFAULTS["collision_layer"],
                "grain_dir": FABRIC_DEFAULTS["grain_dir"],
                "mass": FABRIC_DEFAULTS["mass"],
                "granularity": FABRIC_DEFAULTS["granularity"],
                "thickness": FABRIC_DEFAULTS["thickness"],
                "friction": FABRIC_DEFAULTS["friction"],
                "stretch": np.asarray(FABRIC_DEFAULTS["stretch"], dtype=np.float32),
                "bending": np.asarray(FABRIC_DEFAULTS["bending"], dtype=np.float32),
                "fixed_vertices": np.zeros(len(panel.vertices), dtype=np.float32),
                "attached_vertices": np.zeros(len(panel.vertices), dtype=np.float32),
            }
        )

    restored_vertices = np.vstack([panel.vertices for panel in panels])
    garment_bounds_m = np.stack(
        [restored_vertices.min(axis=0), restored_vertices.max(axis=0)], axis=0
    )
    body_mesh: dict | None = None
    body_stats: dict = {}
    if body_obj is not None:
        body_mesh, body_stats = load_body_mesh(body_obj, garment_bounds_m)
        mesh_list.append(body_mesh)

    validate_mesh_list(mesh_list)
    input_data = {"mesh_list": mesh_list, "sewings": sewings}

    reference_vertices = None
    reference_faces = None
    if files["sim_ply"].is_file():
        reference = trimesh.load(str(files["sim_ply"]), process=False)
        reference_vertices = (
                                     np.asarray(reference.vertices, dtype=np.float64) @ _YUP_TO_ZUP.T
                             ) * CM_TO_M
        reference_faces = np.asarray(reference.faces, dtype=np.int32)

    report: dict = {
        "element_id": element_id,
        "mesh_vertices": int(len(vertices_m)),
        "original_vertices": int(n_original),
        "faces": int(len(faces)),
        "panels": {panel.name: len(panel.vertices) for panel in panels},
        "panel_count": len(panels),
        "stitch_count": len(sewings),
        "sewing_entries": len(sewings),
        "mismatches": [],
        "garment_bounds_m": garment_bounds_m.tolist(),
        "garment_bbox_m": [
            restored_vertices.min(axis=0).tolist(),
            restored_vertices.max(axis=0).tolist(),
        ],
        "body": body_stats,
    }

    return LoadedElement(
        element_id=element_id,
        element_dir=element_dir,
        input_data=input_data,
        report=report,
        reference_vertices=reference_vertices,
        reference_faces=reference_faces,
    )


def load_element(
        element_dir: Path,
        body_obj: Path | None = None,
) -> LoadedElement:
    """Load one GarmentCodeData element, using a per-element disk cache.

    The first call for an element runs the expensive seam restore + edge-length
    fitting and writes the result to ``tests/artifacts/gcd_cache/``. Later calls
    with unchanged inputs (same element files, body object, loader version) hit
    the cache and return immediately, so loading many cases does not repeat the
    costly geometry work every time.
    """
    element_dir = Path(element_dir)
    # cached = _load_cached_element(element_dir, body_obj)
    # if cached is not None:
    #     return cached
    element = _load_element_impl(element_dir, body_obj)
    # # _store_cached_element(element_dir, body_obj, element)
    return element
