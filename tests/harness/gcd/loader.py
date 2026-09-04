"""GarmentCodeData element loader.

Maps one dataset element (box mesh, per-vertex segmentation, specification
JSON, semantic vertex labels, and optionally a neutral body OBJ) to the
Qianyi_DP ``input_data`` contract: one cloth mesh per panel, vertex-pair
sewings derived from stitch labels, attachment weights from semantic labels,
and (when configured) a body obstacle mesh.

Dataset facts verified against the local GarmentCodeData download (recorded in
LOCAL_DEV.md, gitignored):

- Box meshes are in centimetres with a Z-up axis; UV-duplicate vertices share
  consecutive identical positions, so the original-id map is derived by
  scanning for consecutive duplicates (the PLY header has no ``v_id_map``
  property).
- ``sim_segmentation.txt`` has one label per *original* vertex; labels are
  comma-separated (a vertex may belong to several seams at a junction).
- ``specification.json`` lists panel-edge stitches; the mesh seam labels
  ``stitch_N`` map by index to that list and are cross-checked against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml

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
    if vertices.size == 0:
        return np.empty(0, dtype=np.int64)
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
    panel_of = np.empty(len(segmentation), dtype=object)
    panel_of[:] = ""
    stitch_sets: list[set[str]] = []
    for i, label in enumerate(segmentation):
        parts = [p.strip() for p in label.split(",")]
        panels = [p for p in parts if p in panel_names]
        if len(panels) > 1:
            raise InvalidElementError(
                f"vertex {i} has multiple panel labels: {panels}"
            )
        if panels:
            panel_of[i] = panels[0]
        stitch_sets.append({p for p in parts if p.startswith("stitch_")})
    return panel_of, stitch_sets


def _triangle_panel(face: np.ndarray, panel_of: np.ndarray, panel_names: set[str]) -> str:
    """Return the panel of a triangle.

    Vertices are original IDs; a triangle whose vertices carry several panel
    labels (rare boundary case) is assigned by majority vote with an
    alphabetical tie-break, and an all-stitch triangle returns ''.
    """
    counts: dict[str, int] = {}
    for vertex_id in face:
        label = panel_of[int(vertex_id)]
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _assign_triangle_panels(
    faces: np.ndarray,
    panel_of: np.ndarray,
    panel_names: set[str],
) -> np.ndarray:
    """Assign every triangle to exactly one panel.

    Triangles with a panel-labelled vertex go to that panel; the small number
    of all-stitch seam-bridging triangles (verified locally) are assigned by
    majority vote of their already-assigned edge neighbours, keeping every
    vertex inside a panel mesh (no zero-mass dangling vertices).
    """
    n_faces = len(faces)
    assignment = np.empty(n_faces, dtype=object)
    assignment[:] = ""

    for i, face in enumerate(faces):
        assignment[i] = _triangle_panel(face, panel_of, panel_names)

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for i, face in enumerate(faces):
        a, b, c = (int(x) for x in face)
        for edge in ((a, b), (b, c), (c, a)):
            key = (min(edge), max(edge))
            edge_to_faces.setdefault(key, []).append(i)

    pending = [i for i in range(n_faces) if not assignment[i]]
    while pending:
        progress = False
        next_pending: list[int] = []
        for i in pending:
            face = faces[i]
            neighbours: list[str] = []
            a, b, c = (int(x) for x in face)
            for edge in ((a, b), (b, c), (c, a)):
                key = (min(edge), max(edge))
                for j in edge_to_faces.get(key, ()):
                    if assignment[j]:
                        neighbours.append(assignment[j])
            if not neighbours:
                next_pending.append(i)
                continue
            counts: dict[str, int] = {}
            for panel in neighbours:
                counts[panel] = counts.get(panel, 0) + 1
            best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            assignment[i] = best[0]
            progress = True
        pending = next_pending
        if not progress and pending:
            raise InvalidElementError(
                "unassignable all-stitch triangles with no panel neighbours: "
                f"{pending[:10]}..."
            )
    return assignment


@dataclass
class PanelMesh:
    """A decomposed cloth panel with its local index maps."""

    name: str
    vertices: np.ndarray  # (N, 3) metres, original positions
    triangles: np.ndarray  # (T, 3) local int32
    edges: np.ndarray  # (E, 2) local int32
    original_ids: np.ndarray  # (N,) original vertex ids, stable order
    local_index: dict[int, int]  # original id -> local index
    face_ids: np.ndarray  # original face-row indices owned by this panel


def _split_panels(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    panel_of: np.ndarray,
    panel_names: set[str],
) -> dict[str, PanelMesh]:
    """Split the box mesh into one PanelMesh per segmentation panel."""
    assignment = _assign_triangle_panels(faces, panel_of, panel_names)
    panels: dict[str, PanelMesh] = {}
    for name in sorted(panel_names):
        triangle_ids = np.flatnonzero(assignment == name)
        if len(triangle_ids) == 0:
            raise InvalidElementError(f"panel {name!r} has no triangles")
        candidates = faces[triangle_ids].astype(np.int64)
        # Drop degenerate triangles: after UV-duplicate collapse a seam
        # bridging triangle can reference the same original id twice (zero
        # area), and collinear triangles are zero-area too. Such triangles
        # would give their vertices zero mass and crash the linear solver
        # (1/0 invMass), so they are excluded from the panel mesh.
        duplicate_vertex = (
            (candidates[:, 0] == candidates[:, 1])
            | (candidates[:, 1] == candidates[:, 2])
            | (candidates[:, 0] == candidates[:, 2])
        )
        tri_positions = vertices_m[candidates]
        tri_areas = 0.5 * np.linalg.norm(
            np.cross(
                tri_positions[:, 1] - tri_positions[:, 0],
                tri_positions[:, 2] - tri_positions[:, 0],
            ),
            axis=1,
        )
        # Drop near-degenerate slivers too: their stiffness scales as 1/area^2
        # and blows up the linear solve (verified locally on seam-adjacent
        # triangles with aspect ratios above 1e6).
        min_area = max(1e-8, 1e-3 * float(np.median(tri_areas)))
        keep = ~(duplicate_vertex | (tri_areas < min_area))
        triangle_ids = triangle_ids[keep]
        local_triangles = candidates[keep]
        if len(triangle_ids) == 0:
            raise InvalidElementError(f"panel {name!r} has no valid triangles")
        original_ids, first_pos = np.unique(
            local_triangles.reshape(-1), return_index=True
        )
        order = np.argsort(first_pos)
        original_ids = original_ids[order]
        local_index = {int(oid): i for i, oid in enumerate(original_ids)}
        remapped = np.vectorize(local_index.__getitem__)(
            local_triangles
        ).astype(np.int32)
        edges = _derive_edges_from_triangles(remapped)
        panels[name] = PanelMesh(
            name=name,
            vertices=np.ascontiguousarray(vertices_m[original_ids], dtype=np.float32),
            triangles=np.ascontiguousarray(remapped),
            edges=edges,
            original_ids=original_ids.astype(np.int64),
            local_index=local_index,
            face_ids=triangle_ids.astype(np.int64),
        )
    return panels


def _order_chain(
    ids: np.ndarray, edges: set[tuple[int, int]]
) -> list[int]:
    """Order the vertices of an open chain from an endpoint."""
    adjacency: dict[int, list[int]] = {}
    degree: dict[int, int] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    endpoints = [v for v, d in degree.items() if d == 1]
    path = [endpoints[0]]
    prev = None
    current = endpoints[0]
    while True:
        candidates = [n for n in adjacency.get(current, []) if n != prev]
        if not candidates:
            break
        prev, current = current, candidates[0]
        path.append(current)
    return path


def _connected_components(
    ids: set[int], edges: set[tuple[int, int]]
) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in ids:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _pair_seam(
    name: str, member_ids: np.ndarray, mesh_edges: np.ndarray
) -> tuple[list[int], dict]:
    """Order a stitch chain as a single open boundary path.

    The box mesh is a closed garment: the two panels' boundary chains coincide
    along the seam curve, so each original vertex on the chain is shared by
    both sides (UV-duplicate copies). The ordered open path is therefore the
    single chain used by the sewing builder, which binds the coincident panel
    copies pair-by-pair. A branched or multi-component chain is a genuine
    pairing failure (junction vertices still keep the chain simple on the
    verified elements).
    """
    members = set(int(i) for i in member_ids)
    edges: set[tuple[int, int]] = set()
    for a, b in mesh_edges:
        ia, ib = int(a), int(b)
        if ia in members and ib in members:
            edges.add((min(ia, ib), max(ia, ib)))

    components = _connected_components(members, edges)
    meta = {"member_count": len(members), "edge_count": len(edges)}
    if len(components) != 1:
        sizes = sorted(len(c) for c in components)
        raise SeamPairingError(
            f"seam {name} splits into {len(components)} components ({sizes}); "
            "expected a single boundary chain"
        )

    degree: dict[int, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    endpoints = [v for v, d in degree.items() if d == 1]
    branched = [v for v, d in degree.items() if d > 2]
    if branched or len(endpoints) != 2:
        raise SeamPairingError(
            f"seam {name} is not a simple open chain "
            f"(endpoints={len(endpoints)}, branched={branched})"
        )

    path = _order_chain(member_ids, edges)
    meta["chain_length"] = len(path)
    return path, meta


def _build_sewings(
    seam_paths: dict[str, list[int]],
    spec_panels_by_stitch: dict[str, set[str]],
    panels: dict[str, PanelMesh],
) -> list[dict]:
    """Build input_data sewings from ordered seam chains.

    The engine's sewing constraint binds vertex pairs (a vertex spring with a
    2 mm rest length), and its init path requires consecutive stitches of one
    sewing entry to be mesh-edge-adjacent on each side. Because the closed box
    mesh stores coincident seam copies per panel, each seam position ``p``
    yields the identity-position pair ``(panel A : p, panel B : p)``; both
    sides then traverse the full chain so every consecutive-stitch edge exists.
    Same-panel seams (darts) emit one internal entry.
    """
    mesh_index = {name: i for i, name in enumerate(sorted(panels))}
    sewings: list[dict] = []
    for name in sorted(seam_paths, key=lambda s: int(s.split("_")[1])):
        path = seam_paths[name]
        spec_panels = spec_panels_by_stitch.get(name, set())
        panel_list = sorted(spec_panels) if spec_panels else sorted(panels)
        pa = panel_list[0]
        pb = panel_list[1] if len(panel_list) > 1 else panel_list[0]
        local_pairs: list[tuple[int, int]] = []
        for vertex in path:
            la = panels[pa].local_index.get(int(vertex))
            lb = panels[pb].local_index.get(int(vertex))
            if la is not None and lb is not None:
                local_pairs.append((la, lb))
        if not local_pairs:
            continue
        stitches = np.asarray(local_pairs, dtype=np.int32)
        sewings.append(
            {
                "patterns": [mesh_index[pa], mesh_index[pb]],
                "stitches": stitches,
                "angle": 0.0,
            }
        )
    return sewings


def _ensure_seam_edges(
    seam_paths: dict[str, list[int]],
    spec_panels_by_stitch: dict[str, set[str]],
    panels: dict[str, PanelMesh],
) -> dict[str, int]:
    """Add missing seam-chain edges to the owning panels' edge lists.

    Dropping degenerate seam-bridging triangles can remove a chain edge that
    the engine's sewing init requires (consecutive stitches must be connected
    by a mesh edge on each side). The seam line is re-added as a plain edge
    (no triangle), which keeps the init path consistent.
    """
    edge_sets = {
        name: set(map(tuple, panel.edges.tolist()))
        for name, panel in panels.items()
    }
    added: dict[str, int] = {}
    for name in sorted(seam_paths, key=lambda s: int(s.split("_")[1])):
        path = seam_paths[name]
        spec_panels = spec_panels_by_stitch.get(name, set())
        panel_list = sorted(spec_panels) if spec_panels else sorted(panels)
        count = 0
        for a, b in zip(path[:-1], path[1:]):
            for panel_name in panel_list:
                panel = panels[panel_name]
                la = panel.local_index.get(int(a))
                lb = panel.local_index.get(int(b))
                if la is None or lb is None:
                    continue
                edge = tuple(sorted((la, lb)))
                if edge not in edge_sets[panel_name]:
                    panel.edges = np.vstack(
                        [panel.edges, np.asarray([edge], dtype=np.int32)]
                    )
                    edge_sets[panel_name].add(edge)
                    count += 1
        added[name] = count
    return added


def _cross_check_spec(
    specification: dict,
    seam_paths: dict[str, list[int]],
    panel_membership: dict[int, set[str]],
) -> list[dict]:
    """Cross-check mesh seams against the specification's panel-edge stitches.

    Returns a list of mismatch records (reported per element, not fatal).
    A spec panel missing from the mesh seam's actual panel membership is a
    mismatch; extra panels (shared junction vertices) are expected and ignored.
    """
    spec_stitches = specification.get("pattern", {}).get("stitches", [])
    mismatches: list[dict] = []
    for name, path in seam_paths.items():
        index = int(name.split("_")[1])
        if index >= len(spec_stitches):
            mismatches.append(
                {"stitch": name, "reason": "no matching specification stitch"}
            )
            continue
        spec_entry = spec_stitches[index]
        spec_panels = {
            str(entry["panel"])
            for entry in spec_entry
            if isinstance(entry, dict) and "panel" in entry
        }
        mesh_panels = {
            panel
            for v in path
            for panel in panel_membership.get(int(v), set())
        }
        missing = spec_panels - mesh_panels
        if missing:
            mismatches.append(
                {
                    "stitch": name,
                    "spec_panels": sorted(spec_panels),
                    "mesh_panels": sorted(mesh_panels),
                    "missing": sorted(missing),
                }
            )
    return mismatches


def _load_attachments(
    vertex_labels: dict, panels: dict[str, PanelMesh]
) -> dict[str, np.ndarray]:
    """Map semantic vertex labels to per-panel attachment weight arrays."""
    attached_ids = {
        int(v) for values in vertex_labels.values() for v in values
    }
    weights: dict[str, np.ndarray] = {}
    for name, panel in panels.items():
        w = np.zeros(len(panel.vertices), dtype=np.float32)
        for local, oid in enumerate(panel.original_ids.tolist()):
            if oid in attached_ids:
                w[local] = 1.0
        weights[name] = w
    return weights


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


def load_element(
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
    v_id_map = _derive_v_id_map(vertices_cm)
    n_original = int(v_id_map.max()) + 1 if len(v_id_map) else 0
    faces_orig = v_id_map[faces].astype(np.int64)

    segmentation = _read_segmentation(files["segmentation"], n_original)
    specification = json.loads(files["specification"].read_text(encoding="utf-8"))
    panel_names = set(specification.get("pattern", {}).get("panels", {}))
    if not panel_names:
        raise InvalidElementError(
            f"element {element_id} has no panels in its specification"
        )

    panel_of, stitch_sets = _parse_labels(segmentation, panel_names)
    # The dataset is Y-up; the engine's gravity is fixed along -Z, so the
    # garment is rotated to Z-up before the centimetre-to-metre conversion.
    vertices_m = (vertices_cm @ _YUP_TO_ZUP.T) * CM_TO_M
    # UV-duplicate copies are not contiguous with their originals in general
    # (verified: the dataset stores them in adjacent pairs), so the
    # per-original-id position table uses each id's first box occurrence.
    first_occurrence = np.unique(v_id_map, return_index=True)[1]
    vertices_by_id = vertices_m[first_occurrence]

    panels = _split_panels(vertices_by_id, faces_orig, panel_of, panel_names)
    edges_orig = _derive_edges_from_triangles(faces_orig)
    panel_membership: dict[int, set[str]] = {}
    for name, panel in panels.items():
        for oid in panel.original_ids.tolist():
            panel_membership.setdefault(int(oid), set()).add(name)

    seam_paths: dict[str, list[int]] = {}
    stitch_meta: dict[str, dict] = {}
    for name in sorted(
        {s for sets in stitch_sets for s in sets},
        key=lambda s: int(s.split("_")[1]),
    ):
        member_ids = np.asarray(
            [i for i, s in enumerate(stitch_sets) if name in s], dtype=np.int64
        )
        path, meta = _pair_seam(name, member_ids, edges_orig)
        seam_paths[name] = path
        stitch_meta[name] = meta

    spec_stitches = specification.get("pattern", {}).get("stitches", [])
    spec_panels_by_stitch: dict[str, set[str]] = {}
    for index, entry in enumerate(spec_stitches):
        spec_panels_by_stitch[f"stitch_{index}"] = {
            str(item["panel"]) for item in entry if isinstance(item, dict)
        }
    mismatches = _cross_check_spec(
        specification, seam_paths, panel_membership
    )
    attachments = _load_attachments(
        yaml.safe_load(files["vertex_labels"].read_text(encoding="utf-8")) or {},
        panels,
    )

    sewings = _build_sewings(seam_paths, spec_panels_by_stitch, panels)
    added_seam_edges = _ensure_seam_edges(
        seam_paths, spec_panels_by_stitch, panels
    )

    mesh_list: list[dict] = []
    for name in sorted(panels):
        panel = panels[name]
        vertices_flat = panel.vertices.reshape(-1)
        mesh_list.append(
            {
                "vertices": vertices_flat,
                "vertices_sim": vertices_flat.copy(),
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
                "attached_vertices": attachments[name],
            }
        )

    garment_bounds_m = np.stack(
        [vertices_by_id.min(axis=0), vertices_by_id.max(axis=0)], axis=0
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
        "panels": {name: len(panel.vertices) for name, panel in panels.items()},
        "panel_count": len(panels),
        "stitch_count": len(seam_paths),
        "stitch_meta": {
            name: {"members": meta["member_count"], "chain_length": meta["chain_length"]}
            for name, meta in stitch_meta.items()
        },
        "sewing_entries": len(sewings),
        "mismatches": mismatches,
        "garment_bounds_m": garment_bounds_m.tolist(),
        "garment_bbox_m": [
            vertices_by_id.min(axis=0).tolist(),
            vertices_by_id.max(axis=0).tolist(),
        ],
        "body": body_stats,
        "added_seam_edges": added_seam_edges,
    }

    return LoadedElement(
        element_id=element_id,
        element_dir=element_dir,
        input_data=input_data,
        report=report,
        reference_vertices=reference_vertices,
        reference_faces=reference_faces,
        panel_triangle_assignment={
            name: panel.face_ids for name, panel in panels.items()
        },
    )
