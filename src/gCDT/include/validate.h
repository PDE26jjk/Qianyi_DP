#pragma once

struct int3;
typename Point2d;

// Device-side validation of a triangulation.
//
// Returns the number of suspicious triangles: repeated vertex indices,
// zero-area triangles, duplicated triangles, or triangles that touch an edge
// used by more than two triangles (non-manifold / overlapping). Coincident
// vertices are counted as well.
//
// Root cause of the defect this guards against
// -------------------------------------------
// Constraint enforcement (addConstraints.cu) builds, per intersected constraint,
// the polygon on each side of the segment and triangulates it by perpendicular
// distance (formPolygon_s2 -> calDistance -> makeTree -> reTriangle).
//
// Trigger: several constraint segments lying on one supporting line - for
// example several hole outlines whose edges share a line, which is an ordinary
// CAD input. Verified causally: the same scene with the holes moved by a
// fraction of a cell (so no two edges stay collinear) never fails.
//
// Failure mechanism, read off the polygon dump (GCDT_DUMP_POLY=<path>, written
// by addConstraints.cu): for the affected constraint the strip walk emits the
// *same vertex twice* into one side's polygon (a backtrack, e.g. the vertex
// sequence ... 2549, 2550, 2356, 2550, 2551 ...). Both copies carry the same
// distance, so reTriangle's tree search resolves their neighbours differently
// and two polygon vertices end up falling back to the same pair of constraint
// endpoints. Those two triangles share the constraint edge on the same side:
// a small overlap plus a non-manifold edge.
//
// Two candidate fixes were evaluated against that dump and rejected because
// they cannot change this failure: a secondary tie-break key cannot separate
// two copies of the same vertex (they are equal in every key), and dropping
// vertices that lie exactly on the constraint line does not apply either - the
// duplicated vertex measures 1.4e14 away from the line, only the constraint's
// own endpoints sit at distance zero. The real fix belongs to the strip walk /
// polygon construction; until that is done, gcdtBuildMesh() perturbs the
// quantised vertices by a few units (2^-30 of the domain) so this branch is not
// entered, and this validation proves that what is returned is valid.
int gcdtValidateMeshDetailed(const Point2d* points, int numPoints, const int3* tris, int numTriangles);

// Same as above, returns true when the mesh is valid; optionally reports the
// number of suspicious triangles.
bool gcdtValidateMesh(const Point2d* points, int numPoints, const int3* tris, int numTriangles,
                      int* badTriangles = nullptr);

// Applies a deterministic perturbation of at most `amplitude` quantisation
// units (1 unit = 2^-30 of the normalised domain) to every vertex. Used to
// leave degenerate constraint configurations behind when a run produced an
// invalid mesh.
void gcdtPerturbPoints(Point2d* points, int numPoints, unsigned int seed, int amplitude);
