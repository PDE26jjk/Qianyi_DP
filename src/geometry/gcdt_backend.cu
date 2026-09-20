// Optional triangulation backend: gCDT (see src/gCDT).
//
// The sampler hands over its point set plus the boundary/hole constraints; gCDT
// returns a constrained Delaunay triangulation on its own (quantised, permuted)
// vertex array, so the result is mapped back to the sampler's vertex ids. gCDT
// validates every run and retries degenerate configurations internally; if it
// cannot produce a valid mesh the sampler falls back to gDel2D.

#include <cuda_runtime.h>

#include <vector>

#include "pipeline.h"

bool gcdt_2d_impl(const std::vector<float2>& points, const std::vector<int2>& constraints,
    std::vector<int3>& triangles) {
    const int num_points = (int)points.size();
    if ( num_points < 3 ) return false;

    std::vector<double> xy((size_t)num_points * 2);
    for ( int i = 0; i < num_points; ++i ) {
        xy[2 * i] = (double)points[i].x;
        xy[2 * i + 1] = (double)points[i].y;
    }

    GcdtBuildOptions options;
    // Single attempt inside the engine: measurements on pattern inputs show the
    // perturbation retry only rescues rare degenerate runs (it never rescued a
    // hole pattern), so retrying here would multiply the cost before the
    // caller's gDel2D fallback. The standalone driver can retry more.
    options.maxAttempts = 1;
    options.perturbAmplitude = 4;
    options.memoryLimitBytes = 128ull * 1024ull * 1024ull;
    options.dropHelperCorners = true;

    GcdtMesh mesh;
    GcdtBuildStats stats;
    const int2* cons = constraints.empty() ? nullptr : constraints.data();
    if ( !gcdtBuildMesh(xy.data(), num_points, cons, (int)constraints.size(), options, mesh, &stats) ) {
        return false;
    }

    triangles.clear();
    triangles.reserve(mesh.triangles.size());
    for ( const int3& t : mesh.triangles ) {
        const int a = mesh.origIndex[t.x];
        const int b = mesh.origIndex[t.y];
        const int c = mesh.origIndex[t.z];
        if ( a < 0 || b < 0 || c < 0 ) continue;  // helper corner (not expected)
        triangles.push_back(make_int3(a, b, c));
    }
    return !triangles.empty();
}
