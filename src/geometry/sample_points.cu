// Pattern-mesh sampling: boundary-constrained point generation.
//
// Pipeline (see README.md "Pattern meshing"):
//   1. normalise the domain by the bounding box, so the cell grid is scale free;
//   2. build an inside/outside mask for the closed boundary curves (outer loop
//      plus holes) on the device, using per-row edge buckets and a parity ray
//      cast - no Vulkan / rasteriser dependency;
//   3. place the input boundary points and the edge midpoints on the grid as
//      fixed obstacles;
//   4. stratified sampling: one jittered point per free interior cell, with a
//      cell length equal to the requested resolution (see set_radius);
//   5. repulsion relaxation: neighbours come from a uniform grid holding at most
//      four points per cell; interior points are pushed away from neighbours
//      within `radius` while boundary points stay fixed. The caller issues two
//      passes with decreasing gain (f1/t1, then f2/t2);
//   6. validate the interior points with the exact inside test, triangulate with
//      all boundary and hole edges as constraints, and keep the triangles whose
//      centroid is inside.
//
// The inside test is orientation independent and honours `is_holes`: a point is
// inside when it is inside the outer loop (curve 0) and outside every loop
// flagged as a hole. Points exactly on a boundary count as inside.

#include <cuda_runtime.h>

#include "common/cuda_utils.h"
#include "common/atomic_utils.cuh"
#include "common/device.h"
#include "common/vec_math.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "sampler.h"
#include "lbvh_2d.h"

// Constrained Delaunay backends.
std::vector<int3> delaunay_2d_cuda_type_impl(std::vector<float2>& pointVecIn,
    std::vector<int2>& constraintVecIn);
bool gcdt_2d_impl(const std::vector<float2>& points, const std::vector<int2>& constraints,
    std::vector<int3>& triangles);

// Env-gated phase profiler. It only records CUDA events between phases and
// synchronises once at the end, so the deltas include both host-side work and
// device work attributed to the following phase.
namespace {

bool sample_profile_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("QYDP_SAMPLE_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

class PhaseProfiler {
public:
    PhaseProfiler() : enabled_(sample_profile_enabled()) {
        if ( !enabled_ ) return;
        events_.resize(24, nullptr);
        for ( auto& event : events_ ) cudaEventCreate(&event);
    }

    ~PhaseProfiler() {
        for ( auto event : events_ ) {
            if ( event ) cudaEventDestroy(event);
        }
    }

    void mark(const char* name) {
        if ( !enabled_ || count_ >= (int)events_.size() ) return;
        cudaEventRecord(events_[count_], 0);
        names_.push_back(name);
        ++count_;
    }

    void finish(const char* backend, int points, int triangles) const {
        if ( !enabled_ || count_ < 2 ) return;
        cudaEventSynchronize(events_[count_ - 1]);
        std::string line = "[QYDP_SAMPLE_PROFILE] backend=";
        line += backend;
        line += " points=" + std::to_string(points) + " tris=" + std::to_string(triangles);
        float total = 0.0f;
        for ( int i = 1; i < count_; ++i ) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, events_[i - 1], events_[i]);
            total += ms;
            line += " ";
            line += names_[i];
            line += "=" + std::to_string(ms);
        }
        line += " event_total=" + std::to_string(total);
        std::cout << line << std::endl;
    }

private:
    bool enabled_ = false;
    std::vector<cudaEvent_t> events_;
    std::vector<const char*> names_;
    int count_ = 0;
};

}  // namespace

static __device__ __forceinline__ float2 normalize_safe_f2(const float2& a) {
    float l = sqrtf(dot(a, a));
    if ( l < 1e-12f ) return make_float2(0.0f, 0.0f);
    return a * (1.0f / l);
}

// ==========================================
// Inside/outside mask
// ==========================================

struct MaskView {
    const float2* edges;        // two vertices per loop edge
    const unsigned char* kind;  // 0 = outer loop, 1 = hole loop
    const int* rowOffsets;      // fineRows + 1
    const int* rowEdges;        // edge ids, grouped by fine row
    float fineLength;
    int fineRows;
};

static __device__ __forceinline__ bool edgeCrossesRight(float px, float py, float2 a, float2 b) {
    // Even-odd crossing of the ray (px,py) -> +x; edges are half open in y so a
    // shared endpoint is counted once.
    bool aAbove = a.y <= py;
    bool bAbove = b.y <= py;
    if ( aAbove == bAbove ) return false;
    float t = (py - a.y) / (b.y - a.y);
    return (a.x + t * (b.x - a.x)) > px;
}

static __device__ __forceinline__ bool insideDomain(const MaskView& mask, float2 p) {
    int row = (int)floorf(p.y / mask.fineLength);
    row = max(0, min(mask.fineRows - 1, row));

    int outer = 0;
    int holes = 0;
    for ( int k = mask.rowOffsets[row]; k < mask.rowOffsets[row + 1]; ++k ) {
        int e = mask.rowEdges[k];
        if ( edgeCrossesRight(p.x, p.y, mask.edges[2 * e], mask.edges[2 * e + 1]) ) {
            const unsigned char kind = mask.kind[e];
            if ( kind == 0 ) outer ^= 1;
            else if ( kind == 1 ) holes ^= 1;
            // kind 2 (internal non-hole line) is margin-only.
        }
    }
    return outer == 1 && holes == 0;
}

static __device__ __forceinline__ float distSqToSegment(float2 p, float2 a, float2 b) {
    float2 ab = b - a;
    float denom = dot(ab, ab);
    float t = denom > 1e-20f ? fminf(fmaxf(dot(p - a, ab) / denom, 0.0f), 1.0f) : 0.0f;
    float2 q = a + t * ab;
    return dot(p - q, p - q);
}

// Inside test with a clearance from the constraint loops. Interior points that
// sit on (or a few float ULPs away from) a straight boundary edge form
// collinear sliver triangles; requiring a margin keeps the boundary ring
// regular and lets the area filter stay a pure numerical safety net.
static __device__ __forceinline__ bool insideDomainWithMargin(
    const MaskView& mask, float2 p, float margin
) {
    if ( !insideDomain(mask, p) ) return false;

    int row = (int)floorf(p.y / mask.fineLength);
    row = max(0, min(mask.fineRows - 1, row));
    // The row buckets must cover every edge that can pass within `margin` of
    // the point, so widen the window with the requested clearance.
    int span = 1 + (int)ceilf(margin / mask.fineLength);
    int r0 = max(0, row - span);
    int r1 = min(mask.fineRows - 1, row + span);
    float margin2 = margin * margin;
    for ( int k = mask.rowOffsets[r0]; k < mask.rowOffsets[r1 + 1]; ++k ) {
        int e = mask.rowEdges[k];
        if ( distSqToSegment(p, mask.edges[2 * e], mask.edges[2 * e + 1]) < margin2 ) return false;
    }
    return true;
}

static __device__ __forceinline__ unsigned int hashCell(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// ==========================================
// Kernels
// ==========================================

static __device__ int2 grid_index(float2 p, float one_grid_length) {
    // floor() cells: cell (i,j) owns [i*L, (i+1)*L) with centre (i+0.5)*L, which
    // matches k_generate_interior and the inside test.
    return make_int2((int)floorf(p.x / one_grid_length), (int)floorf(p.y / one_grid_length));
}

static __device__ bool try_test(int2 index, unsigned char* grid_status, int max_size) {
    int idx = index.y * max_size + index.x;
    unsigned char old = atomicOr(&grid_status[idx], (unsigned char)(1 << 1));
    return old == unsigned char(0);
}

static __global__ void k_count_row_edges(const float2* loop_edges, int num_loop_edges,
    int* row_counts, float fine_length, int fine_rows) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= num_loop_edges ) return;

    float2 a = loop_edges[2 * e];
    float2 b = loop_edges[2 * e + 1];
    float lo = fminf(a.y, b.y);
    float hi = fmaxf(a.y, b.y);
    int r0 = max(0, min(fine_rows - 1, (int)floorf(lo / fine_length)));
    int r1 = max(0, min(fine_rows - 1, (int)floorf(hi / fine_length)));
    for ( int r = r0; r <= r1; ++r ) atomicAdd(&row_counts[r], 1);
}

static __global__ void k_scatter_row_edges(const float2* loop_edges, int num_loop_edges,
    int* row_offsets, int* row_edges, int* row_cursor, float fine_length, int fine_rows) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= num_loop_edges ) return;

    float2 a = loop_edges[2 * e];
    float2 b = loop_edges[2 * e + 1];
    float lo = fminf(a.y, b.y);
    float hi = fmaxf(a.y, b.y);
    int r0 = max(0, min(fine_rows - 1, (int)floorf(lo / fine_length)));
    int r1 = max(0, min(fine_rows - 1, (int)floorf(hi / fine_length)));
    for ( int r = r0; r <= r1; ++r ) {
        int slot = atomicAdd(&row_cursor[r], 1);
        row_edges[slot] = e;
    }
}

static __global__ void k_build_mask(unsigned char* grid_status, int grid_size, int max_size,
    float one_grid_length, MaskView mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x >= grid_size || y >= grid_size ) return;

    float2 center = make_float2(((float)x + 0.5f) * one_grid_length,
        ((float)y + 0.5f) * one_grid_length);
    if ( insideDomain(mask, center) ) {
        int idx = y * max_size + x;
        atomicOr(&grid_status[idx], (unsigned char)1);  // bit0: inside the domain
    }
}

static __global__ void k_place_input_points(
    const float2* points,
    int num_points,
    float2* final_pts,
    unsigned char* grid_status,
    int* grid_point,
    int* d_nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_points ) return;

    float2 point = points[i];
    int2 index = grid_index(point, p.one_grid_length);
    if ( index.x < 0 || index.x >= p.grid_size || index.y < 0 || index.y >= p.grid_size ) return;

    int idx_flat = index.y * p.max_size + index.x;
    if ( (grid_status[idx_flat] & 2) == 0 ) {  // not occupied by another point
        if ( try_test(index, grid_status, p.max_size) ) {
            int j = atomicAdd(d_nb_points, 1);
            final_pts[j] = point;
            grid_point[idx_flat] = i;
        }
    }
}

static __global__ void k_place_edge_midpoints(
    const float2* points,
    const int2* edge_indices,
    int num_edges,
    float2* final_pts,
    unsigned char* grid_status,
    int* grid_point,
    int* d_nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;

    int2 edge = edge_indices[i];
    float2 mid_point = (points[edge.x] + points[edge.y]) * 0.5f;
    int2 mid_index = grid_index(mid_point, p.one_grid_length);
    if ( mid_index.x < 0 || mid_index.x >= p.grid_size || mid_index.y < 0 || mid_index.y >= p.grid_size ) return;

    int mid_idx_flat = mid_index.y * p.max_size + mid_index.x;
    if ( (grid_status[mid_idx_flat] & 2) == 0 ) {
        if ( try_test(mid_index, grid_status, p.max_size) ) {
            int k = atomicAdd(d_nb_points, 1);
            final_pts[k] = mid_point;
            grid_point[mid_idx_flat] = edge.x;
        }
    }
}

static __global__ void k_generate_interior(
    unsigned char* grid_status,
    float2* final_pts,
    int* d_nb_points,
    Params p,
    MaskView mask,
    float margin
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x >= p.grid_size || y >= p.grid_size ) return;

    int idx = y * p.max_size + x;
    if ( (grid_status[idx] & 3) != 1 ) return;  // inside and not occupied

    int cell = y * p.grid_size + x;
    float2 center = make_float2(((float)x + 0.5f) * p.one_grid_length,
        ((float)y + 0.5f) * p.one_grid_length);

    float2 point = center;
    for ( int attempt = 0; attempt < 4; ++attempt ) {
        unsigned int h = hashCell((unsigned int)cell * 0x9E3779B9u + (unsigned int)attempt * 0x85EBCA6Bu);
        float2 offset = make_float2((float)(h & 0xFFFFu) * (1.0f / 65536.0f) - 0.5f,
            (float)((h >> 16) & 0xFFFFu) * (1.0f / 65536.0f) - 0.5f);
        float2 candidate = center + offset * p.one_grid_length;
        if ( insideDomainWithMargin(mask, candidate, margin) ) {
            point = candidate;
            break;
        }
    }

    int j = atomicAdd(d_nb_points, 1);
    final_pts[j] = point;
}

static __global__ void k_reset_grid_multi(unsigned char* grid_multi_point_size, Params p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x < p.max_size && y < p.max_size ) {
        grid_multi_point_size[y * p.max_size + x] = 0;
    }
}

constexpr int max_grid_particles_size = 4;

static __global__ void k_build_grid(
    float2* final_pts,
    unsigned char* grid_multi_point_size,
    int* grid_multi_point,
    int nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_points ) return;

    float2 point = final_pts[i];
    int2 index = grid_index(point, p.one_grid_length);
    if ( index.x < 0 || index.x >= p.max_size || index.y < 0 || index.y >= p.max_size ) return;

    int idx_flat = index.y * p.max_size + index.x;
    unsigned char id = atomicAdd(&grid_multi_point_size[idx_flat], 1);
    if ( id < max_grid_particles_size ) {
        grid_multi_point[idx_flat * 4 + id] = i;
    }
}

static __global__ void k_compute_repulsion(
    float2* final_pts,
    unsigned char* grid_multi_point_size,
    int* grid_multi_point,
    float2* force,
    int nb_boundary_points,
    int nb_points,
    Params p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_i = i + nb_boundary_points;
    if ( actual_i >= nb_points ) return;

    float2 point = final_pts[actual_i];
    int2 index = grid_index(point, p.one_grid_length);
    // Repulsion reach is one sampling cell. Tying it to `radius` instead ties
    // it to 0.707 cells once the cell length is the requested edge length,
    // which leaves jittered neighbours in [0.707, 1] cells untouched and turns
    // the relaxed grid into a Poisson-like distribution.
    float cutoff2 = p.one_grid_length * p.one_grid_length;
    float2 d = make_float2(0.0f, 0.0f);

    int x = max(0, index.x);
    int y = max(0, index.y);
    for ( int nx = max(0, x - 1); nx < min(p.max_size, x + 2); ++nx ) {
        for ( int ny = max(0, y - 1); ny < min(p.max_size, y + 2); ++ny ) {
            int n_idx = ny * p.max_size + nx;
            int count = min((int)grid_multi_point_size[n_idx], max_grid_particles_size);
            for ( int k = 0; k < count; ++k ) {
                int p2_i = grid_multi_point[n_idx * 4 + k];
                if ( p2_i == actual_i ) continue;

                float2 l = point - final_pts[p2_i];
                float l2 = dot(l, l);
                if ( l2 < cutoff2 + 1e-6f && l2 > 1e-10f ) {
                    d = d + (l * (1.0f / (sqrtf(l2) * l2)));
                }
            }
        }
    }
    force[actual_i] = d;
}

static __global__ void k_apply_force(
    float2* final_pts,
    float2* force,
    int nb_boundary_points,
    int nb_points,
    float factor,
    Params p,
    MaskView mask,
    float margin
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_i = i + nb_boundary_points;
    if ( actual_i >= nb_points ) return;

    float2 d = force[actual_i];
    float maxd = p.radius * factor;
    float d_len_sq = dot(d, d);
    if ( d_len_sq > maxd * maxd ) d = normalize_safe_f2(d) * maxd;

    float2 moved = final_pts[actual_i] + d;
    // Never let a relaxed point leave the domain: staying put is better than
    // being dropped later, which would leave a hole in the sampling.
    if ( insideDomainWithMargin(mask, moved, margin) ) final_pts[actual_i] = moved;
}

static __global__ void k_validate(
    float2* final_pts,
    unsigned char* valid_status,
    int nb_boundary_points,
    int nb_points,
    MaskView mask,
    float margin
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_i = i + nb_boundary_points;
    if ( actual_i >= nb_points ) return;

    valid_status[actual_i] = insideDomainWithMargin(mask, final_pts[actual_i], margin) ? 1 : 0;
}

// Keeps triangles whose centroid is inside the polygon. The triangulation
// contains every boundary and hole edge as a constraint, so a triangle lies
// entirely inside or entirely outside and the centroid test is exact - the old
// cell-resolution mask test (which leaked triangles at concave boundaries) is
// gone, together with its "any vertex is an interior point" bypass.
static __global__ void k_validate_triangles(
    unsigned char* valid_status,
    const int3* triangles,
    const float2* pts,
    int nb_tris,
    MaskView mask
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_tris ) return;

    int3 vs = triangles[i];
    float2 p0 = pts[vs.x], p1 = pts[vs.y], p2 = pts[vs.z];
    float2 e0 = p1 - p0, e1 = p2 - p0, e2 = p2 - p1;
    float max_l2 = fmaxf(dot(e0, e0), fmaxf(dot(e1, e1), dot(e2, e2)));
    // Drop numerical slivers. In normalised coordinates a collinear triple on
    // a straight constraint edge sits at float-noise level (1e-10..1e-8);
    // legal triangles stay orders of magnitude above that floor, especially
    // now that interior points keep a boundary margin.
    if ( max_l2 <= 1e-30f || fabsf(cross(e0, e1)) < 1e-8f ) {
        valid_status[i] = 0;
        return;
    }
    float2 center = (p0 + p1 + p2) / 3.0f;
    valid_status[i] = insideDomain(mask, center) ? 1 : 0;
}

// ==========================================
// Host controller
// ==========================================

Sampler::Sampler()
    : d_nb_points(nullptr), d_grid_status(nullptr), d_grid_point(nullptr), d_final(nullptr),
      d_grid_multi_point(nullptr), d_grid_multi_point_size(nullptr), d_force(nullptr),
      d_valid_status(nullptr), d_input_points(nullptr), d_edge_indices(nullptr),
      d_loop_edges(nullptr), d_row_counts(nullptr), d_row_offsets(nullptr), d_row_edges(nullptr),
      d_row_cursor(nullptr) {
    cudaMalloc(&d_nb_points, sizeof(int));
}

Sampler::~Sampler() {
    if ( cudaFree(nullptr) == cudaErrorCudartUnloading ) return;
    cudaFree(d_grid_status);
    cudaFree(d_grid_point);
    cudaFree(d_final);
    cudaFree(d_grid_multi_point);
    cudaFree(d_grid_multi_point_size);
    cudaFree(d_force);
    cudaFree(d_valid_status);
    cudaFree(d_nb_points);
    if ( d_input_points ) cudaFree(d_input_points);
    if ( d_edge_indices ) cudaFree(d_edge_indices);
    if ( d_loop_edges ) cudaFree(d_loop_edges);
    if ( d_row_counts ) cudaFree(d_row_counts);
    if ( d_row_offsets ) cudaFree(d_row_offsets);
    if ( d_row_edges ) cudaFree(d_row_edges);
    if ( d_row_cursor ) cudaFree(d_row_cursor);
}

void Sampler::set_radius(float _radius) {
    // The requested radius is the target triangle edge length: one sampling cell
    // is one edge long, so cells of length L produce triangles with edges ~L.
    params.radius = _radius;
    params.one_grid_length = _radius;
    params.grid_size = (int)ceilf(1.0f / params.one_grid_length) + 1;
    params.max_size = ((params.grid_size + 31) / 32) * 32;
    params.n = params.max_size * params.max_size;

    fine_length = params.one_grid_length / (float)fine_rows_per_cell;
    fine_rows = params.grid_size * fine_rows_per_cell + 1;

    if ( d_grid_status ) cudaFree(d_grid_status);
    if ( d_grid_point ) cudaFree(d_grid_point);
    if ( d_grid_multi_point ) cudaFree(d_grid_multi_point);
    if ( d_grid_multi_point_size ) cudaFree(d_grid_multi_point_size);
    if ( d_row_counts ) cudaFree(d_row_counts);
    if ( d_row_offsets ) cudaFree(d_row_offsets);
    if ( d_row_cursor ) cudaFree(d_row_cursor);

    cudaMalloc(&d_grid_status, params.max_size * params.max_size * sizeof(unsigned char));
    cudaMalloc(&d_grid_point, params.max_size * params.max_size * sizeof(int));
    cudaMalloc(&d_grid_multi_point, params.max_size * params.max_size * 4 * sizeof(int));
    cudaMalloc(&d_grid_multi_point_size, params.max_size * params.max_size * sizeof(unsigned char));
    cudaMalloc(&d_row_counts, (fine_rows + 1) * sizeof(int));
    cudaMalloc(&d_row_offsets, (fine_rows + 1) * sizeof(int));
    cudaMalloc(&d_row_cursor, (fine_rows + 1) * sizeof(int));
    current_radius_scaled = _radius;
}

void Sampler::sample(
    std::vector<float2>& output_points,
    std::vector<int3>& output_tris,
    const std::vector<float2>& all_points,
    const std::vector<int2>& edge_indices,
    const std::vector<int>& curve_sizes,
    const std::vector<bool>& is_holes,
    float raw_radius,
    float f1, int t1, float f2, int t2,
    int triangulator,
    float boundary_margin_cells
) {
    if ( all_points.empty() || edge_indices.empty() ) return;

    PhaseProfiler profiler;
    profiler.mark("start");

    int num_input_points = (int)all_points.size();
    int num_edges = (int)edge_indices.size();

    // ---------------------------------------------------------
    // 1. Bounding box and normalisation
    // ---------------------------------------------------------
    float x_min = FLT_MAX, y_min = FLT_MAX;
    float x_max = -FLT_MAX, y_max = -FLT_MAX;
    for ( const auto& p : all_points ) {
        x_min = fminf(x_min, p.x);
        y_min = fminf(y_min, p.y);
        x_max = fmaxf(x_max, p.x);
        y_max = fmaxf(y_max, p.y);
    }
    float scale = 1.0f / (fmaxf(x_max - x_min, y_max - y_min) + raw_radius);
    float2 offset = make_float2(x_min, y_min);
    float radius_scaled = raw_radius * scale;

    set_radius(radius_scaled);

    std::vector<float2> points_normalized(num_input_points);
    for ( int i = 0; i < num_input_points; ++i ) {
        points_normalized[i].x = (all_points[i].x - offset.x + raw_radius * 0.5f) * scale;
        points_normalized[i].y = (all_points[i].y - offset.y + raw_radius * 0.5f) * scale;
    }
    profiler.mark("setup");

    // ---------------------------------------------------------
    // 2. Constraint edges and their row buckets. Outer and hole loops drive
    //    the inside/outside parity; internal lines (darts, fold lines) only
    //    participate in the boundary-clearance test.
    // ---------------------------------------------------------
    std::vector<float2> loop_edges;
    std::vector<unsigned char> loop_kind;
    loop_edges.reserve((size_t)num_edges * 2);
    loop_kind.reserve((size_t)num_edges);
    int edge_offset = 0;
    for ( int c = 0; c < (int)curve_sizes.size(); ++c ) {
        const unsigned char kind =
            (c == 0) ? 0 : ((c < (int)is_holes.size() && is_holes[c]) ? 1 : 2);
        for ( int e = 0; e < curve_sizes[c]; ++e ) {
            int2 edge = edge_indices[edge_offset + e];
            loop_edges.push_back(points_normalized[edge.x]);
            loop_edges.push_back(points_normalized[edge.y]);
            loop_kind.push_back(kind);
        }
        edge_offset += curve_sizes[c];
    }
    int num_loop_edges = (int)loop_kind.size();

    if ( d_loop_edges ) { cudaFree(d_loop_edges); d_loop_edges = nullptr; }
    if ( d_row_edges ) { cudaFree(d_row_edges); d_row_edges = nullptr; }
    unsigned char* d_loop_kind = nullptr;
    if ( num_loop_edges > 0 ) {
        cudaMalloc(&d_loop_edges, loop_edges.size() * sizeof(float2));
        cudaMemcpy(d_loop_edges, loop_edges.data(), loop_edges.size() * sizeof(float2), cudaMemcpyHostToDevice);

        cudaMemset(d_row_counts, 0, (fine_rows + 1) * sizeof(int));
        int blocks = (num_loop_edges + 255) / 256;
        k_count_row_edges << <blocks, 256 >> > (d_loop_edges, num_loop_edges, d_row_counts,
            fine_length, fine_rows);

        // Prefix sum turns the per-row edge counts into bucket offsets.
        thrust::exclusive_scan(thrust::device, d_row_counts, d_row_counts + fine_rows + 1, d_row_offsets);

        int total_entries = 0;
        // The total lives at the end of the prefix sum (row_counts[fine_rows]
        // is never written by the counting kernel).
        cudaMemcpy(&total_entries, d_row_offsets + fine_rows, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_row_cursor, d_row_offsets, (fine_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMalloc(&d_row_edges, (size_t)std::max(total_entries, 1) * sizeof(int));
        k_scatter_row_edges << <blocks, 256 >> > (d_loop_edges, num_loop_edges, d_row_offsets,
            d_row_edges, d_row_cursor, fine_length, fine_rows);
    }
    profiler.mark("mask_buckets");

    MaskView mask;
    mask.edges = d_loop_edges;
    mask.rowOffsets = d_row_offsets;
    mask.rowEdges = d_row_edges;
    mask.fineLength = fine_length;
    mask.fineRows = fine_rows;

    // The loop kind array is tiny; keep it on the device for the kernels.
    if ( num_loop_edges > 0 ) {
        cudaMalloc(&d_loop_kind, num_loop_edges * sizeof(unsigned char));
        cudaMemcpy(d_loop_kind, loop_kind.data(), num_loop_edges * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }
    mask.kind = d_loop_kind;

    // ---------------------------------------------------------
    // 3. Per-input device buffers
    // ---------------------------------------------------------
    int max_points = params.grid_size * params.grid_size + num_input_points + num_edges + 8;
    if ( d_final ) cudaFree(d_final);
    if ( d_force ) cudaFree(d_force);
    if ( d_valid_status ) cudaFree(d_valid_status);
    cudaMalloc(&d_final, (size_t)max_points * sizeof(float2));
    cudaMalloc(&d_force, (size_t)max_points * sizeof(float2));
    cudaMalloc(&d_valid_status, (size_t)max_points * sizeof(unsigned char));

    if ( d_input_points ) cudaFree(d_input_points);
    if ( d_edge_indices ) cudaFree(d_edge_indices);
    cudaMalloc(&d_input_points, num_input_points * sizeof(float2));
    cudaMalloc(&d_edge_indices, num_edges * sizeof(int2));
    cudaMemcpy(d_input_points, points_normalized.data(), num_input_points * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_indices, edge_indices.data(), num_edges * sizeof(int2), cudaMemcpyHostToDevice);

    cudaMemset(d_grid_status, 0, params.max_size * params.max_size * sizeof(unsigned char));
    cudaMemset(d_nb_points, 0, sizeof(int));

    // ---------------------------------------------------------
    // 4. Inside mask (exact, orientation independent)
    // ---------------------------------------------------------
    {
        dim3 dimBlock(16, 16);
        dim3 dimGrid((params.grid_size + dimBlock.x - 1) / dimBlock.x,
            (params.grid_size + dimBlock.y - 1) / dimBlock.y);
        k_build_mask << <dimGrid, dimBlock >> > (d_grid_status, params.grid_size, params.max_size,
            params.one_grid_length, mask);
    }
    profiler.mark("mask_build");

    // ---------------------------------------------------------
    // 5. Boundary points and edge midpoints as fixed obstacles
    // ---------------------------------------------------------
    int block_size = 256;
    k_place_input_points << <(num_input_points + block_size - 1) / block_size, block_size >> > (
        d_input_points, num_input_points, d_final, d_grid_status, d_grid_point, d_nb_points, params);
    k_place_edge_midpoints << <(num_edges + block_size - 1) / block_size, block_size >> > (
        d_input_points, d_edge_indices, num_edges, d_final, d_grid_status, d_grid_point, d_nb_points, params);

    int nb_boundary_points = 0;
    cudaMemcpy(&nb_boundary_points, d_nb_points, sizeof(int), cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // 6. Interior points: one jittered point per free interior cell
    // ---------------------------------------------------------
    // Clearance from every constraint edge, in units of the requested edge
    // length. 0.8 keeps the first interior ring far enough from a legal
    // (boundary spacing <= radius) constraint edge that boundary triangles
    // stay well shaped; 0.1 still allowed quality ~0.2 seam slivers.
    float boundary_margin = fmaxf(boundary_margin_cells, 0.0f) * params.one_grid_length;
    {
        dim3 dimBlock(16, 16);
        dim3 dimGrid((params.grid_size + dimBlock.x - 1) / dimBlock.x,
            (params.grid_size + dimBlock.y - 1) / dimBlock.y);
        k_generate_interior << <dimGrid, dimBlock >> > (
            d_grid_status, d_final, d_nb_points, params, mask, boundary_margin);
    }
    int h_nb_points = 0;
    cudaMemcpy(&h_nb_points, d_nb_points, sizeof(int), cudaMemcpyDeviceToHost);
    profiler.mark("point_gen");

    // ---------------------------------------------------------
    // 7. Repulsion relaxation
    // ---------------------------------------------------------
    auto run_repulsion = [&](float rep_factor, int times) {
        for ( int t = 0; t < times; ++t ) {
            dim3 dimBlock(16, 16);
            dim3 dimGrid((params.max_size + dimBlock.x - 1) / dimBlock.x,
                (params.max_size + dimBlock.y - 1) / dimBlock.y);
            k_reset_grid_multi << <dimGrid, dimBlock >> > (d_grid_multi_point_size, params);

            int blocks = (h_nb_points + 255) / 256;
            k_build_grid << <blocks, 256 >> > (
                d_final, d_grid_multi_point_size, d_grid_multi_point, h_nb_points, params);

            int inner_count = h_nb_points - nb_boundary_points;
            if ( inner_count > 0 ) {
                blocks = (inner_count + 255) / 256;
                k_compute_repulsion << <blocks, 256 >> > (
                    d_final, d_grid_multi_point_size, d_grid_multi_point, d_force,
                    nb_boundary_points, h_nb_points, params);
                k_apply_force << <blocks, 256 >> > (
                    d_final, d_force, nb_boundary_points, h_nb_points, rep_factor, params, mask,
                    boundary_margin);
            }
        }
    };
    run_repulsion(f1, t1);
    profiler.mark("relax1");
    run_repulsion(f2, t2);
    profiler.mark("relax2");

    // ---------------------------------------------------------
    // 8. Validate interior points (exact inside test)
    // ---------------------------------------------------------
    int inner_count = h_nb_points - nb_boundary_points;
    if ( inner_count > 0 ) {
        int blocks = (inner_count + 255) / 256;
        k_validate << <blocks, 256 >> > (
            d_final, d_valid_status, nb_boundary_points, h_nb_points, mask, boundary_margin);
    }
    cudaDeviceSynchronize();
    profiler.mark("validate");

    // ---------------------------------------------------------
    // 9. Output points: input points first (order preserved), then valid
    //    interior points
    // ---------------------------------------------------------
    std::vector<float2> result_final(h_nb_points);
    std::vector<unsigned char> result_valid(h_nb_points);
    if ( h_nb_points > 0 ) {
        cudaMemcpy(result_final.data(), d_final, h_nb_points * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_valid.data(), d_valid_status, h_nb_points * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    }

    output_points.clear();
    output_points.reserve(num_input_points + inner_count);
    output_points.insert(output_points.end(), points_normalized.begin(), points_normalized.end());
    for ( int i = nb_boundary_points; i < h_nb_points; ++i ) {
        if ( result_valid[i] > 0 ) output_points.push_back(result_final[i]);
    }

    // ---------------------------------------------------------
    // 10. Triangulate with all edges as constraints
    // ---------------------------------------------------------
    std::vector<int2> constraints(num_edges);
    memcpy(constraints.data(), edge_indices.data(), num_edges * sizeof(int2));

    std::vector<int3> tris;
    if ( triangulator == 1 ) {
        if ( !gcdt_2d_impl(output_points, constraints, tris) ) {
            std::cerr << "[Qianyi] gCDT produced no valid mesh; falling back to gDel2D" << std::endl;
            tris = delaunay_2d_cuda_type_impl(output_points, constraints);
        }
    }
    else {
        tris = delaunay_2d_cuda_type_impl(output_points, constraints);
    }
    profiler.mark("triangulate");

    // ---------------------------------------------------------
    // 11. Keep inside triangles
    // ---------------------------------------------------------
    thrust::device_vector<int3> d_tris(tris.begin(), tris.end());
    thrust::device_vector<unsigned char> d_valid_status_tris(tris.size());
    thrust::device_vector<float2> d_pts(output_points.begin(), output_points.end());

    if ( !tris.empty() ) {
        int blocks = ((int)tris.size() + 255) / 256;
        k_validate_triangles << <blocks, 256 >> > (
            d_valid_status_tris.data().get(), d_tris.data().get(), d_pts.data().get(),
            (int)tris.size(), mask);
    }

    // ---------------------------------------------------------
    // 12. Un-normalise output points
    // ---------------------------------------------------------
    for ( auto& output_point : output_points ) {
        output_point.x = output_point.x / scale - raw_radius * 0.5f + offset.x;
        output_point.y = output_point.y / scale - raw_radius * 0.5f + offset.y;
    }

    thrust::host_vector<unsigned char> h_valid_status_tris(d_valid_status_tris);
    output_tris.clear();
    output_tris.reserve(tris.size());
    for ( int i = 0; i < (int)tris.size(); ++i ) {
        if ( h_valid_status_tris[i] == 1 ) output_tris.push_back(tris[i]);
    }
    profiler.mark("filter");
    profiler.finish(triangulator == 1 ? "gcdt" : "gdel2d",
        (int)output_points.size(), (int)output_tris.size());

    if ( d_loop_kind ) cudaFree(d_loop_kind);
}

// ==========================================
// Python-facing entry point
// ==========================================

void sample_points_impl(std::vector<float>& boundary, std::vector<int>& edge_indices_flat,
    std::vector<int>& curve_sizes, std::vector<int>& is_holes_int,
    float radius, std::vector<float>& output_points,
    std::vector<int>& output_tris, int triangulator,
    float relax_gain1, int relax_iters1, float relax_gain2, int relax_iters2,
    float boundary_margin_cells
) {
    init_device();
    Sampler sampler{};

    int num_points = (int)boundary.size() / 2;
    std::vector<float2> all_points(num_points);
    std::memcpy(all_points.data(), boundary.data(), sizeof(float) * boundary.size());

    int num_edges = (int)edge_indices_flat.size() / 2;
    std::vector<int2> edges(num_edges);
    std::memcpy(edges.data(), edge_indices_flat.data(), sizeof(int) * edge_indices_flat.size());

    std::vector<bool> is_holes(is_holes_int.size());
    for ( int i = 0; i < (int)is_holes_int.size(); ++i ) is_holes[i] = (is_holes_int[i] != 0);

    std::vector<float2> points;
    std::vector<int3> tris;
    sampler.sample(points, tris, all_points, edges, curve_sizes, is_holes, radius,
        relax_gain1, relax_iters1, relax_gain2, relax_iters2, triangulator,
        boundary_margin_cells);

    output_points.resize(points.size() * 2);
    std::memcpy(output_points.data(), points.data(), sizeof(float) * output_points.size());
    output_tris.resize(tris.size() * 3);
    std::memcpy(output_tris.data(), tris.data(), sizeof(int) * output_tris.size());
}

struct MapPointsFunctor {
    const float* bounds_source;
    const float* bounds_target;
    __device__ void operator()(float2& point) {
        float2 src_min = { bounds_source[0], bounds_source[1] };
        float2 src_max = { bounds_source[2], bounds_source[3] };
        float2 tgt_min = { bounds_target[0], bounds_target[1] };
        float2 tgt_max = { bounds_target[2], bounds_target[3] };

        point.x = ((point.x - src_min.x) / (src_max.x - src_min.x)) * (tgt_max.x - tgt_min.x) + tgt_min.x;
        point.y = ((point.y - src_min.y) / (src_max.y - src_min.y)) * (tgt_max.y - tgt_min.y) + tgt_min.y;
    }
};

static void find_points_locations(
    thrust::device_vector<float2>& query_pts,
    const thrust::device_vector<float2>& vertices,
    const thrust::device_vector<int3>& faces,
    thrust::device_vector<lbvh2d::LocationResult>& results,
    const bool scale_to_fix = false) {
    unsigned int n_faces = faces.size();
    unsigned int n_queries = query_pts.size();

    if ( n_faces == 0 || n_queries == 0 ) return;

    lbvh2d::initialize(faces.size());
    BVH2D bvh;
    lbvh2d::build_face_bvh(vertices, faces, bvh);
    if ( scale_to_fix ) {
        float bounds[4];
        lbvh2d::calc_bounds(query_pts, bounds);
        thrust::device_vector<float> bounds_source(4);
        thrust::copy(bounds, bounds + 4, bounds_source.begin());

        lbvh2d::calc_bounds(vertices, bounds);
        thrust::device_vector<float> bounds_target(4);
        thrust::copy(bounds, bounds + 4, bounds_target.begin());
        MapPointsFunctor mapper = {
            thrust::raw_pointer_cast(bounds_source.data()),
            thrust::raw_pointer_cast(bounds_target.data())
        };
        thrust::for_each(query_pts.begin(), query_pts.end(), mapper);
    }
    results.resize(n_queries);
    query_location_kernel<<<(n_queries + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(query_pts.data()),
        n_queries,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()),
        bvh.root_idx,
        thrust::raw_pointer_cast(vertices.data()),
        thrust::raw_pointer_cast(faces.data()),
        thrust::raw_pointer_cast(results.data())
        );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void find_map_weight_impl(
    const std::vector<float>& map_points,
    const std::vector<int>& map_tris,
    const std::vector<float>& query_points,
    std::vector<int>& res_index, std::vector<float>& res_weight,
    bool map_bounds
) {
    thrust::device_vector<float2> d_points(query_points.size() / 2);
    cudaMemcpy(d_points.data().get(), query_points.data(), d_points.size() * sizeof(float2), cudaMemcpyHostToDevice);
    thrust::device_vector<float2> d_vertices(map_points.size() / 2);
    cudaMemcpy(d_vertices.data().get(), map_points.data(), d_vertices.size() * sizeof(float2), cudaMemcpyHostToDevice);
    thrust::device_vector<int3> d_faces(map_tris.size() / 3);
    cudaMemcpy(d_faces.data().get(), map_tris.data(), d_faces.size() * sizeof(int3), cudaMemcpyHostToDevice);
    thrust::device_vector<lbvh2d::LocationResult> d_results;
    find_points_locations(d_points, d_vertices, d_faces, d_results, map_bounds);
    thrust::host_vector<lbvh2d::LocationResult> h_results = d_results;
    res_index.resize(h_results.size());
    res_weight.resize(h_results.size() * 3);
    for ( int i = 0; i < h_results.size(); ++i ) {
        auto res = h_results[i];
        res_index[i] = res.prim_idx;
        res_weight[i * 3] = res.u;
        res_weight[i * 3 + 1] = res.v;
        res_weight[i * 3 + 2] = res.w;
    }
}
