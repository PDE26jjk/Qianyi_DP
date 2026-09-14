#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "common/atomic_utils.cuh"


constexpr char stitch_status_running = (char)(0);
constexpr char stitch_status_done = (char)(1 << 1);
constexpr char stitch_status_suspend = (char)(1 << 2);
constexpr char stitch_status_torn = (char)(1 << 3);

void Geometry::init_sewing() {
    // Option B: vertex_proxy stays identity forever; only consumers that
    // expect per-vertex self-mapping read it. Stitch status and the
    // closure counter are owned by init_stitch_cluster_buffers, and
    // stitch_sewing by init_bend_structure.
    vertex_proxy.assign(thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(params.nb_all_vertices));
    sewing_done = false;
    init_stitch_cluster_buffers();
}


static __global__ void check_stitch_closure_kernel(
    int* stitches_done_count,
    char* stitches_status,
    const float3* __restrict__ vertices,
    const int2* __restrict__ stitches,
    float close_dist_sq,
    int num_stitches
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_stitches ) return;
    if ( stitches_status[idx] != stitch_status_running ) return;
    auto [v0, v1] = stitches[idx];
    if ( len_sq(vertices[v0] - vertices[v1]) < close_dist_sq ) {
        stitches_status[idx] = stitch_status_done;
        atomicAdd(stitches_done_count, 1);
    }
}

void Geometry::check_sewing() {
    if ( sewing_done ) return;
    const int n = params.nb_all_stitches;
    int done_count = 0;
    if ( n > 0 )
        cudaMemcpy(&done_count, stitches_done_count.data().get(),
            sizeof(int), cudaMemcpyDeviceToHost);
    // No-stitch meshes count as fully closed so gravity stays enabled.
    sewing_done = (done_count >= n);
    float close_dist = max(0.f, get_global_parameter("sewing_close_dist", 1e-2f));
    int block = 256;
    check_stitch_closure_kernel<<<(n + block - 1) / block, block>>>(
        stitches_done_count.data().get(),
        stitches_status.data().get(),
        pos_world.data().get(),
        stitches.data().get(),
        close_dist * close_dist, n);
    int done_count_new = 0;
    cudaMemcpy(&done_count_new, stitches_done_count.data().get(),
        sizeof(int), cudaMemcpyDeviceToHost);
    if ( done_count_new > done_count )
        std::cout << "stitches closed: " << done_count_new << "/" << n << std::endl;

}

// Permanent seam constraint: zero-rest-length spring with the exact
// quadratic Hessian. Active for the whole simulation (a closed seam
// still carries load and provides the tension signal for tearing);
// only torn stitches are skipped.
static __global__ void compute_stitch_constraint(
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    const float3* __restrict__ vertices,
    const char* __restrict__ stitches_status,
    const int2* __restrict__ stitches,
    float k_input,
    float force_cap,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    if ( stitches_status[idx] == stitch_status_torn ) return;
    auto [p0_i, p1_i] = stitches[idx];
    float3 e = vertices[p0_i] - vertices[p1_i];
    float length = norm(e);
    float k = k_input;
    if ( force_cap > 0.f && length > 1e-12f )
        k = min(k, force_cap / length);
    if ( Jx_diag ) {
        // E = 0.5*k*|x0-x1|^2 -> constant exact Hessian blocks.
        Mat3 K = Mat3::identity(k);
        atomicAddMat3(&Jx_diag[p0_i], K);
        atomicAddMat3(&Jx_diag[p1_i], K);
    }
    if ( length > 1e-12f ) {
        float3 force = (e / length) * (length * k);
        atomicAddFloat3(&forces[p0_i], -force);
        atomicAddFloat3(&forces[p1_i], force);
    }
}

void Geometry::accumulate_sewing_force(Mat3* Jx_diag, cudaStream_t stream) {
    const int n = params.nb_all_stitches;
    if ( n <= 0 ) return;
    float sewing_k = max(0.f, get_global_parameter("sewing_k", 1e5f));
    float force_cap = max(0.f, get_global_parameter("sewing_max_force", 1e5f));
    int block = 256;
    compute_stitch_constraint<<<(n + block - 1) / block, block, 0, stream>>>(
        Jx_diag, elastic_forces.data().get(),
        pos_world.data().get(), stitches_status.data().get(),
        stitches.data().get(), sewing_k, force_cap, n);
}


// Compact the stitch list: drop self pairs (fold-chain apex),
// out-of-range pairs and torn stitches. Output order is arbitrary;
// the final tables are canonicalized by sorting on (root, vertex id).
static __global__ void collect_active_stitches_kernel(
    const int2* __restrict__ stitches,
    const char* __restrict__ status,
    int2* __restrict__ active_out,
    int* __restrict__ active_count,
    int nb_all_vertices,
    int num_stitches
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_stitches ) return;
    int2 s = stitches[i];
    if ( s.x == s.y ) return;
    if ( s.x < 0 || s.y < 0 || s.x >= nb_all_vertices || s.y >= nb_all_vertices ) return;
    if ( status && status[i] == stitch_status_torn ) return;
    int k = atomicAdd(active_count, 1);
    active_out[k] = s;
}
// Merge core extracted verbatim from check_sewing_kernel: one-level
// resolve, then two-way atomicMin link.
static __global__ void stitch_hook_kernel(
    int* __restrict__ vertex_proxy,
    const int2* __restrict__ active,
    int m
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= m ) return;
    int2 s = active[i];
    int v0 = min(vertex_proxy[s.x], s.x);
    int v1 = min(vertex_proxy[s.y], s.y);
    if ( v0 == v1 ) return;
    atomicMin(&vertex_proxy[v1], v0);
    atomicMin(&vertex_proxy[v0], v1);
}

static __global__ void stitch_unresolved_kernel(
    const int* __restrict__ vertex_proxy,
    const int2* __restrict__ active,
    int* __restrict__ unresolved,
    int m
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= m ) return;
    int2 s = active[i];
    if ( vertex_proxy[s.x] != vertex_proxy[s.y] ) atomicAdd(unresolved, 1);
}

static __global__ void mark_stitch_endpoints_kernel(
    const int2* __restrict__ active, char* __restrict__ flag, int m
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= m ) return;
    flag[active[i].x] = 1;
    flag[active[i].y] = 1;
}

static __global__ void pack_member_keys_kernel(
    const int* __restrict__ vertex_proxy,
    const int* __restrict__ members,
    unsigned long long* __restrict__ keys,
    int m
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= m ) return;
    unsigned long long root =
        (unsigned long long)(unsigned int)vertex_proxy[members[k]];
    keys[k] = (root << 32) | (unsigned long long)(unsigned int)members[k];
}

static __global__ void assign_stitch_cluster_ids_kernel(
    const int* __restrict__ vertex_proxy,
    const char* __restrict__ endpoint_flag,
    int* __restrict__ cluster_id,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    cluster_id[i] = endpoint_flag[i] ? vertex_proxy[i] : -1;
}
// Full path compression on the label forest; no mask side effects.
static __global__ void compress_cluster_labels_kernel(
    int* __restrict__ label, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int r = i;
    while ( label[r] != r ) r = label[r];
    if ( r != i ) label[i] = r;
}

// flag-marked vertices are appended with an atomic counter. Order is arbitrary; the later
// (root, id) sort canonicalizes it.
static __global__ void compact_flagged_kernel(
    const char* __restrict__ flag,
    int* __restrict__ out,
    int* __restrict__ count,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n || !flag[i] ) return;
    out[atomicAdd(count, 1)] = i;
}

// Unpack the member id from the low half of a sorted (root<<32|id) key.
static __global__ void unpack_members_kernel(
    const unsigned long long* __restrict__ keys,
    int* __restrict__ members,
    int m
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= m ) return;
    members[k] = (int)(keys[k] & 0xFFFFFFFFULL);
}

void Geometry::init_stitch_cluster_buffers() {
    const int n = params.nb_all_vertices;
    const int ns = params.nb_all_stitches;
    const int max_members = 2 * ns;

    stitch_cluster_id.assign(n, -1);
    stitch_cluster_lookup.assign(n, make_int2(0, 0));
    stitch_cluster_members.resize(max_members);
    stitch_cluster_locked.assign(n, 0);

    cluster_scratch_active.resize(ns);
    cluster_scratch_label.resize(n);
    cluster_scratch_endpoint.resize(n);
    cluster_scratch_keys_a.resize(max_members);
    cluster_scratch_keys_b.resize(max_members);
    cluster_scratch_counts.assign(2, 0);

    // Sort temp queried once for the worst-case key count; the requirement
    // is monotonic in n, so smaller rebuilds always fit.
    unsigned long long* ka = cluster_scratch_keys_a.data().get();
    unsigned long long* kb = cluster_scratch_keys_b.data().get();
    cub::DeviceRadixSort::SortKeys(nullptr, cluster_sort_temp_bytes,
        ka, kb, max_members);
    cluster_scratch_sort_temp.resize(cluster_sort_temp_bytes);

    stitches_done_count.assign((size_t)1, 0);
    stitches_status.assign(ns, stitch_status_running); // torn filter reads this
}
void Geometry::build_stitch_clusters() {
    const int n = params.nb_all_vertices;
    const int ns = params.nb_all_stitches;
    constexpr int block = 256;

    // Stale tables must not outlive a rebuild (e.g. tearing everything
    // open): reset first, early-out leaves a clean empty state.
    cudaMemsetAsync(stitch_cluster_id.data().get(), -1, n * sizeof(int));
    cudaMemsetAsync(stitch_cluster_lookup.data().get(), 0, n * sizeof(int2));
    if ( ns == 0 ) return;

    // 1. Active stitches (drop self pairs / torn).
    int* d_counts = cluster_scratch_counts.data().get();
    cudaMemsetAsync(d_counts, 0, sizeof(int));
    collect_active_stitches_kernel<<<(ns + block - 1) / block, block>>>(
        stitches.data().get(), stitches_status.data().get(),
        cluster_scratch_active.data().get(), d_counts, n, ns);
    int m = 0;
    cudaMemcpyAsync(&m, d_counts, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    if ( m == 0 ) return;

    // 2. Union-find labels in scratch: hook + full compression until the
    // forest is flat and every edge is resolved (2-3 rounds in practice;
    // guards against the concurrent atomicMin lost-link race).
    int* label = cluster_scratch_label.data().get();
    thrust::sequence(thrust::cuda::par_nosync,
        cluster_scratch_label.begin(), cluster_scratch_label.end());
    for ( int round = 0; round < 64; ++round ) {
        stitch_hook_kernel<<<(m + block - 1) / block, block>>>(
            label, cluster_scratch_active.data().get(), m);
        compress_cluster_labels_kernel<<<(n + block - 1) / block, block>>>(
            label, n);
        cudaMemsetAsync(d_counts + 1, 0, sizeof(int));
        stitch_unresolved_kernel<<<(m + block - 1) / block, block>>>(
            label, cluster_scratch_active.data().get(), d_counts + 1, m);
        int remaining = 1;
        cudaMemcpy(&remaining, d_counts + 1, sizeof(int),
            cudaMemcpyDeviceToHost);
        if ( remaining == 0 ) break;
    }

    // 3. Members = endpoints of active stitches, compacted then sorted by
    // (root, id) with the preallocated cub sort (value == key low half).
    char* flag = cluster_scratch_endpoint.data().get();
    cudaMemsetAsync(flag, 0, n * sizeof(char));
    mark_stitch_endpoints_kernel<<<(m + block - 1) / block, block>>>(
        cluster_scratch_active.data().get(), flag, m);
    cudaMemsetAsync(d_counts, 0, sizeof(int));
    compact_flagged_kernel<<<(n + block - 1) / block, block>>>(
        flag, stitch_cluster_members.data().get(), d_counts, n);
    int mc = 0;
    cudaMemcpy(&mc, d_counts, sizeof(int), cudaMemcpyDeviceToHost);
    if ( mc == 0 ) return;

    pack_member_keys_kernel<<<(mc + block - 1) / block, block>>>(
        label, stitch_cluster_members.data().get(),
        cluster_scratch_keys_a.data().get(), mc);
    cub::DeviceRadixSort::SortKeys(cluster_scratch_sort_temp.data().get(),
        cluster_sort_temp_bytes,
        cluster_scratch_keys_a.data().get(), cluster_scratch_keys_b.data().get(), mc);
    unpack_members_kernel<<<(mc + block - 1) / block, block>>>(
        cluster_scratch_keys_b.data().get(), stitch_cluster_members.data().get(), mc);

    // 4. Per-root ranges and per-vertex cluster ids.
    compute_lookup<<<(n + block - 1) / block, block>>>(
        cluster_scratch_keys_b.data().get(), mc, n,
        stitch_cluster_lookup.data().get());
    assign_stitch_cluster_ids_kernel<<<(n + block - 1) / block, block>>>(
        label, flag, stitch_cluster_id.data().get(), n);
    
    update_seam_state(); // areas + bend validity follow the cluster state
}

// Per-cluster coincidence projection: all free members snap to the
// inv-mass weighted centroid in one pass (no iteration; clusters are
// 2-4 members). Pinned members anchor the target. Gated by cluster
// spread and clamped into the CCD trajectory envelope.
static __global__ void project_stitch_clusters_kernel(
    float3* __restrict__ pos,
    const float* __restrict__ inv_mass,
    int* __restrict__ cluster_locked,
    const int2* __restrict__ cluster_lookup,
    const int* __restrict__ cluster_members,
    const float3* __restrict__ pos_prev,
    const float3* __restrict__ pos_target,
    float snap_dist,
    // 1 = force the merge: ignore the spread gate and the trajectory envelope
    // for this call, i.e. put every free member on the cluster target.
    int force_merge,
    int n
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if ( c >= n ) return;
    int2 range = cluster_lookup[c];
    if ( range.y == 0 ) return; // only roots with members proceed
    int begin = range.x, end = range.x + range.y;

    float3 sum = make_float3(0.f, 0.f, 0.f);
    float3 pinned_sum = make_float3(0.f, 0.f, 0.f);
    float wsum = 0.f;
    int pinned = 0;
    for ( int k = begin; k < end; k++ ) {
        int v = cluster_members[k];
        float w = inv_mass[v];
        if ( w > 0.f ) { sum += pos[v] * w; wsum += w; }
        else { pinned_sum += pos[v]; pinned++; }
    }
    if ( wsum <= 0.f ) return; // whole cluster pinned
    float3 target = (pinned > 0) ? pinned_sum / (float)pinned : sum / wsum;

    // Lock only when the whole cluster has closed (junctions need all
    // panels in place before the snap).
    float spread = 0.f;
    for ( int k = begin; k < end; k++ )
        spread = fmaxf(spread, norm(pos[cluster_members[k]] - target));
    if ( !force_merge && spread > snap_dist ) return;

    cluster_locked[c] = 1;
    for ( int k = begin; k < end; k++ ) {
        int v = cluster_members[k];
        if ( inv_mass[v] <= 0.f ) continue;
        pos[v] = target;
    }
}

void Geometry::project_stitches(cudaStream_t stream) {
    if ( stitch_cluster_members.empty() || stitch_cluster_lookup.empty() ) return;
    const int n = params.nb_all_vertices;
    // Projection activates after the assembly window; the gate then
    // ramps so stuck seams are absorbed progressively (replaces the old
    // forced-connect teleport; the envelope clamp caps each substep).
    int activation = max(0, (int)get_global_parameter("sewing_forced_connect_frame", 80.f));
    if ( simulator->frame <= activation ) return;
    float snap_dist = max(0.f, get_global_parameter("sewing_snap_dist", 3e-3f));
    // The ramp widens the gate as the assembly settles. Its ceiling has to be
    // wide enough to admit the seam gaps that actually exist: a box-arranged
    // element starts with stitch pairs tens of centimetres apart (181 mm mean,
    // 552 mm max on the study-1 element), so a 5 cm ceiling made every such
    // cluster permanently ineligible and left the stitch springs to drag the
    // panels together - which is what stretches the fabric. The per-step
    // motion is still bounded by the trajectory envelope below.
    const float snap_max = max(0.f, get_global_parameter("sewing_snap_max_dist", 1.f));
    snap_dist = min(
        snap_dist * powf(1.5f, (float)(simulator->frame - activation)), snap_max);
    // A forced merge runs for the first few frames after the activation gate:
    // it puts every free stitch member on its cluster target exactly once, so
    // the assembly does not have to be dragged there by the stitch springs
    // (which is what leaves the panels stretched). Continuous teleporting
    // would freeze the clusters, hence the short window.
    const int force_window =
        max(0, (int)get_global_parameter("sewing_force_merge_frames", 0.f));
    const int force_merge =
        (force_window > 0 && simulator->frame <= activation + force_window) ? 1 : 0;
    int block = 256;
    cudaMemsetAsync(stitch_cluster_locked.data().get(), 0, sizeof(int) * n, stream);
    project_stitch_clusters_kernel<<<(n + block - 1) / block, block, 0, stream>>>(
        pos_world.data().get(), mass_inv.data().get(),
        stitch_cluster_locked.data().get(),
        stitch_cluster_lookup.data().get(),
        stitch_cluster_members.data().get(),
        pos_step_prev.data().get(), pos_pred.data().get(),
        snap_dist, force_merge, n);
}

// Remove the relative velocity injected by the snap. The snap preserves
// net momentum (mass-weighted centroid), so a momentum-weighted average
// restores a consistent cluster velocity.
static __global__ void average_stitch_cluster_velocities_kernel(
    float3* __restrict__ vel,
    const float* __restrict__ mass,
    const float* __restrict__ inv_mass,
    const int* __restrict__ cluster_locked,
    const int2* __restrict__ cluster_lookup,
    const int* __restrict__ cluster_members,
    int n
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if ( c >= n ) return;
    int2 range = cluster_lookup[c];
    if ( range.y == 0 || !cluster_locked[c] ) return;
    int begin = range.x, end = range.x + range.y;
    float3 p = make_float3(0.f, 0.f, 0.f);
    float msum = 0.f;
    for ( int k = begin; k < end; k++ ) {
        int v = cluster_members[k];
        if ( inv_mass[v] > 0.f ) { p += vel[v] * mass[v]; msum += mass[v]; }
    }
    if ( msum <= 0.f ) return;
    float3 v_avg = p / msum;
    for ( int k = begin; k < end; k++ ) {
        int v = cluster_members[k];
        if ( inv_mass[v] > 0.f ) vel[v] = v_avg;
    }
}

void Geometry::average_stitch_cluster_velocities() {
    if ( stitch_cluster_members.empty() || stitch_cluster_lookup.empty() ) return;
    const int n = params.nb_all_vertices;
    int block = 256;
    average_stitch_cluster_velocities_kernel<<<(n + block - 1) / block, block>>>(
        velocities.data().get(), masses.data().get(), mass_inv.data().get(),
        stitch_cluster_locked.data().get(), stitch_cluster_lookup.data().get(),
        stitch_cluster_members.data().get(), n);
}

// ===========================================================================
// Unified bending structure (mesh edges + seam hinges)
// ===========================================================================

// Build the static part of one seam hinge entry from the consecutive
// stitch pair (i, i+1): locate both sides' boundary edges, take the
// first valid side as the hinge, first valid opposite slot as apexes.
// Writes the unified slot arrays at index i (launched with a +ne offset).
static __global__ void build_seam_bend_entries_kernel(
    int4* __restrict__ bend_points,     // slot pointers, base = ne
    float* __restrict__ bend_factor,
    float* __restrict__ bend_rest_theta,
    char* __restrict__ static_ok,
    const int2* __restrict__ stitches,
    const int* __restrict__ stitch_sewing,
    const SewingData* __restrict__ sewing_lines,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const int2* __restrict__ edge_opposite_points,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    const float* __restrict__ edge_lengths,
    const float* __restrict__ areas,
    const float3* __restrict__ pos_2D,
    const int* __restrict__ vertices_obj,
    const ObjectDataInput* __restrict__ obj_data,
    int nb_all_edges,
    int ns
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= ns ) return;
    bend_rest_theta[i] = 0.f;
    static_ok[i] = 0;
    if ( i + 1 >= ns || stitch_sewing[i] != stitch_sewing[i + 1] ) return;
    int2 s0 = stitches[i], s1 = stitches[i + 1];

    int eA = -1, eB = -1;
    bool fA = find_edge(s0.x, s1.x, edge_lookup, dir_edges, eA);
    bool fB = find_edge(s0.y, s1.y, edge_lookup, dir_edges, eB);
    int hinge = fA ? eA : (fB ? eB : -1); // first valid side is the hinge
    if ( hinge < 0 || hinge >= nb_all_edges ) return;
    int2 opA = fA ? edge_opposite_points[eA] : make_int2(-1, -1);
    int2 opB = fB ? edge_opposite_points[eB] : make_int2(-1, -1);
    int x2 = opA.x != -1 ? opA.x : opA.y; // first valid slot
    int x3 = opB.x != -1 ? opB.x : opB.y;
    int2 he = edges[hinge];
    if ( x2 < 0 || x3 < 0 ) return;
    // Index degeneracy guard (dart apexes collapse onto the hinge).
    if ( x2 == x3 || x2 == he.x || x2 == he.y || x3 == he.x || x3 == he.y ) return;

    bend_points[i] = make_int4(he.x, he.y, x2, x3);
    bend_rest_theta[i] = sewing_lines[stitch_sewing[i]].angle;

    // Dihedral factor: same formula as precompute_dihedral_bending_factor,
    // from the two adjacent triangles of the seam sides.
    int2 trisA = fA ? e2t[eA] : make_int2(-1, -1);
    int2 trisB = fB ? e2t[eB] : make_int2(-1, -1);
    int tA = trisA.x != -1 ? trisA.x : trisA.y;
    int tB = trisB.x != -1 ? trisB.x : trisB.y;
    float area_sum = 0.f;
    if ( tA != -1 ) area_sum += areas[tA];
    if ( tB != -1 ) area_sum += areas[tB];
    float l = edge_lengths[hinge];
    if ( area_sum <= 1e-12f || l <= 1e-12f ) {
        bend_factor[i] = 1.f;
        static_ok[i] = 1;
        return;
    }
    const ObjectDataInput& od = obj_data[vertices_obj[he.x]];
    float3 ev = pos_2D[he.y] - pos_2D[he.x];
    float evn = norm(ev);
    if ( evn <= 1e-12f ) {
        bend_factor[i] = 1.f;
        static_ok[i] = 1;
        return;
    }
    ev /= evn;
    float3 grain = make_float3(cosf(od.grain_dir), sinf(od.grain_dir), 0.f);
    float3 cross_grain = make_float3(-sinf(od.grain_dir), cosf(od.grain_dir), 0.f);
    float lu = dot(ev, grain), lv = dot(ev, cross_grain);
    float lon = od.bending.x + od.bending.z;
    float lat = od.bending.y + od.bending.z;
    bend_factor[i] = 3.f * l * l / area_sum * (lu * lu * lon + lv * lv * lat);
    static_ok[i] = 1;
}

// Areas are a function of cluster state: a triangle holding two members
// of one cluster collapses once it snaps. Recomputed from Dms (never
// mutated) so previously zeroed areas restore on tear.
static __global__ void refresh_triangle_areas_kernel(
    float* __restrict__ areas,
    const Mat2* __restrict__ Dms,
    const int3* __restrict__ triangle_indices,
    const int* __restrict__ cluster_id,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int3 t = triangle_indices[i];
    bool collapsed =
        (cluster_id[t.x] >= 0 && (cluster_id[t.x] == cluster_id[t.y]
            || cluster_id[t.x] == cluster_id[t.z])) ||
        (cluster_id[t.y] >= 0 && cluster_id[t.y] == cluster_id[t.z]);
    areas[i] = collapsed ? 0.f : 0.5f * fabsf(Dms[i].det());
}

// Mesh-edge bending validity: needs both apexes, and must not be a
// zero-length edge after cluster snaps (dart cross edges).
static __global__ void refresh_edge_bend_validity_kernel(
    char* __restrict__ bend_valid,
    const int2* __restrict__ eop,
    const int2* __restrict__ edges,
    const int* __restrict__ cluster_id,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int2 op = eop[i];
    int2 e = edges[i];
    bend_valid[i] = (op.x != -1 && op.y != -1 &&
        !(cluster_id[e.x] >= 0 && cluster_id[e.x] == cluster_id[e.y]));
}

// Seam hinge validity: static part (apexes found) plus "both endpoint
// stitches connected" — torn invalidates, running does not (bending
// guides the seam toward its rest angle during assembly too).
static __global__ void refresh_seam_bend_validity_kernel(
    char* __restrict__ bend_valid,
    const char* __restrict__ static_ok,
    const char* __restrict__ status,
    int nb_edges,
    int ns
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= ns ) return;
    bool ok = static_ok[i] && status[i] != stitch_status_torn
        && (i + 1 < ns && status[i + 1] != stitch_status_torn);
    bend_valid[nb_edges + i] = ok;
}

// Emit the six normalized pair keys of one entry; MISS when a vertex id
// is invalid (dead entries / boundary apexes) or the pair degenerates.
static __global__ void generate_bend_pair_keys_kernel(
    const int4* __restrict__ bend_points,
    unsigned long long* __restrict__ keys, // 6 per entry
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int4 p = bend_points[i];
    int v[4] = { p.x, p.y, p.z, p.w };
    constexpr unsigned long long MISS = ~0ull;
    int slot = 6 * i;
    const int idx[6][2] = { {0,1},{0,2},{1,2},{0,3},{1,3},{2,3} };
    #pragma unroll
    for ( int k = 0; k < 6; k++ ) {
        int a = v[idx[k][0]], b = v[idx[k][1]];
        if ( a < 0 || b < 0 || a == b ) { keys[slot + k] = MISS; continue; }
        int lo = min(a, b), hi = max(a, b);
        keys[slot + k] = ((unsigned long long)lo << 32) | (unsigned int)hi;
    }
}

// Resolve the six row indices of one entry: natural pair -> edge id
// (== front index of the table), otherwise binary search in the sorted
// extension keys. Dead slots point at row 0 (blocks never written).
static __global__ void resolve_bend_rows_kernel(
    const int4* __restrict__ bend_points,
    const unsigned long long* __restrict__ ext_keys, // sorted, deduped
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    int* __restrict__ rows,          // 6 per entry
    int nb_all_cloth_edges,
    int ext_count,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int4 p = bend_points[i];
    int v[4] = { p.x, p.y, p.z, p.w };
    constexpr unsigned long long MISS = ~0ull;
    int slot = 6 * i;
    const int idx[6][2] = { {0,1},{0,2},{1,2},{0,3},{1,3},{2,3} };
    #pragma unroll
    for ( int k = 0; k < 6; k++ ) {
        int a = v[idx[k][0]], b = v[idx[k][1]];
        if ( a < 0 || b < 0 || a == b ) { rows[slot + k] = 0; continue; }
        int lo = min(a, b), hi = max(a, b);
        int e = v2e(lo, hi, edge_lookup, dir_edges);
        if ( e >= 0 ) { rows[slot + k] = e; continue; } // front of the table
        unsigned long long key = ((unsigned long long)lo << 32) | (unsigned int)hi;
        int l = 0, r = ext_count; // lower_bound over ext_keys
        while ( l < r ) {
            int mid = (l + r) >> 1;
            if ( ext_keys[mid] < key ) l = mid + 1; else r = mid;
        }
        rows[slot + k] = nb_all_cloth_edges + l; // exact hit: key came from ext
    }
}

static __global__ void unpack_pairs_kernel(
    const unsigned long long* __restrict__ keys,
    int2* __restrict__ pairs, int offset, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    pairs[offset + i] = make_int2(
        (int)(keys[i] >> 32), (int)(keys[i] & 0xFFFFFFFFULL));
}
// Mesh-edge slots of the unified bend arrays: hinge from the edge,
// apexes from edge_opposite_points, factor from the precomputed
// per-edge dihedral factor.
static __global__ void fill_mesh_bend_entries_kernel(
    int4* __restrict__ bend_points,
    float* __restrict__ bend_factor,
    const int2* __restrict__ edges,
    const int2* __restrict__ edge_opposite_points,
    const float* __restrict__ bending_factor,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int2 e = edges[i];
    int2 op = edge_opposite_points[i];
    bend_points[i] = make_int4(e.x, e.y, op.x, op.y);
    bend_factor[i] = bending_factor[i];
}

void Geometry::init_bend_structure() {
    const int ne = params.nb_all_cloth_edges;
    const int ns = params.nb_all_stitches;
    const int N = ne + ns;

    // stitch -> sewing line (static; same construction the old
    // init_sewing used).
    stitch_sewing.resize(ns);
    auto source_iter_begin = thrust::make_transform_iterator(
        sewing_lines.begin(),
        [] __host__ __device__ (const SewingData& key) { return key.start_idx; });
    thrust::upper_bound(thrust::device,
        source_iter_begin, source_iter_begin + (int)sewing_lines.size(),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(ns),
        stitch_sewing.begin());
    thrust::transform(thrust::device, stitch_sewing.begin(), stitch_sewing.end(),
        stitch_sewing.begin(), thrust::placeholders::_1 - 1);

    bend_points.assign(N, make_int4(-1, -1, -1, -1)); // dead slots keep -1
    bend_factor.assign(N, 1.f);
    bend_rest_theta.assign(N, 0.f); // mesh edges flat; internal lines later
    bend_valid.assign(N, 0);
    seam_bend_static_ok.assign(ns, 0);

    int block = 256;
    fill_mesh_bend_entries_kernel<<<(ne + block - 1) / block, block>>>(
        bend_points.data().get(), bend_factor.data().get(),
        edges.data().get(), edge_opposite_points.data().get(),
        bending_factor.data().get(), ne);
    build_seam_bend_entries_kernel<<<(ns + block - 1) / block, block>>>(
        bend_points.data().get() + ne,   // seam slots start at ne
        bend_factor.data().get() + ne,
        bend_rest_theta.data().get() + ne,
        seam_bend_static_ok.data().get(),
        stitches.data().get(), stitch_sewing.data().get(),
        sewing_lines.data().get(),
        edges.data().get(), e2t.data().get(),
        edge_opposite_points.data().get(),
        edge_lookup.data().get(), dir_edges.data().get(),
        edge_lengths.data().get(), areas.data().get(),
        pos_2D.data().get(), vertices_obj.data().get(), obj_data.data().get(),
        params.nb_all_edges, ns);


    // Build the unified pair table: front = natural edges, tail = every
    // non-natural bend pair, sorted and deduplicated.
    constexpr unsigned long long MISS = ~0ull;
    thrust::device_vector<unsigned long long> keys(6 * N);
    generate_bend_pair_keys_kernel<<<(N + block - 1) / block, block>>>(
        bend_points.data().get(), keys.data().get(), N);

    thrust::device_vector<unsigned long long> tmp(6 * N);
    auto valid_end = thrust::copy_if(thrust::cuda::par_nosync,
        keys.begin(), keys.end(), tmp.begin(),
        [] __device__ (unsigned long long k) { return k != MISS; });
    thrust::device_vector<unsigned long long> ext(valid_end - tmp.begin());
    auto nat_end = thrust::copy_if(thrust::cuda::par_nosync,
        tmp.begin(), valid_end, ext.begin(),
        [lookup = edge_lookup.data().get(),
         dir_edges = dir_edges.data().get()] __device__ (unsigned long long k) {
            int a = (int)(k >> 32), b = (int)(k & 0xFFFFFFFFULL);
            return v2e(a, b, lookup, dir_edges) < 0;
        });
    ext.resize(nat_end - ext.begin());
    thrust::sort(thrust::cuda::par_nosync, ext.begin(), ext.end());
    ext.erase(thrust::unique(thrust::cuda::par_nosync, ext.begin(), ext.end()), ext.end());
    const int ext_count = (int)ext.size();

    valid_pairs.resize(ne + ext_count);
    cudaMemcpyAsync(valid_pairs.data().get(), edges.data().get(),
        ne * sizeof(int2), cudaMemcpyDeviceToDevice);
    if ( ext_count > 0 )
        unpack_pairs_kernel<<<(ext_count + block - 1) / block, block>>>(
            ext.data().get(), valid_pairs.data().get(), ne, ext_count);

    bend_cross_rows.assign(6 * N, 0);
    resolve_bend_rows_kernel<<<(N + block - 1) / block, block>>>(
        bend_points.data().get(), ext.data().get(),
        edge_lookup.data().get(), dir_edges.data().get(),
        bend_cross_rows.data().get(), ne, ext_count, N);

    // Fill both validity arrays and refresh collapsed-triangle areas
    // (the call inside build_stitch_clusters no-ops while bend_valid
    // was still empty).
    update_seam_state();
}

void Geometry::update_seam_state() {
    if ( bend_valid.empty() ) return; // bend structure not built yet
    const int ntri = params.nb_all_cloth_triangles;
    const int ne = params.nb_all_cloth_edges;
    const int ns = params.nb_all_stitches;
    int block = 256;
    refresh_triangle_areas_kernel<<<(ntri + block - 1) / block, block>>>(
        areas.data().get(), Dms.data().get(), triangle_indices.data().get(),
        stitch_cluster_id.data().get(), ntri);
    refresh_edge_bend_validity_kernel<<<(ne + block - 1) / block, block>>>(
        bend_valid.data().get(), edge_opposite_points.data().get(),
        edges.data().get(), stitch_cluster_id.data().get(), ne);
    if ( ns > 0 )
        refresh_seam_bend_validity_kernel<<<(ns + block - 1) / block, block>>>(
            bend_valid.data().get(), seam_bend_static_ok.data().get(),
            stitches_status.data().get(), ne, ns);
}
