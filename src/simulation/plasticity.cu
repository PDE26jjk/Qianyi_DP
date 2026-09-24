#include "geometry.cuh"

#include "common/cuda_utils.h"
#include "dynamics/bending.cuh"

#include <cmath>

// ---------------------------------------------------------------------------
// Cloth plasticity: the rest-shape input that authors the rest state, and the
// time-dependent wrinkle model that moves it.
//
// See the `cloth-plasticity` capability. The model is the bending-family
// friction + elasto-plastic update of Gong et al. 2025 in the engine's own
// dihedral convention (0 = flat): an internal-friction anchor angle with
// stick-slip and a dwell-dependent threshold, plus an elasto-plastic rest angle
// with a yield threshold and time-dependent hardening. The bending kernels
// consume the two angles as one aggregate first derivative (see D3 in
// design.md), so nothing else in the solver changes.
//
// Everything here is per bend entry and per substep: no atomics, no ordering
// dependence, and the whole update is skipped when no panel opted in.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rest-shape input (design D12-D14)
// ---------------------------------------------------------------------------

// `compress` is the relative change of an edge's rest length, so it feeds the
// spring-mass rest length directly. Applied once at scene init, after the
// pattern-derived weights exist, so the bending factor, the triangle areas and
// the masses keep describing the pattern (see design D13).
static __global__ void apply_edge_compress_kernel(
    float* __restrict__ edge_lengths,
    const float* __restrict__ edge_compress,
    int nb_all_cloth_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_all_cloth_edges ) return;
    const float c = edge_compress[i];
    if ( c == 0.f ) return;
    edge_lengths[i] *= (1.f + c);
}

// The FEM (BW) rest state is a per-triangle metric, so an edge-shrink input is
// folded in as the mean relative change of the triangle's three edges: the rest
// metric carries the authored rest shape while the StVK energy weight (`areas`)
// and the masses stay those of the pattern.
static __global__ void apply_triangle_compress_kernel(
    Mat2* __restrict__ Dms,
    const int3* __restrict__ triangle_edges,
    const float* __restrict__ edge_compress,
    int nb_all_cloth_triangles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_all_cloth_triangles ) return;
    const int3 te = triangle_edges[i];
    if ( te.x < 0 || te.y < 0 || te.z < 0 ) return;
    float s = 1.f + (edge_compress[te.x] + edge_compress[te.y] + edge_compress[te.z]) / 3.f;
    if ( s == 1.f ) return;
    // A metric that collapses or inverts would make Dm^-1 blow up.
    s = fminf(fmaxf(s, 1e-2f), 1e2f);
    Dms[i] = Dms[i] * s;
}

// ---------------------------------------------------------------------------
// Plastic state
// ---------------------------------------------------------------------------

// Per-entry opt-in: the owning object of the hinge edge (the same object the
// bending factor precompute uses) plus "the entry exists at all". The live
// validity (`bend_valid`) is checked by the update kernel itself, because it is
// refreshed on cluster rebuilds.
static __global__ void build_plastic_mask_kernel(
    char* __restrict__ bend_plastic_enabled,
    const int4* __restrict__ bend_points,
    const int* __restrict__ vertices_obj,
    const ObjectDataInput* __restrict__ obj_data,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const int4 p = bend_points[i];
    if ( p.x < 0 ) {
        bend_plastic_enabled[i] = 0;
        return;
    }
    bend_plastic_enabled[i] = obj_data[vertices_obj[p.x]].plastic ? 1 : 0;
}

// A single flag telling the host whether any entry takes part, so the per-
// substep update and its launch disappear entirely for an unflagged scene.
static __global__ void any_plastic_kernel(
    const char* __restrict__ bend_plastic_enabled,
    int n,
    int* __restrict__ out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    if ( bend_plastic_enabled[i] ) atomicExch(out, 1);
}

// Restore the input rest shape and clear every derived plastic quantity:
// scene init uses it after copying the input angles into
// `bend_rest_theta_elastic`, and the explicit reset uses it as is.
static __global__ void reset_bend_plasticity_kernel(
    float* __restrict__ bend_rest_theta,
    float* __restrict__ bend_anchor_theta,
    float* __restrict__ bend_stick_t,
    float* __restrict__ bend_plastic_t,
    float* __restrict__ bend_plastic_hp,
    float* __restrict__ bend_yield_theta,
    const float* __restrict__ bend_rest_theta_elastic,
    const float yield0,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const float rest = bend_rest_theta_elastic[i];
    bend_rest_theta[i] = rest;
    bend_anchor_theta[i] = rest;
    bend_stick_t[i] = 0.f;
    bend_plastic_t[i] = 0.f;
    bend_plastic_hp[i] = 0.f;
    bend_yield_theta[i] = yield0;
}

// One substep of the model at the current dihedral angle of every enabled
// entry: internal friction first (anchor + stick-slip + dwell), then the
// elasto-plastic rest-angle flow (yield + hardening). `h_eff` is the scaled
// substep (`plasticity_time_scale * h`), so a time scale of 0 evaluates the
// model at t = 0 without advancing either timer.
static __global__ void update_bend_plasticity_kernel(
    float* __restrict__ bend_rest_theta,
    float* __restrict__ bend_anchor_theta,
    float* __restrict__ bend_stick_t,
    float* __restrict__ bend_plastic_t,
    float* __restrict__ bend_plastic_hp,
    float* __restrict__ bend_yield_theta,
    const float* __restrict__ bend_rest_theta_elastic,
    const char* __restrict__ bend_plastic_enabled,
    const char* __restrict__ bend_valid,
    const int4* __restrict__ bend_points,
    const float3* __restrict__ vertices,
    const float thres0,
    const float thres_inf,
    const float tau_f,
    const float yield0,
    const float hardening_ratio,
    const float hardening_g,
    const float tau_p,
    const float h_eff,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    if ( !bend_plastic_enabled[i] || !bend_valid[i] ) return;

    const int4 p = bend_points[i];
    float3 th_dp0, th_dp1, th_dp2, th_dp3;
    float theta;
    get_theta_dpk(vertices[p.x], vertices[p.y], vertices[p.z], vertices[p.w],
        th_dp0, th_dp1, th_dp2, th_dp3, theta);

    // Internal friction: the anchor only moves once the deviation passes the
    // (dwell-dependent) slip threshold, and the stick timer then resets.
    const float thres = thres_inf - (thres_inf - thres0) * __expf(-bend_stick_t[i] / tau_f);
    const float delta = theta - bend_anchor_theta[i];
    if ( fabsf(delta) > thres ) {
        bend_anchor_theta[i] += copysignf(fabsf(delta) - thres, delta);
        bend_stick_t[i] = 0.f;
    }
    else {
        bend_stick_t[i] += h_eff;
    }

    // Elasto-plastic rest angle. The plastic strain is measured against the
    // input rest angle, which doubles as the comparison direction of the
    // hardening timer.
    const float eps_e = theta - bend_rest_theta[i];
    const float yield_now = bend_yield_theta[i];
    if ( fabsf(eps_e) > yield_now ) {
        const float eps_p = bend_rest_theta[i] - bend_rest_theta_elastic[i];
        const float sign_e = copysignf(1.f, eps_e);
        if ( sign_e == copysignf(1.f, eps_p) ) {
            bend_plastic_t[i] += h_eff;
        }
        else {
            bend_plastic_t[i] = 0.f;
        }
        // Hardening stiffness: K_h = K_h0 (1 - g (1 - exp(-t / tau_p))), as a
        // ratio of the edge stiffness, so the flow share beta and the yield
        // growth need no per-edge geometry.
        float kh_ratio = hardening_ratio
            * (1.f - hardening_g * (1.f - __expf(-bend_plastic_t[i] / tau_p)));
        kh_ratio = fmaxf(kh_ratio, 0.f);
        const float beta = 1.f / (1.f + kh_ratio);
        const float excess = fabsf(eps_e) - yield_now;
        bend_plastic_hp[i] += beta * excess;
        bend_rest_theta[i] += sign_e * beta * excess;
        bend_yield_theta[i] = yield0 + bend_plastic_hp[i] * kh_ratio;
    }
}

// Freeze: adopt the current configuration as the rest shape. The elastic
// reference is untouched, so a reset still returns to the input rest shape.
static __global__ void freeze_bend_rest_theta_kernel(
    float* __restrict__ bend_rest_theta,
    float* __restrict__ bend_anchor_theta,
    float* __restrict__ bend_stick_t,
    float* __restrict__ bend_plastic_t,
    const char* __restrict__ bend_plastic_enabled,
    const char* __restrict__ bend_valid,
    const int4* __restrict__ bend_points,
    const float3* __restrict__ vertices,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    if ( !bend_plastic_enabled[i] || !bend_valid[i] ) return;
    const int4 p = bend_points[i];
    float3 th_dp0, th_dp1, th_dp2, th_dp3;
    float theta;
    get_theta_dpk(vertices[p.x], vertices[p.y], vertices[p.z], vertices[p.w],
        th_dp0, th_dp1, th_dp2, th_dp3, theta);
    bend_rest_theta[i] = theta;
    bend_anchor_theta[i] = theta;
    bend_stick_t[i] = 0.f;
    bend_plastic_t[i] = 0.f;
}

// Readback layout: five floats per bend entry.
//   [rest angle, anchor angle, yield angle, stick timer, plastic timer]
static __global__ void gather_plasticity_state_kernel(
    float* __restrict__ out,
    const float* __restrict__ bend_rest_theta,
    const float* __restrict__ bend_anchor_theta,
    const float* __restrict__ bend_yield_theta,
    const float* __restrict__ bend_stick_t,
    const float* __restrict__ bend_plastic_t,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    float* row = out + 5 * i;
    row[0] = bend_rest_theta[i];
    row[1] = bend_anchor_theta[i];
    row[2] = bend_yield_theta[i];
    row[3] = bend_stick_t[i];
    row[4] = bend_plastic_t[i];
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

void Geometry::apply_rest_shape_input() {
    const int n_edges = params.nb_all_cloth_edges;
    const int n_tris = params.nb_all_cloth_triangles;
    if ( n_edges <= 0 ) return;
    const int block = 256;

    bool any_compress = false;
    for ( int i = 0; i < (int)edge_compress.size() && i < n_edges; ++i ) {
        if ( edge_compress[i] != 0.f ) {
            any_compress = true;
            break;
        }
    }
    if ( !any_compress ) return;

    apply_edge_compress_kernel<<<(n_edges + block - 1) / block, block>>>(
        edge_lengths.data().get(), edge_compress.data().get(), n_edges);

    // The FEM planar model carries the rest state as a per-triangle metric.
    if ( constitutive_model == ConstitutiveModel::FEM_BW && n_tris > 0 ) {
        apply_triangle_compress_kernel<<<(n_tris + block - 1) / block, block>>>(
            Dms.data().get(), triangles.data().get(), edge_compress.data().get(), n_tris);
    }
}

void Geometry::init_plastic_state() {
    const int n = (int)bend_rest_theta.size();
    if ( n <= 0 ) return;
    bend_rest_theta_elastic.resize(n);
    bend_anchor_theta.resize(n);
    bend_stick_t.resize(n);
    bend_plastic_t.resize(n);
    bend_plastic_hp.resize(n);
    bend_yield_theta.resize(n);
    bend_plastic_enabled.resize(n);
    // The rest angle of this scene is the elastic reference; every derived
    // quantity starts from it.
    thrust::copy(bend_rest_theta.begin(), bend_rest_theta.end(),
        bend_rest_theta_elastic.begin());

    const int block = 256;
    build_plastic_mask_kernel<<<(n + block - 1) / block, block>>>(
        bend_plastic_enabled.data().get(),
        bend_points.data().get(),
        vertices_obj.data().get(),
        obj_data.data().get(), n);

    const float yield0 = get_global_parameter("plastic_bend_yield", 1.8f);
    reset_bend_plasticity_kernel<<<(n + block - 1) / block, block>>>(
        bend_rest_theta.data().get(),
        bend_anchor_theta.data().get(),
        bend_stick_t.data().get(),
        bend_plastic_t.data().get(),
        bend_plastic_hp.data().get(),
        bend_yield_theta.data().get(),
        bend_rest_theta_elastic.data().get(),
        yield0, n);

    thrust::device_vector<int> any_flag(1, 0);
    any_plastic_kernel<<<(n + block - 1) / block, block>>>(
        bend_plastic_enabled.data().get(), n, any_flag.data().get());
    plasticity_enabled = any_flag[0] != 0;
}

void Geometry::accumulate_bend_plasticity(float h) {
    if ( !plasticity_enabled ) return;
    const int n = (int)bend_rest_theta.size();
    if ( n <= 0 ) return;
    // The timers advance by the scaled substep; a scale of 0 keeps both timers
    // at zero, which is the model evaluated at t = 0.
    const float time_scale = fmaxf(0.f, get_global_parameter("plasticity_time_scale", 0.f));
    const float h_eff = time_scale * h;
    const int block = 256;
    update_bend_plasticity_kernel<<<(n + block - 1) / block, block>>>(
        bend_rest_theta.data().get(),
        bend_anchor_theta.data().get(),
        bend_stick_t.data().get(),
        bend_plastic_t.data().get(),
        bend_plastic_hp.data().get(),
        bend_yield_theta.data().get(),
        bend_rest_theta_elastic.data().get(),
        bend_plastic_enabled.data().get(),
        bend_valid.data().get(),
        bend_points.data().get(),
        // The state is measured on the configuration the substep starts from;
        // `pos_world` has already been moved to the warm-start prediction by
        // the time the PDNewton iteration begins.
        pos_step_prev.data().get(),
        fmaxf(0.f, get_global_parameter("plastic_bend_thres0", 0.1f)),
        fmaxf(0.f, get_global_parameter("plastic_bend_thres_inf", 1.2f)),
        fmaxf(1e-6f, get_global_parameter("plastic_bend_dwell_tau", 30.f)),
        fmaxf(0.f, get_global_parameter("plastic_bend_yield", 1.8f)),
        fmaxf(0.f, get_global_parameter("plastic_bend_hardening", 1.f)),
        fminf(fmaxf(get_global_parameter("plastic_bend_hardening_g", 0.99f), 0.f), 1.f),
        fmaxf(1e-6f, get_global_parameter("plastic_bend_hardening_tau", 30.f)),
        h_eff, n);
}

void Geometry::freeze_bend_rest_shape() {
    const int n = (int)bend_rest_theta.size();
    if ( n <= 0 || !plasticity_enabled ) return;
    const int block = 256;
    freeze_bend_rest_theta_kernel<<<(n + block - 1) / block, block>>>(
        bend_rest_theta.data().get(),
        bend_anchor_theta.data().get(),
        bend_stick_t.data().get(),
        bend_plastic_t.data().get(),
        bend_plastic_enabled.data().get(),
        bend_valid.data().get(),
        bend_points.data().get(),
        pos_world.data().get(), n);
}

void Geometry::reset_bend_plasticity() {
    const int n = (int)bend_rest_theta.size();
    if ( n <= 0 || bend_rest_theta_elastic.size() != (size_t)n ) return;
    const int block = 256;
    const float yield0 = get_global_parameter("plastic_bend_yield", 1.8f);
    reset_bend_plasticity_kernel<<<(n + block - 1) / block, block>>>(
        bend_rest_theta.data().get(),
        bend_anchor_theta.data().get(),
        bend_stick_t.data().get(),
        bend_plastic_t.data().get(),
        bend_plastic_hp.data().get(),
        bend_yield_theta.data().get(),
        bend_rest_theta_elastic.data().get(),
        yield0, n);
}

void Geometry::copy_plasticity_state(float* out) const {
    const int n = (int)bend_rest_theta.size();
    if ( n <= 0 || out == nullptr ) return;
    thrust::device_vector<float> staged(5 * n);
    const int block = 256;
    gather_plasticity_state_kernel<<<(n + block - 1) / block, block>>>(
        staged.data().get(),
        bend_rest_theta.data().get(),
        bend_anchor_theta.data().get(),
        bend_yield_theta.data().get(),
        bend_stick_t.data().get(),
        bend_plastic_t.data().get(), n);
    cudaMemcpy(out, staged.data().get(), 5 * n * sizeof(float), cudaMemcpyDeviceToHost);
}
