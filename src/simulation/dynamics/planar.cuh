#pragma once
#include "common/vec_math.h"
#include "common/atomic_utils.cuh"

// Membrane stiffness: the force per unit strain of one mesh edge (N/m). One
// value drives both planar models so the spring-mass and the FEM_BW paths
// stay in correspondence; the FEM factor is the equilateral-triangle
// approximation (2*sqrt(3)) this model has always used.
//
// The value comes from the `base_spring_stiffness` parameter (see
// Geometry::init). The default is calibrated against low-load tensile data of
// woven apparel fabric: the measured E*t of such fabric is 3-10 kN/m (FAST
// E100 / KES) and a triangular spring lattice has E*t ~= 1.15 * k, which puts
// the shipped value at 4 kN/m.
constexpr float default_base_spring_stiffness = 4.0e3f;
constexpr float fem_stiffness_factor = 3.4641f; // 2*sqrt(3)

// Strain stiffening (X. Provot, Graphics Interface 1995): below `start` the
// membrane keeps its calibrated modulus; above it the modulus grows
// exponentially with the strain, so the cloth resists being pulled long while
// a forced deformation (drag, pin) can still stretch it against a rapidly
// growing force. Tension only - compression keeps the base modulus, so the
// excess material still folds instead of pushing back.
constexpr float default_strain_stiffen_start = 5.0e-2f; // 5% strain
constexpr float default_strain_stiffen_rate = 2.0f;
constexpr float strain_stiffen_max_factor = 1.0e3f;

__device__ inline float strain_stiffen_factor(float strain, float start, float rate) {
    if ( start <= 0.f || strain <= start ) return 1.f;
    return fminf(expf(rate * (strain / start - 1.f)), strain_stiffen_max_factor);
}


// T. Liu, A. W. Bargteil, J. F. O’Brien, and L. Kavan, "Fast simulation of mass-spring systems," ACM Trans. Graph., vol. 32, no. 6, p. 214:1-214:7, Nov. 2013, doi: 10.1145/2508363.2508406.
static __global__ void pd_precompute_spring_forces(
    float* __restrict__ Jx_diag_scalar,
    float* __restrict__ Jx_nondiag_scalar,
    const int2* __restrict__ edges,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int n, // edge size
    const float base_spring_k // membrane stiffness, N/m
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [v0,v1] = edges[i];
        float3 ks = obj_data[vertices_obj[v0]].stretch;

        float k = base_spring_k * (ks.x + ks.y + ks.z) * 0.333f;
        float weight = k;
        atomicAdd(&Jx_diag_scalar[v0], weight);
        atomicAdd(&Jx_diag_scalar[v1], weight);
        Jx_nondiag_scalar[i] -= weight;
    }
}
enum HessianRegularization:char {
    NONE = 0,                  // Exact Hessian (may be indefinite under compression)
    PROJECTIVE,                // H = k * I  (isotropic, always SPD)
    SPD_CLAMP                  // H = k*(n n^T) + k*max(1-L0/l, -eps)*(I - n n^T)
};

__device__ inline void calc_spring_elastic(
    const float3 p0,
    const float3 p1,
    const float rest_length,
    float stiffness,
    float3& force_elastic,         // f for v0, -f for v1
    Mat3* H_elastic_ptr = nullptr, // K for diag, -K for non-diag
    HessianRegularization reg_type = NONE,
    float* energy_ptr = nullptr
) {
    float3 d = p0 - p1;
    float len = norm(d);
    // ---- Degenerate case: length nearly zero ----
    if ( len < 1e-12f ) {
        force_elastic = make_float3(0, 0, 0);
        if ( H_elastic_ptr ) *H_elastic_ptr = Mat3::zero();
        if ( energy_ptr ) *energy_ptr = 0.0f;
        return;
    }

    // ---- Direction and strain ----
    float3 n = d / len;                  // unit vector from v1 to v0
    float strain = len - rest_length;

    // ---- Force (exact gradient) ----
    // E = 1/2 k (l - L0)^2
    // f0 = -dE/dx0 = -k (l - L0) * (dl/dx0) = -k (l - L0) n
    force_elastic = -stiffness * strain * n;

    // ---- Energy (optional) ----
    if ( energy_ptr ) {
        *energy_ptr = 0.5f * strain * strain * stiffness;
    }

    // ---- Hessian (self‑block for v0) ----
    if ( H_elastic_ptr ) {
        Mat3& H_elastic = *H_elastic_ptr;
        if ( reg_type == PROJECTIVE ) {
            // Projective Dynamics isotropic approximation:
            // H_PD = k I   (always SPD, ignores directional stiffness)
            H_elastic = Mat3::identity(stiffness);
        }
        else if ( reg_type == SPD_CLAMP ) {
            // Exact H decomposed as:
            //   H = k (n n^T) + k(1 - L0/l) (I - n n^T)
            // We clamp the lateral eigenvalue to avoid indefiniteness:
            //   k_t = k * max(1 - L0/l, epsilon)
            // Then H_SPD = k (n n^T) + k_t (I - n n^T)
            const float eps = -1.0e-3f;
            float kt = stiffness * max(1.0f - rest_length / len, eps);
            float kn = stiffness;
            // Construct as: H = (kn - kt) * (n n^T) + kt * I
            Mat3 h = Mat3::outer_product(n, n * (kn - kt));
            h.add_diag(kt);
            H_elastic = h;
        }
        else { // NONE
            // Exact Hessian:
            // H = k I - k (L0/l) (I - n n^T)
            //   = (k - k L0/l) I + (k L0/l) n n^T
            float coef = stiffness * (rest_length / len); // = k * L0/l
            Mat3 h = Mat3::outer_product(n, n * coef); // + coef * n n^T
            h.add_diag(stiffness - coef); // + (k - coef) * I
            H_elastic = h;
        }
    }
}


static __global__ void accumulate_spring_forces(
    Mat3* __restrict__ Jx_nondiag,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ energys,
    const float3* __restrict__ vertices, // world space
    const int2* __restrict__ edges,
    const float* __restrict__ edge_lengths,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int n, // edge size
    const float base_spring_k, // membrane stiffness, N/m
    const float stiffen_start, // strain where stiffening starts
    const float stiffen_rate   // exponential stiffening rate
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [v0,v1] = edges[i];
        float3 p0 = vertices[v0], p1 = vertices[v1];
        float3 ks = obj_data[vertices_obj[v0]].stretch;
        float rest_length = edge_lengths[i];
        float k = base_spring_k * (ks.x + ks.y + ks.z) * 0.333f;
        const float strain = rest_length > 1e-12f
            ? (norm(p0 - p1) - rest_length) / rest_length : 0.f;
        k *= strain_stiffen_factor(strain, stiffen_start, stiffen_rate);
        float3 force;
        Mat3 K;
        float energy;
        calc_spring_elastic(p0, p1, rest_length, k, force,
            Jx_diag ? &K : nullptr, SPD_CLAMP,
            energys ? &energy : nullptr);
        atomicAddFloat3(&forces[v0], force);
        atomicAddFloat3(&forces[v1], -force);
        if ( energys ) {
            atomicAdd(&energys[v0], energy);
        }
        if ( Jx_diag ) {
            atomicAddMat3(&Jx_diag[v0], K);
            atomicAddMat3(&Jx_diag[v1], K);
            if ( Jx_nondiag ) {
                atomicAddMat3(&Jx_nondiag[i], -K);
            }
        }
    }
}

// triangular finite element. The formula derivation comes from
//T. Kim and D. Eberle, "Dynamic deformables: implementation and production practicalities (now with code!)," in ACM SIGGRAPH 2022 Courses  (Chapter 10)
static __global__ void compute_BW_FEM(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangle_edges,
    const int2* __restrict__ edges,
    const Mat2* __restrict__ Dms,
    const float* __restrict__ areas,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    int num_triangles,
    // 1 = clamp the lateral eigenvalue of the stretch Hessian to >= -eps.
    // The exact value is ku * (1 - 1/|wu|), which is negative in compression;
    // the assembled operator then loses positive definiteness and the linear
    // solve stops converging (see the SPD_CLAMP regularization used by the
    // spring element below).
    float psd_clamp,
    // 1 = add the (positive semidefinite) rank-one part of the shear Hessian,
    // which the force terms already use but the operator was missing.
    float shear_hessian,
    // Membrane stiffness of the spring-mass model (N/m); the planar
    // finite-element stiffness is derived from it with `fem_stiffness_factor`.
    const float base_spring_k,
    const float stiffen_start, // strain where stiffening starts
    const float stiffen_rate   // exponential stiffening rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_triangles )
        return;
    float area = areas[i];
    if ( area <= 0.f ) return; // collapsed dart triangle (seam cluster)

    Mat2 Dm = Dms[i];
    // float Dm_det = Dm.det();
    // float area = fabs(Dm_det) * 0.5f;
    Mat2 Dm_inv = Dm.inverse();

    auto tri_edges = triangle_edges[i];
    // auto [v0_idx,v1_idx,v2_idx] = indeices[i];
    auto [e1_ii,e2_ii,e3_ii] = tri_edges;
    int2 e1_i = edges[tri_edges.x];
    int2 e2_i = edges[tri_edges.y];
    int2 e3_i = edges[tri_edges.z];

    int v0_idx = e1_i.x;
    int v1_idx = e1_i.y;
    int v2_idx = e2_i.y;

    float3 v0 = vertices[v0_idx];
    float3 e1 = vertices[v1_idx] - v0;
    float3 e2 = vertices[v2_idx] - v0;

    // stiffnesses from object data (stretch.x = u, .y = v, .z = shear)
    const float3 stretch =
        obj_data[vertices_obj[v0_idx]].stretch * (base_spring_k * fem_stiffness_factor);
    float ku = stretch.x;
    float kv = stretch.y;
    const float ks = stretch.z;

    float wudp1 = Dm_inv.r[0].x; // Dm_inv[0, 0]
    float wvdp1 = Dm_inv.r[0].y; // Dm_inv[0, 1]
    float wudp2 = Dm_inv.r[1].x; // Dm_inv[1, 0]
    float wvdp2 = Dm_inv.r[1].y; // Dm_inv[1, 1]
    float3 wu = e1 * wudp1 + e2 * wudp2;
    float3 wv = e1 * wvdp1 + e2 * wvdp2;
    float wu_norm = norm(wu);
    float wv_norm = norm(wv);
    float3 wu_ = wu_norm > 1e-12f ? wu * (1.0f / wu_norm) : make_float3(0.0f, 0.0f, 0.0f);
    float3 wv_ = wv_norm > 1e-12f ? wv * (1.0f / wv_norm) : make_float3(0.0f, 0.0f, 0.0f);

    // Constraint violations
    float Cu = wu_norm - 1.0f;
    float Cv = wv_norm - 1.0f;

    // Strain stiffening, tension only (see accumulate_spring_forces).
    ku *= strain_stiffen_factor(Cu, stiffen_start, stiffen_rate);
    kv *= strain_stiffen_factor(Cv, stiffen_start, stiffen_rate);

    // Gradients w.r.t. material coordinates
    float3 Cudp1 = wu_ * wudp1;
    float3 Cvdp1 = wv_ * wvdp1;
    float3 Cudp2 = wu_ * wudp2;
    float3 Cvdp2 = wv_ * wvdp2;

    // ---- shear ----
    float wu_dot_wv = dot(wu_, wv_);
    float Cshear = wu_dot_wv; // cos angle

    // Derivatives of the shear constraint w.r.t. the two material directions,
    // used by both the force and the (optional) shear Hessian below.
    float3 dCs_dx1 = (wv_ - wu_ * wu_dot_wv) * (wu_norm > 1e-6f ? 1.0f / wu_norm : 0.0f) * wudp1
        + (wu_ - wv_ * wu_dot_wv) * (wv_norm > 1e-6f ? 1.0f / wv_norm : 0.0f) * wvdp1;
    float3 dCs_dx2 = (wv_ - wu_ * wu_dot_wv) * (wu_norm > 1e-6f ? 1.0f / wu_norm : 0.0f) * wudp2
        + (wu_ - wv_ * wu_dot_wv) * (wv_norm > 1e-6f ? 1.0f / wv_norm : 0.0f) * wvdp2;

    // ---- forces (negative gradient of energy) ----
    // energy = 0.5 * area * (ku * Cu^2 + kv * Cv^2 + ks * Cs^2)
    // => f = -area * (ku * Cu * dCu/dp + kv * Cv * dCv/dp + ks * Cs * dCs/dp)
    float3 f1 = -area * (ku * Cu * Cudp1 + kv * Cv * Cvdp1);
    float3 f2 = -area * (ku * Cu * Cudp2 + kv * Cv * Cvdp2);
    float shear_coef = -area * ks * Cshear;
    float3 f1_s = dCs_dx1 * shear_coef;
    float3 f2_s = dCs_dx2 * shear_coef;
    float3 f0 = (f1_s + f2_s + f1 + f2) * -1.0f;
    atomicAddFloat3(&forces[v0_idx], f0);
    atomicAddFloat3(&forces[v1_idx], f1 + f1_s);
    atomicAddFloat3(&forces[v2_idx], f2 + f2_s);
    if ( enerys ) {
        float energy = 0.5f * area * (ku * Cu * Cu + kv * Cv * Cv + ks * Cshear * Cshear);
        atomicAdd(&enerys[v0_idx], energy);
    }

    // ---- Hessian blocks ----
    Mat3 I = Mat3::identity();
    // Lateral eigenvalue factor of 0.5 * area * k * (|w| - 1)^2: it is
    // (|w| - 1) / |w| = 1 - 1/|w|, positive in tension and negative in
    // compression. Clamping it (to the same -1e-3 floor the spring element
    // uses) keeps the element Hessian positive semidefinite.
    const float lat_eps = -1.0e-3f;
    float lat_u = 1.0f - 1.0f / max(wu_norm, 1e-12f);
    float lat_v = 1.0f - 1.0f / max(wv_norm, 1e-12f);
    if ( psd_clamp > 0.5f ) {
        lat_u = fmaxf(lat_u, lat_eps);
        lat_v = fmaxf(lat_v, lat_eps);
    }
    Mat3 wu_lat = I - Mat3::outer_product(wu_, wu_);
    Mat3 wv_lat = I - Mat3::outer_product(wv_, wv_);

    float coef = area;
    // Hessian of stretch energy (w.r.t. material coords)
    // d²E / d p1² = area * [ ku * (Cudp1 Cudp1ᵀ + Cu * wu_proj * wudp1²) +
    //                          kv * (Cvdp1 Cvdp1ᵀ + Cv * wv_proj * wvdp1²) ]
    // (similarly for p2 and cross term)
    Mat3 f1d1 = (Mat3::outer_product(Cudp1, Cudp1 * ku) + Mat3::outer_product(Cvdp1, Cvdp1 * kv) +
        wu_lat * (lat_u * wudp1 * wudp1 * ku) + wv_lat * (lat_v * wvdp1 * wvdp1 * kv)) * coef;

    Mat3 f2d2 = (Mat3::outer_product(Cudp2, Cudp2 * ku) + Mat3::outer_product(Cvdp2, Cvdp2 * kv) +
        wu_lat * (lat_u * wudp2 * wudp2 * ku) + wv_lat * (lat_v * wvdp2 * wvdp2 * kv)) * coef;

    Mat3 f1d2 = (Mat3::outer_product(Cudp1, Cudp2 * ku) + Mat3::outer_product(Cvdp1, Cvdp2 * kv) +
        wu_lat * (lat_u * wudp1 * wudp2 * ku) + wv_lat * (lat_v * wvdp1 * wvdp2 * kv)) * coef;

    // Rank-one part of the shear Hessian, d²E_s / d p_i d p_j with
    // E_s = 0.5 * area * ks * Cs². Jx holds the energy Hessian, so the
    // coefficient is +area * ks (the disabled block below used the opposite
    // sign, which is the force Jacobian convention).
    if ( shear_hessian > 0.5f ) {
        float s_coef = area * ks;
        f1d1 += Mat3::outer_product(dCs_dx1, dCs_dx1 * s_coef);
        f2d2 += Mat3::outer_product(dCs_dx2, dCs_dx2 * s_coef);
        f1d2 += Mat3::outer_product(dCs_dx1, dCs_dx2 * s_coef);
    }
    if ( Jx_diag ) {
        atomicAddMat3(&Jx_diag[v1_idx], f1d1);
        atomicAddMat3(&Jx_diag[v2_idx], f2d2);

        Mat3 f0d2 = (f1d2 + f2d2) * -1.0f;
        Mat3 f0d1 = (f1d1 + f1d2.transpose()) * -1.0f;
        Mat3 f0d0 = (f0d1 + f0d2) * -1.0f;

        atomicAddMat3(&Jx_diag[v0_idx], f0d0);
    }
    if ( Jx ) {
        if ( v1_idx < v2_idx )
            atomicAddMat3(&Jx[e3_ii], f1d2);
        else
            atomicAddMat3(&Jx[e3_ii], f1d2.transpose());

        auto f0d2 = -(f1d2 + f2d2);
        atomicAddMat3(&Jx[e2_ii], f0d2);
        auto f0d1 = -(f1d1 + f1d2.transpose());
        atomicAddMat3(&Jx[e1_ii], f0d1);
    }

    #if 0 // Hessian of shear, only SPD part (superseded by `shear_hessian`
           // above; kept for reference - note its coefficient sign).
    float3 dCs_dx1 = wv_proj_ * wudp1 + wu_proj_ * wvdp1;
    float3 dCs_dx2 = wv_proj_ * wudp2 + wu_proj_ * wvdp2;

    float s_coef = -area * ks; 

    Mat3 f1d1_s = Mat3::outer_product(dCs_dx1, dCs_dx1) * s_coef;
    Mat3 f2d2_s = Mat3::outer_product(dCs_dx2, dCs_dx2) * s_coef;
    Mat3 f1d2_s = Mat3::outer_product(dCs_dx1, dCs_dx2) * s_coef;

    if (Jx_diag) {
        atomicAddMat3(&Jx_diag[v1_idx], f1d1_s);
        atomicAddMat3(&Jx_diag[v2_idx], f2d2_s);
        
        Mat3 f0d1_s = (f1d1_s + f1d2_s.transpose()) * -1.0f;
        Mat3 f0d2_s = (f1d2_s + f2d2_s) * -1.0f;
        Mat3 f0d0_s = (f0d1_s + f0d2_s) * -1.0f;
        atomicAddMat3(&Jx_diag[v0_idx], f0d0_s);
    }

    if (Jx) {
        // e3: v1-v2, e2: v0-v2, e1: v0-v1
        if (v1_idx < v2_idx) atomicAddMat3(&Jx[e3_ii], f1d2_s);
        else atomicAddMat3(&Jx[e3_ii], f1d2_s.transpose());

        atomicAddMat3(&Jx[e2_ii], -(f1d2_s + f2d2_s));
        atomicAddMat3(&Jx[e1_ii], -(f1d1_s + f1d2_s.transpose()));
    }
    #endif
}


// === 1. 解析 SVD 计算 U(3x3), S(2), V(2x2) ===
static __device__ void svd3x2_analytic(const float3& wu, const float3& wv,
    float3& U0, float3& U1, float3& U2,
    float& s0, float& s1,
    float& v00, float& v01, float& v10, float& v11) {
    // F^T * F
    float c00 = dot(wu, wu);
    float c11 = dot(wv, wv);
    float c01 = dot(wu, wv);

    float delta = c00 - c11;
    float disc = sqrtf(delta * delta + 4.0f * c01 * c01);

    // 特征值 (奇异值的平方)
    float l0 = (c00 + c11 + disc) * 0.5f;
    float l1 = (c00 + c11 - disc) * 0.5f;
    s0 = sqrtf(max(l0, 1e-8f));
    s1 = sqrtf(max(l1, 1e-8f));

    // 计算 V (2x2)
    if ( abs(c01) > 1e-6f ) {
        float angle = 0.5f * atan2f(2.0f * c01, delta);
        v00 = cosf(angle);
        v10 = sinf(angle);
        v01 = -v10;
        v11 = v00;
    }
    else {
        v00 = 1.0f;
        v01 = 0.0f;
        v10 = 0.0f;
        v11 = 1.0f;
    }

    // 计算 U 的前两列 (3x1)
    U0 = (wu * v00 + wv * v10) * (1.0f / s0);
    U1 = (wu * v01 + wv * v11) * (1.0f / s1);

    // 计算 U 的第三列 (法线，用于面外 Twist)
    U2 = cross(U0, U1);
    float n_len = norm(U2);
    U2 = (n_len > 1e-6f) ? (U2 * (1.0f / n_len)) : make_float3(0.0f, 0.0f, 1.0f);
}

template<bool FixedR = true>
static __global__ void compute_ARAP_FEM(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangle_edges,
    const int2* __restrict__ edges,
    const int* __restrict__ vertices_obj,
    const float* __restrict__ YoungsModulus,
    const Mat2* __restrict__ Dms,
    int num_triangles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_triangles ) return;

    Mat2 Dm_inv = Dms[i].inverse();
    float area = fabs(Dms[i].det()) * 0.5f;

    auto tri_edges = triangle_edges[i];
    int v0_idx = edges[tri_edges.x].x;
    int v1_idx = edges[tri_edges.x].y;
    int v2_idx = edges[tri_edges.y].y;

    float3 v0 = vertices[v0_idx], v1 = vertices[v1_idx], v2 = vertices[v2_idx];
    float3 e1 = v1 - v0, e2 = v2 - v0;

    float m00 = Dm_inv.r[0].x, m10 = Dm_inv.r[0].y;
    float m01 = Dm_inv.r[1].x, m11 = Dm_inv.r[1].y;

    // F = [wu, wv]
    float3 wu = e1 * m00 + e2 * m10;
    float3 wv = e1 * m01 + e2 * m11;

    // 1. SVD 分析
    float3 U0, U1, U2;
    float s0, s1, v00, v01, v10, v11;
    svd3x2_analytic(wu, wv, U0, U1, U2, s0, s1, v00, v01, v10, v11);

    // int obj = vertices_obj[v0_idx];
    // float mu = YoungsModulus[obj]; TODO
    float mu = 8e3;
    float coef = -mu * area;

    // --- (First Piola-Kirchhoff) ---
    // R = U * I_{3x2} * V^T
    float3 r_col0 = U0 * v00 + U1 * v01;
    float3 r_col1 = U0 * v10 + U1 * v11;

    float3 p0 = wu - r_col0;
    float3 p1 = wv - r_col1;

    float3 f1 = (p0 * m00 + p1 * m01) * coef;
    float3 f2 = (p0 * m10 + p1 * m11) * coef;
    // float3 f0 = -(f1 + f2);

    atomicAddFloat3(&forces[v0_idx], -(f1 + f2));
    atomicAddFloat3(&forces[v1_idx], f1);
    atomicAddFloat3(&forces[v2_idx], f2);

    if ( enerys ) {
        atomicAdd(&enerys[v0_idx], 0.5f * coef * (dot(p0, p0) + dot(p1, p1)));
    }

    // --- Hessian 的特征值分析与截断 (Exact Eigensystem) ---
    if ( Jx || Jx_diag ) {
        // 计算 Dm_inv_T_inv 矩阵 (2x2)
        Mat3 I = Mat3::identity();
        Mat3 f1d1, f2d2, f1d2;
        if constexpr ( FixedR ) {
            f1d1 = I * (m00 * m00 + m01 * m01) * coef;
            f2d2 = I * (m10 * m10 + m11 * m11) * coef;
            f1d2 = I * (m00 * m10 + m01 * m11) * coef;
        }
        else {
            // 构造 Twist 模式 (严格对应 MATLAB 中的 T0, T1, T2 降维到 3x2)
            const float inv_sqrt2 = 0.707106781f;

            // 面内 Twist (T0)
            float3 t0_u = (U1 * v00 - U0 * v01) * inv_sqrt2;
            float3 t0_v = (U1 * v10 - U0 * v11) * inv_sqrt2;

            // 面外 Twist 1 (T1), s2 = 0
            float3 t1_u = -U2 * v01 * inv_sqrt2;
            float3 t1_v = -U2 * v11 * inv_sqrt2;

            // 面外 Twist 2 (T2), s2 = 0
            float3 t2_u = -U2 * v00 * inv_sqrt2;
            float3 t2_v = -U2 * v10 * inv_sqrt2;

            // 特征值计算与截断 (Clamping)
            // 原始 Hessian 特征值为 lambda_i = 2 - 4/(s_i + s_j)
            // 截断后的权重 weight_i = 2 - max(0, lambda_i)
            auto clamp_weight = [](float s_a, float s_b) {
                float lambda = 2.0f - 4.0f / (s_a + s_b + 1e-8f);
                return 2.0f - max(0.0f, lambda);
            };

            float w0 = clamp_weight(s0, s1);
            float w1 = clamp_weight(s1, 0.0f); // s2 = 0
            float w2 = clamp_weight(s0, 0.0f); // s2 = 0

            // 构造 6x6 d2E/dF2 的四个 3x3 子块
            // H_F = 2 * I - w0 * (t0*t0^T) - w1 * (t1*t1^T) - w2 * (t2*t2^T)
            Mat3 H_uu = I * 2.0f;
            Mat3 H_uv = Mat3::zero();
            Mat3 H_vu = Mat3::zero();
            Mat3 H_vv = I * 2.0f;

            auto rank1_update = [&](float weight, const float3& tu, const float3& tv) {
                H_uu = H_uu - Mat3::outer_product(tu, tu) * weight;
                H_vv = H_vv - Mat3::outer_product(tv, tv) * weight;
                Mat3 uv = Mat3::outer_product(tu, tv) * weight;
                H_uv = H_uv - uv;
                H_vu = H_vu - uv.transpose();
            };

            rank1_update(w0, t0_u, t0_v);
            rank1_update(w1, t1_u, t1_v);
            rank1_update(w2, t2_u, t2_v);

            // 应用材质坐标链式法则 (映射到 9x9 Node Hessian)
            f1d1 = (H_uu * (m00 * m00) + H_uv * (m00 * m01) + H_vu * (m01 * m00) + H_vv * (m01 * m01)) * coef;
            f2d2 = (H_uu * (m10 * m10) + H_uv * (m10 * m11) + H_vu * (m11 * m10) + H_vv * (m11 * m11)) * coef;
            f1d2 = (H_uu * (m00 * m10) + H_uv * (m00 * m11) + H_vu * (m01 * m10) + H_vv * (m01 * m11)) * coef;
        }
        if ( Jx_diag ) {
            atomicAddMat3(&Jx_diag[v1_idx], f1d1);
            atomicAddMat3(&Jx_diag[v2_idx], f2d2);
            Mat3 f0d0 = f1d1 + f2d2 + f1d2 + f1d2.transpose();
            atomicAddMat3(&Jx_diag[v0_idx], f0d0);
        }
        if ( Jx ) {
            if ( v1_idx < v2_idx ) atomicAddMat3(&Jx[tri_edges.z], f1d2);
            else atomicAddMat3(&Jx[tri_edges.z], f1d2.transpose());

            atomicAddMat3(&Jx[tri_edges.y], -(f1d2 + f2d2));    // f0d2
            atomicAddMat3(&Jx[tri_edges.x], -(f1d1 + f1d2.transpose())); // f0d1
        }

    }
}
