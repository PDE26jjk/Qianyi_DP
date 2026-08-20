#pragma once

#include "common/vec_math.h"
#include "common/atomic_utils.cuh"

//  Reference: Z. Wang, Y. Yang, H. Wang, "Stable Discrete Bending by Analytic
//  Eigensystem and Adaptive Orthotropic Geometric Stiffness", ACM TOG 42(6),
//  Article 183, 2023.

// ============================================================================
// Vertex order (follow style3D):
//   x0, x1 : hinge endpoints (e = normalize(x1 - x0))
//   x2     : off-hinge vertex of the left triangle
//   x3     : off-hinge vertex of the right triangle
// Normals (paper convention):
//   n1 = normalize(cross(x2 - x0, x1 - x0))
//   n2 = normalize(cross(x1 - x0, x3 - x0))
// Altitudes:
//   m1 = normalize(proj(x2) - x2), m2 = normalize(proj(x3) - x3)
//        (from the off-hinge vertex toward the hinge line)
//
// The returned angle keeps the original project convention, 0 when flat
//   theta = atan2(dot(cross(n1,n2), e), dot(n1,n2))
// The gradient is the closed-form 
//   grad(theta) = t1*n1 + t2*n2,
// ============================================================================
static __device__ void get_theta_dpk(
    float3 x0, float3 x1, float3 x2, float3 x3,
    float3& theta_dp0, float3& theta_dp1, float3& theta_dp2, float3& theta_dp3, float& theta
) {
    float3 ev = x1 - x0;
    float l = norm(ev);

    float3 n1_raw = cross(x2 - x0, ev);
    float3 n2_raw = cross(ev, x3 - x0);
    float n1n = norm(n1_raw), n2n = norm(n2_raw);

    float3 e_ = (l > 1e-6f) ? ev / l : make_float3(0, 0, 0);
    float3 n1_ = (n1n > 1e-6f) ? n1_raw / n1n : make_float3(0, 0, 0);
    float3 n2_ = (n2n > 1e-6f) ? n2_raw / n2n : make_float3(0, 0, 0);

    theta = atan2(dot(cross(n1_, n2_), e_), dot(n1_, n2_));

    if ( n1n <= 1e-6f || n2n <= 1e-6f || l <= 1e-6f ) {
        theta_dp0 = theta_dp1 = theta_dp2 = theta_dp3 = make_float3(0, 0, 0);
        return;
    }

    // Hinge altitudes and barycentric weights (h1,h2,omega1,omega2).
    float3 proj2 = x0 + e_ * dot(x2 - x0, e_);
    float3 alt2 = proj2 - x2;                 // m1 direction
    float h1 = norm(alt2);
    float ih1 = (h1 > 1e-6f) ? 1.0f / h1 : 0.0f;
    float w1 = dot(x2 - x0, e_) / l;

    float3 proj3 = x0 + e_ * dot(x3 - x0, e_);
    float3 alt3 = proj3 - x3;                 // m2 direction
    float h2 = norm(alt3);
    float ih2 = (h2 > 1e-6f) ? 1.0f / h2 : 0.0f;
    float w2 = dot(x3 - x0, e_) / l;

    //   t1 = [(w1-1), -w1, 1, 0]/h1
    //   t2 = [(w2-1), -w2, 0, 1]/h2
    // grad(theta) = t1*n1 + t2*n2.
    theta_dp0 = n1_ * ((w1 - 1.0f) * ih1) + n2_ * ((w2 - 1.0f) * ih2);
    theta_dp1 = n1_ * (-w1 * ih1) + n2_ * (-w2 * ih2);
    theta_dp2 = n1_ * ih1;
    theta_dp3 = n2_ * ih2;
}

// Discrete Shells model, The energy density is proportional to one-half of the square of the difference between the dihedral angle and its rest dihedral angle.
// The algorithm uses the Hessian matrix of energy with only the outer product term. (Gauss-Newton method)
// When the fabric undergoes severe bending, intense folding, or inversion due to large-scale self-penetration, the value of ∂E/∂θ becomes very large.
// At this point, the geometric stiffness term ∂E/∂θ ∇²θ, which discarded, actually dominates the total stiffness.
// Discarding it will cause the curvature information obtained by the solver to be severely distorted, and the number of Newton iteration steps will increase exponentially.
static __global__ void compute_dihedral_bending_GN(
    Mat3* Jx,
    Mat3* Jx_diag,
    Mat3* Jx_bend_cross,
    float3* forces,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const float* __restrict__ rest_thetas,
    const int3* __restrict__ triangles,
    const int2* __restrict__ edge_opposite_points,
    int num_edges, float kb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;

    int2 p_op = edge_opposite_points[i];
    if ( p_op.x == -1 || p_op.y == -1 ) return; // No need to calculate bending force at the boundary.

    int2 e_i = edges[i];
    int x0_idx = e_i.x, x1_idx = e_i.y;
    int x2_idx = p_op.x, x3_idx = p_op.y;

    float3 theta_dp0, theta_dp1, theta_dp2, theta_dp3;
    float theta;
    get_theta_dpk(vertices[x0_idx], vertices[x1_idx], vertices[x2_idx], vertices[x3_idx],
        theta_dp0, theta_dp1, theta_dp2, theta_dp3, theta);

    float coef = kb;
    if ( Jx_diag != nullptr ) {
        atomicAddMat3(&Jx_diag[x0_idx], Mat3::outer_product(theta_dp0, theta_dp0 * coef));
        atomicAddMat3(&Jx_diag[x1_idx], Mat3::outer_product(theta_dp1, theta_dp1 * coef));
        atomicAddMat3(&Jx_diag[x2_idx], Mat3::outer_product(theta_dp2, theta_dp2 * coef));
        atomicAddMat3(&Jx_diag[x3_idx], Mat3::outer_product(theta_dp3, theta_dp3 * coef));
        if ( Jx != nullptr ) {
            auto [t1_i, t2_i] = e2t[i];
            auto tri1 = triangles[t1_i];
            auto tri2 = triangles[t2_i];
            auto f0d1 = Mat3::outer_product(theta_dp0, theta_dp1 * coef);
            atomicAddMat3(&Jx[i], f0d1);

            auto f2d0 = Mat3::outer_product(theta_dp2, theta_dp0 * coef);
            auto f2d1 = Mat3::outer_product(theta_dp2, theta_dp1 * coef);
            if ( x2_idx < x0_idx ) {
                atomicAddMat3(&Jx[tri1.x], f2d0);
                atomicAddMat3(&Jx[tri1.y], f2d1);
            }
            else {
                atomicAddMat3(&Jx[tri1.y], f2d0.transpose());
                atomicAddMat3(&Jx[tri1.z], x2_idx < x1_idx ? f2d1 : f2d1.transpose());
            }
            auto f3d0 = Mat3::outer_product(theta_dp3, theta_dp0 * coef);
            auto f3d1 = Mat3::outer_product(theta_dp3, theta_dp1 * coef);
            if ( x3_idx < x0_idx ) {
                atomicAddMat3(&Jx[tri2.y], f3d0);
                atomicAddMat3(&Jx[tri2.x], f3d1);
            }
            else {
                atomicAddMat3(&Jx[tri2.x], f3d0.transpose());
                atomicAddMat3(&Jx[tri2.z], x3_idx < x1_idx ? f3d1 : f3d1.transpose());
            }
            auto f2d3 = Mat3::outer_product(theta_dp2, theta_dp3 * coef);
            // atomicAddMat3(&Jx_bend_cross[i],p0_idx < p3_idx ? f0d3 : f0d3.transpose());
            atomicAddMat3(&Jx_bend_cross[i], f2d3);
        }
    }
    coef *= -(theta - rest_thetas[i]);
    if ( forces != nullptr ) {
        atomicAddFloat3(&forces[x0_idx], theta_dp1 * coef);
        atomicAddFloat3(&forces[x1_idx], theta_dp2 * coef);
        atomicAddFloat3(&forces[x2_idx], theta_dp0 * coef);
        atomicAddFloat3(&forces[x3_idx], theta_dp3 * coef);
    }
}

struct AOGSGeo {
    float3 e;        // hinge unit direction, e = normalize(x1 - x0)
    float l;         // hinge length
    float3 n1, n2;   // triangle normals
    float3 m1, m2;   // altitude unit vectors, from x2/x3 toward the hinge line
    float h1, h2;    // hinge altitudes
    float w1, w2;    // barycentric weights of x2/x3 on the hinge
    float t1[4];     // t1 = [(w1-1), -w1, 1, 0]/h1
    float t2[4];     // t2 = [(w2-1), -w2, 0, 1]/h2
    float s[4];
    float ct, st;    // cos(theta_paper) = -n1.n2, sin(theta_paper) = n2.m1
};

// One pass: theta (project convention), its closed-form gradient, and the full
// paper geometry. Used by the AOGS kernel so nothing is recomputed.
static __device__ void get_theta_dpk_aogs(
    float3 x0, float3 x1, float3 x2, float3 x3,
    float3& theta_dp0, float3& theta_dp1, float3& theta_dp2, float3& theta_dp3,
    float& theta, AOGSGeo& geo
) {
    float3 ev = x1 - x0;
    geo.l = norm(ev);
    geo.e = (geo.l > 1e-6f) ? ev / geo.l : make_float3(0, 0, 0);

    float3 n1_raw = cross(x2 - x0, ev);
    float3 n2_raw = cross(ev, x3 - x0);
    float n1n = norm(n1_raw), n2n = norm(n2_raw);
    geo.n1 = (n1n > 1e-6f) ? n1_raw / n1n : make_float3(0, 0, 0);
    geo.n2 = (n2n > 1e-6f) ? n2_raw / n2n : make_float3(0, 0, 0);

    theta = atan2(dot(cross(geo.n1, geo.n2), geo.e), dot(geo.n1, geo.n2));

    // Altitudes: m points from the off-hinge vertex toward the hinge line.
    float3 proj2 = x0 + geo.e * dot(x2 - x0, geo.e);
    float3 alt2 = proj2 - x2;
    geo.h1 = norm(alt2);
    geo.m1 = (geo.h1 > 1e-6f) ? alt2 / geo.h1 : make_float3(0, 0, 0);
    geo.w1 = dot(x2 - x0, geo.e) / (geo.l + 1e-14f);

    float3 proj3 = x0 + geo.e * dot(x3 - x0, geo.e);
    float3 alt3 = proj3 - x3;
    geo.h2 = norm(alt3);
    geo.m2 = (geo.h2 > 1e-6f) ? alt3 / geo.h2 : make_float3(0, 0, 0);
    geo.w2 = dot(x3 - x0, geo.e) / (geo.l + 1e-14f);

    // Paper t1, t2, s over (x0,x1,x2,x3).
    float ih1 = 1.0f / (geo.h1 + 1e-14f);
    float ih2 = 1.0f / (geo.h2 + 1e-14f);
    float il = 1.0f / (geo.l + 1e-14f);
    geo.t1[0] = (geo.w1 - 1.0f) * ih1;
    geo.t1[1] = -geo.w1 * ih1;
    geo.t1[2] = ih1;
    geo.t1[3] = 0.0f;
    geo.t2[0] = (geo.w2 - 1.0f) * ih2;
    geo.t2[1] = -geo.w2 * ih2;
    geo.t2[2] = 0.0f;
    geo.t2[3] = ih2;
    geo.s[0] = il;
    geo.s[1] = -il;
    geo.s[2] = 0.0f;
    geo.s[3] = 0.0f;

    // Paper dihedral angle: n2 = -cos(theta)*n1 + sin(theta)*m1.
    geo.ct = -dot(geo.n1, geo.n2);
    geo.st = dot(geo.n2, geo.m1);

    // Closed-form gradient grad(theta) = t1*n1 + t2*n2.
    if ( n1n <= 1e-6f || n2n <= 1e-6f || geo.l <= 1e-6f ) {
        theta_dp0 = theta_dp1 = theta_dp2 = theta_dp3 = make_float3(0, 0, 0);
        return;
    }
    theta_dp0 = geo.n1 * geo.t1[0] + geo.n2 * geo.t2[0];
    theta_dp1 = geo.n1 * geo.t1[1] + geo.n2 * geo.t2[1];
    theta_dp2 = geo.n1 * ih1;
    theta_dp3 = geo.n2 * ih2;
}

// Diagonal of F' = E * clamp(Lambda,0) * E^T without forming the 8x8 product.
// AOGS only needs F'00..F'77 (paper Eq.15).
// g is the project delta angle (identical to the paper g), st/ct are the paper
// sin/cos of the dihedral angle.
static __device__ void aogs_fp_diag(
    float p, float g, float st, float ct, float Fp[8]
) {
    for ( int k = 0; k < 8; ++k ) Fp[k] = 0.0f;

    // Eigenvalues of F0 and F1 (paper Eq.8).
    float lam[8];
    float r = sqrtf(p * p + g * g);
    lam[0] = p + r;
    lam[1] = g;
    lam[2] = p - r;
    lam[3] = -g;
    float s2 = st * st;
    float A = sqrtf(s2 + 4.0f * (1.0f - ct));
    float B = sqrtf(s2 + 4.0f * (1.0f + ct));
    lam[4] = 0.5f * g * (st + A);
    lam[5] = 0.5f * g * (-st + B);
    lam[6] = 0.5f * g * (st - A);
    lam[7] = 0.5f * g * (-st - B);
#pragma unroll
    for ( int k = 0; k < 8; ++k ) lam[k] = fmaxf(lam[k], 0.0f); // Algorithm 1 clamp

    // g = 0 (reference bending state): single nonzero eigenvalue 2p with
    // eigenvector [1, -ct, st, 0]/sqrt(2).
    if ( fabsf(g) < 1e-5f ) {
        Fp[0] = p;
        Fp[1] = p * ct * ct;
        Fp[2] = p * st * st;
        return;
    }

    // F0 block: analytic eigenvectors (paper Eq.9).
#pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        if ( lam[i] <= 0.0f ) continue;
        float delta = (i == 1 || i == 3) ? -1.0f : 1.0f;
        float lg = lam[i] / g;
        float v0 = lg;
        float v1 = delta * (st - ct * lg);
        float v2 = delta * (ct + st * lg);
        float v3 = 1.0f;
        float inv_n = rsqrtf(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
        Fp[0] += lam[i] * (v0 * inv_n) * (v0 * inv_n);
        Fp[1] += lam[i] * (v1 * inv_n) * (v1 * inv_n);
        Fp[2] += lam[i] * (v2 * inv_n) * (v2 * inv_n);
        Fp[3] += lam[i] * (v3 * inv_n) * (v3 * inv_n);
    }

    // F1 block.
    if ( fabsf(st) < 1e-4f ) {
        // theta ~ 0, pi, 2pi: only the largest clamped eigenvalue of F1
        // survives; its limit eigenvector has squared components
        // (1/2, 0, 1/4, 1/4).
        float lmax = fmaxf(fmaxf(lam[4], lam[5]), fmaxf(lam[6], lam[7]));
        Fp[4] = 0.5f * lmax;
        Fp[5] = 0.0f;
        Fp[6] = 0.25f * lmax;
        Fp[7] = 0.25f * lmax;
        return;
    }
#pragma unroll
    for ( int i = 4; i < 8; ++i ) {
        if ( lam[i] <= 0.0f ) continue;
        float delta = (i == 5 || i == 7) ? -1.0f : 1.0f;
        float lg = lam[i] / g;
        float v0 = lg;
        float v1 = lg * (delta + ct) / st;
        float v2 = delta;
        float v3 = 1.0f;
        float inv_n = rsqrtf(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
        Fp[4] += lam[i] * (v0 * inv_n) * (v0 * inv_n);
        Fp[5] += lam[i] * (v1 * inv_n) * (v1 * inv_n);
        Fp[6] += lam[i] * (v2 * inv_n) * (v2 * inv_n);
        Fp[7] += lam[i] * (v3 * inv_n) * (v3 * inv_n);
    }
}

// ============================================================================
// Adaptive Orthotropic Geometric Stiffness bending
// The instruction count is more than double that of the GN method, but more stable.
// 
// H_o = mu [ p*P + a0*S0 + a1*S1 + a2*S2 + a3*S3 ]        (paper Eq.14)
//     = mu [ (p+a0/2) q0q0^T + (a0/2) q1q1^T
//            + (a1/2)(q2q2^T+q3q3^T) + (a2/2)(q4q4^T+q5q5^T)
//            + (a3/2)(q6q6^T+q7q7^T) ]
// with p = 1 (Discrete Shells), mu folded into kb, and a0..a3 from the
// diagonal of F' (paper Eq.15), computed directly without forming E*L'*E^T.
// ============================================================================
static __global__ void compute_dihedral_bending_AOGS(
    Mat3* Jx,
    Mat3* Jx_diag,
    Mat3* Jx_bend_cross,
    float3* forces,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const float* __restrict__ rest_thetas,
    const int3* __restrict__ triangles,
    const int2* __restrict__ edge_opposite_points,
    int num_edges, float kb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;

    int2 p_op = edge_opposite_points[i];
    if ( p_op.x == -1 || p_op.y == -1 ) return; // boundary: no bending element

    int2 e_i = edges[i];
    int x0_idx = e_i.x, x1_idx = e_i.y;   // hinge endpoints 
    int x2_idx = p_op.x, x3_idx = p_op.y; // off-hinge vertices 

    // ---- theta, gradient, and geometry in one pass ----
    AOGSGeo geo;
    float3 th_dp0, th_dp1, th_dp2, th_dp3;
    float theta;
    get_theta_dpk_aogs(
        vertices[x0_idx], vertices[x1_idx], vertices[x2_idx], vertices[x3_idx],
        th_dp0, th_dp1, th_dp2, th_dp3, theta, geo);

    // ---- forces: unchanged from the GN kernel ----
    // g = d psi / d theta = theta - rest_theta
    float g = theta - rest_thetas[i];
    if ( forces != nullptr ) {
        float coef = -kb * g;
        atomicAddFloat3(&forces[x0_idx], th_dp0 * coef);
        atomicAddFloat3(&forces[x1_idx], th_dp1 * coef);
        atomicAddFloat3(&forces[x2_idx], th_dp2 * coef);
        atomicAddFloat3(&forces[x3_idx], th_dp3 * coef);
    }
    if ( Jx_diag == nullptr ) return; // only forces were requested

    if ( geo.h1 < 1e-6f || geo.h2 < 1e-6f || geo.l < 1e-6f ) return; // degenerate

    // ---- adaptive parameters from the F' diagonal (paper Eq.15) ----
    const float p = 1.0f; // Discrete Shells: p = d2psi/dtheta2 = 1
    float Fp[8];
    aogs_fp_diag(p, g, geo.st, geo.ct, Fp);
    float a0 = fmaxf(0.0f, fmaxf(Fp[0], Fp[1]) - p);
    float a1 = fmaxf(0.0f, fmaxf(Fp[2], Fp[3]));
    float a2 = fmaxf(0.0f, fmaxf(Fp[6], Fp[7]));
    float a3 = fmaxf(0.0f, fmaxf(Fp[4], Fp[5]));

    // ---- disassembly of S_0..3 (Paper Eq.13) ---- 
    // 7 base outer products, computed once per edge.
    Mat3 N11 = Mat3::outer_product(geo.n1, geo.n1);
    Mat3 N22 = Mat3::outer_product(geo.n2, geo.n2);
    Mat3 M11 = Mat3::outer_product(geo.m1, geo.m1);
    Mat3 M22 = Mat3::outer_product(geo.m2, geo.m2);
    Mat3 N12 = Mat3::outer_product(geo.n1, geo.n2);
    Mat3 N21 = Mat3::outer_product(geo.n2, geo.n1);
    Mat3 EE = Mat3::outer_product(geo.e, geo.e);

    // Block (a,b) of H_o in paper vertex order, by exact rank reduction.
    auto calc_B = [&](int a, int b) -> Mat3 {
        float T11 = geo.t1[a] * geo.t1[b];
        float T22 = geo.t2[a] * geo.t2[b];
        float T12 = geo.t1[a] * geo.t2[b];
        float T21 = geo.t2[a] * geo.t1[b];
        float Sab = geo.s[a] * geo.s[b];
        float w_n1n1 = (p + a0) * T11 + a3 * Sab;
        float w_n2n2 = (p + a0) * T22 + a3 * Sab;
        float w_m1m1 = a1 * T11;
        float w_m2m2 = a1 * T22;
        float w_n1n2 = p * T12;
        float w_n2n1 = p * T21;
        float w_ee = a2 * (T11 + T22);
        return (N11 * w_n1n1 + N22 * w_n2n2 + M11 * w_m1m1 + M22 * w_m2m2 +
            N12 * w_n1n2 + N21 * w_n2n1 + EE * w_ee) * kb;
    };

    atomicAddMat3(&Jx_diag[x0_idx], calc_B(0, 0));
    atomicAddMat3(&Jx_diag[x1_idx], calc_B(1, 1));
    atomicAddMat3(&Jx_diag[x2_idx], calc_B(2, 2));
    atomicAddMat3(&Jx_diag[x3_idx], calc_B(3, 3));

    if ( Jx ) {
        auto [t1_i, t2_i] = e2t[i];
        auto tri1 = triangles[t1_i]; // triangle (x0,x1,x2)
        auto tri2 = triangles[t2_i]; // triangle (x0,x1,x3)

        atomicAddMat3(&Jx[i], calc_B(0, 1));          // hinge (x0,x1)
        atomicAddMat3(&Jx_bend_cross[i], calc_B(2, 3)); // cross (x2,x3)

        Mat3 B20 = calc_B(2, 0), B21 = calc_B(2, 1);   // left triangle
        if ( x2_idx < x0_idx ) {
            atomicAddMat3(&Jx[tri1.x], B20);
            atomicAddMat3(&Jx[tri1.y], B21);
        }
        else {
            atomicAddMat3(&Jx[tri1.y], B20.transpose());
            atomicAddMat3(&Jx[tri1.z], x2_idx < x1_idx ? B21 : B21.transpose());
        }

        Mat3 B30 = calc_B(3, 0), B31 = calc_B(3, 1);   // right triangle
        if ( x3_idx < x0_idx ) {
            atomicAddMat3(&Jx[tri2.y], B30);
            atomicAddMat3(&Jx[tri2.x], B31);
        }
        else {
            atomicAddMat3(&Jx[tri2.x], B30.transpose());
            atomicAddMat3(&Jx[tri2.z], x3_idx < x1_idx ? B31 : B31.transpose());
        }
    }

}

static __device__ float compute_cotangent(float3 p0, float3 p1, float3 p2) {
    float3 e1 = normalized(p1 - p0), e2 = normalized(p2 - p0);
    float dot_ = dot(e1, e2);
    float cross_ = norm(cross(e1, e2));
    const float eps = 1e-6f;
    return dot_ / (cross_ + eps);
}
static __global__ void precompute_IBM_Q(
    float4* q,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const float3* __restrict__ vertices, // Material space
    const int2* __restrict__ edge_opposite_points,
    const Mat2* __restrict__ Dms,
    int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;
    int2 p_op = edge_opposite_points[i];
    if ( p_op.x == -1 || p_op.y == -1 ) {
        return;
    }
    int2 e_i = edges[i];
    int p0_idx = e_i.x, p1_idx = e_i.y;
    int p2_idx = p_op.x, p3_idx = p_op.y;
    auto [t1_i, t2_i] = e2t[i];
    // auto t1 = triangles[t1_i];
    // auto t2 = triangles[t2_i];
    float area1 = fabs(Dms[t1_i].det()) * 0.5f;
    float area2 = fabs(Dms[t2_i].det()) * 0.5f;
    float k = 3.0f / (area1 + area2 + 1e-8f);
    float3 x0 = vertices[p0_idx], x1 = vertices[p1_idx], x2 = vertices[p2_idx], x3 = vertices[p3_idx];
    float c01 = compute_cotangent(x0, x1, x2); // cot(x1-x0, x2-x0) 
    float c02 = compute_cotangent(x0, x3, x1); // cot(x1-x0, x3-x0)
    float c03 = compute_cotangent(x1, x2, x0); // cot(x0-x1, x2-x1)
    float c04 = compute_cotangent(x1, x0, x3); // cot(x0-x1, x3-x1)
    float4 _q = make_float4(c03 + c04, c01 + c02, -c01 - c03, -c02 - c04);
    q[i] = _q * sqrtf(k);
    // Q[i] = Mat4::outer_product(q,q) * k;
}


// M. Wardetzky, M. Bergou, D. Harmon, D. Zorin, and E. Grinspun, "Discrete quadratic curvature energies" , Computer Aided Geometric Design, vol. 24, no. 8, pp. 499–518, Nov. 2007, doi: 10.1016/j.cagd.2007.07.006.
// Discrete Willmore Energy of isometric bending model (IBM), assume the edge lengths remain unchanged.
// It is valid only when the rest dihedral angle is straight angle.
static __global__ void compute_quadratic_bending_IBM(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    Mat3* __restrict__ Jx_bend_cross,
    float3* __restrict__ forces,
    float* __restrict__ energys,
    const float4* __restrict__ IBM_q,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const int3* __restrict__ triangles,
    const int2* __restrict__ edge_opposite_points,
    int num_edges, float kb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;
    int2 p_op = edge_opposite_points[i];
    if ( p_op.x == -1 || p_op.y == -1 ) return; // No need to calculate bending force at the boundary.
    int2 e_i = edges[i];

    int p0_idx = e_i.x, p1_idx = e_i.y;
    int p2_idx = p_op.x, p3_idx = p_op.y;
    float3 x0 = vertices[p0_idx], x1 = vertices[p1_idx], x2 = vertices[p2_idx], x3 = vertices[p3_idx];
    float4 q = IBM_q[i];
    float3 qtX = x0 * q.x + x1 * q.y + x2 * q.z + x3 * q.w;
    if ( energys ) {
        atomicAdd(&energys[p0_idx], 0.5f * kb * len_sq(qtX));
    }
    qtX = qtX * kb;

    if ( forces ) {
        atomicAddFloat3(&forces[p0_idx], -qtX * q.x);
        atomicAddFloat3(&forces[p1_idx], -qtX * q.y);
        atomicAddFloat3(&forces[p2_idx], -qtX * q.z);
        atomicAddFloat3(&forces[p3_idx], -qtX * q.w);
    }

    // These should be precomputed and are written here for completeness.
    if ( Jx_diag ) {
        atomicAddMat3(&Jx_diag[p0_idx], Mat3::identity(q.x * q.x * kb));
        atomicAddMat3(&Jx_diag[p1_idx], Mat3::identity(q.y * q.y * kb));
        atomicAddMat3(&Jx_diag[p2_idx], Mat3::identity(q.z * q.z * kb));
        atomicAddMat3(&Jx_diag[p3_idx], Mat3::identity(q.w * q.w * kb));
    }
    if ( Jx ) {
        auto [t1_i, t2_i] = e2t[i];
        auto t1 = triangles[t1_i];
        auto t2 = triangles[t2_i];
        auto f0d1 = Mat3::identity(q.x * q.y * kb);
        atomicAddMat3(&Jx[i], f0d1);

        auto f2d3 = Mat3::identity(q.z * q.w * kb);
        atomicAddMat3(&Jx_bend_cross[i], f2d3);

        auto f0d2 = Mat3::identity(q.x * q.z * kb);
        auto f1d2 = Mat3::identity(q.y * q.z * kb);
        auto f0d3 = Mat3::identity(q.x * q.w * kb);
        auto f1d3 = Mat3::identity(q.y * q.w * kb);
        if ( p0_idx > p2_idx ) {
            atomicAddMat3(&Jx[t1.x], f0d2);
            atomicAddMat3(&Jx[t1.y], f1d2);
        }
        else {
            atomicAddMat3(&Jx[t1.y], f0d2);
            atomicAddMat3(&Jx[t1.z], f1d2);
        }
        if ( p3_idx < p0_idx ) {
            atomicAddMat3(&Jx[t2.y], f0d3);
            atomicAddMat3(&Jx[t2.x], f1d3);
        }
        else {
            atomicAddMat3(&Jx[t2.x], f0d3);
            atomicAddMat3(&Jx[t2.z], f1d3);
        }
    }
}
