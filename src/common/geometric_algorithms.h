#pragma once

#include "vec_math.h"

// Point-triangle interaction in 3D (Ericson's algorithm)
static __device__ float dist_sq_point_triangle_3d(float3 p, float3 a, float3 b, float3 c) {
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if ( d1 <= 0.0f && d2 <= 0.0f ) return dot(ap, ap); // 顶点 A 最近

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if ( d3 >= 0.0f && d4 <= d3 ) return dot(bp, bp); // 顶点 B 最近

    float vc = d1 * d4 - d3 * d2;
    if ( vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f ) {
        float v = d1 / (d1 - d3);
        float3 closest = a + v * ab;
        float3 diff = p - closest;
        return dot(diff, diff); // 边 AB 最近
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if ( d6 >= 0.0f && d5 <= d6 ) return dot(cp, cp); // 顶点 C 最近

    float vb = d5 * d2 - d1 * d6;
    if ( vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f ) {
        float w = d2 / (d2 - d6);
        float3 closest = a + w * ac;
        float3 diff = p - closest;
        return dot(diff, diff); // 边 AC 最近
    }

    float va = d3 * d6 - d5 * d4;
    if ( va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f ) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        float3 bc = c - b;
        float3 closest = b + w * bc;
        float3 diff = p - closest;
        return dot(diff, diff); // 边 BC 最近
    }

    // 点在三角形内部或投影在三角形面上
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float3 closest = a + v * ab + w * ac;
    float3 diff = p - closest;
    return dot(diff, diff);
}
// __device__ __forceinline__ float dist_sq_segment_segment_3d(
//     float3 a, float3 b,
//     float3 c, float3 d
// ) {
//     float3 u = b - a;
//     float3 v = d - c;
//     float3 w = a - c;
//     float a_dot = dot(u, u);
//     float b_dot = dot(v, v);
//     float c_dot = dot(u, v);
//     float d_dot = dot(u, w);
//     float e_dot = dot(v, w);
//     float denom = a_dot * b_dot - c_dot * c_dot;
//     float s, t;
//     if ( denom < 1e-12f ) {
//         s = 0.0f;
//         t = d_dot / b_dot;
//     }
//     else {
//         s = (c_dot * e_dot - b_dot * d_dot) / denom;
//         t = (a_dot * e_dot - c_dot * d_dot) / denom;
//     }
//     s = fmaxf(0.0f, fminf(1.0f, s));
//     t = fmaxf(0.0f, fminf(1.0f, t));
//     float3 p = a + s * u;
//     float3 q = c + t * v;
//     return dot(p - q, p - q);
// }
__device__ __forceinline__ void segment_segment_closest_robust(
    float3 A, float3 B, float3 C, float3 D, float& sc, float& tc, float3& dP
) {
    float3 u = B - A;
    float3 v = D - C;
    float3 w = A - C;

    float a = dot(u, u);
    float b = dot(u, v);
    float c = dot(v, v);
    float d = dot(u, w);
    float e = dot(v, w);

    constexpr float EPS = 1e-10f;

    float sN, sD, tN, tD;

    if ( a < EPS && c < EPS ) {
        sc = 0.0f;
        tc = 0.0f;
    }
    else if ( a < EPS ) {
        sc = 0.0f;
        tN = e;
        tD = c;
        tc = fminf(fmaxf(tN / tD, 0.0f), 1.0f);
    }
    else if ( c < EPS ) {
        tc = 0.0f;
        sN = -d;
        sD = a;
        sc = fminf(fmaxf(sN / sD, 0.0f), 1.0f);
    }
    else {
        float D_denom = fmaf(a, c, -(b * b));
        sD = D_denom;
        tD = D_denom;

        if ( D_denom < EPS ) {
            sN = 0.0f;
            sD = 1.0f;
            tN = e;
            tD = c;
        }
        else {
            sN = fmaf(b, e, -(c * d));
            tN = fmaf(a, e, -(b * d));
            if ( sN < 0.0f ) {
                sN = 0.0f;
                tN = e;
                tD = c;
            }
            else if ( sN > sD ) {
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if ( tN < 0.0f ) {
            tN = 0.0f;
            sN = fminf(fmaxf(-d, 0.0f), a);
            sD = a;
        }
        else if ( tN > tD ) {
            tN = tD;
            sN = fminf(fmaxf(-d + b, 0.0f), a);
            sD = a;
        }

        sc = (fabsf(sN) < EPS) ? 0.0f : __fdividef(sN, sD);
        tc = (fabsf(tN) < EPS) ? 0.0f : __fdividef(tN, tD);
    }

    dP = w + (sc * u) - (tc * v);
}
__device__ __forceinline__ float segment_segment_dist_sq_robust(float3 A, float3 B, float3 C, float3 D) {
    float sc, tc;
    float3 dP;
    segment_segment_closest_robust(A, B, C, D, sc, tc, dP);

    return dot(dP, dP);
}
__device__ __forceinline__ bool edge_edge_closest_points(const float3& p1, const float3& p2, const float3& p3, const float3& p4,
    float3& closest_a, float3& closest_b,
    float& s,
    float& t) {
    float3 u = p2 - p1;
    float3 v = p4 - p3;
    float3 w = p1 - p3;
    float a_dot = dot(u, u);
    float b_dot = dot(v, v);
    float c_dot = dot(u, v);
    float d_dot = dot(u, w);
    float e_dot = dot(v, w);
    float denom = a_dot * b_dot - c_dot * c_dot;
    if ( denom < 1e-16f ) {
        s = t = 0.5f;
        return false;
    }
    s = (c_dot * e_dot - b_dot * d_dot) / denom;
    t = (a_dot * e_dot - c_dot * d_dot) / denom;
    closest_a = p1 + s * u;
    closest_b = p3 + t * v;
    return true;
}
static __device__ __forceinline__ float edge_edge_dist_sq(const float3& p1, const float3& p2, const float3& p3, const float3& p4,
    float3& closest_a, float3& closest_b,
    float& s,
    float& t) {
    if ( !edge_edge_closest_points(p1, p2, p3, p4, closest_a, closest_b, s, t) ) {
        return FLT_MAX;
    }
    return dot(closest_a - closest_b, closest_a - closest_b);
}

static __device__ float point_triangle_sq_dist(float3 p, float3 a, float3 b, float3 c, float3* closest_pt) {
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if ( d1 <= 0.0f && d2 <= 0.0f ) {
        *closest_pt = a;
        return len_sq(p - a);
    } // Vertex A

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if ( d3 >= 0.0f && d4 <= d3 ) {
        *closest_pt = b;
        return len_sq(p - b);
    } // Vertex B

    float vc = d1 * d4 - d3 * d2;
    if ( vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f ) {
        float v = d1 / (d1 - d3);
        *closest_pt = a + ab * v;
        return len_sq(p - *closest_pt); // Edge AB
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if ( d6 >= 0.0f && d5 <= d6 ) {
        *closest_pt = c;
        return len_sq(p - c);
    } // Vertex C

    float vb = d5 * d2 - d1 * d6;
    if ( vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f ) {
        float w = d2 / (d2 - d6);
        *closest_pt = a + ac * w;
        return len_sq(p - *closest_pt); // Edge AC
    }

    float va = d3 * d6 - d5 * d4;
    if ( va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f ) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *closest_pt = b + (c - b) * w;
        return len_sq(p - *closest_pt); // Edge BC
    }

    // Inside Face
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    *closest_pt = a + ab * v + ac * w;
    return len_sq(p - *closest_pt);
}
static __device__ __forceinline__ int get_hash(int3 p, int table_size) {
    long long h = (p.x * 73856093LL) ^ (p.y * 19349663LL) ^ (p.z * 83492791LL);
    // Use unsigned modulo arithmetic
    return (int)(abs(h) % table_size);
}

static void __device__ barycentric(const float3& A, const float3& B, const float3& C,
    const float3& P, float& u, float& v, float& w) {
    float3 v0 = B - A;
    float3 v1 = C - A;
    float3 v2 = P - A;

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    if ( fabs(denom) < 1e-10 ) {
        u = v = w = -1.0;
        return;
    }

    u = (d11 * d20 - d01 * d21) / denom;
    v = (d00 * d21 - d01 * d20) / denom;
    w = 1.0f - u - v;
}
