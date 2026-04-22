#pragma once

#include "vec_math.h"

static __device__ float point_triangle_sq_dist(float3 p, float3 a, float3 b, float3 c, float3* closest_pt) {
	float3 ab = b - a;
	float3 ac = c - a;
	float3 ap = p - a;

	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) {
		*closest_pt = a;
		return len_sq(p - a);
	} // Vertex A

	float3 bp = p - b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) {
		*closest_pt = b;
		return len_sq(p - b);
	} // Vertex B

	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		*closest_pt = a + ab * v;
		return len_sq(p - *closest_pt); // Edge AB
	}

	float3 cp = p - c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) {
		*closest_pt = c;
		return len_sq(p - c);
	} // Vertex C

	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		*closest_pt = a + ac * w;
		return len_sq(p - *closest_pt); // Edge AC
	}

	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
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
