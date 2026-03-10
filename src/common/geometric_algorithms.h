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
