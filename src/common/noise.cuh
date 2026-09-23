#pragma once
#include <cuda_runtime.h>

#include "vec_math.h"

// Device-side value noise and curl noise.
//
// Header-only and device-only: include this from CUDA translation units, not
// from host-compiled files. The value noise is a hash lattice with smoothstep
// interpolation; `curl_noise` takes the curl of a three-channel potential so
// the result is divergence-free, which is what makes a procedural wind field
// read as eddies instead of as jitter.

static __device__ __forceinline__ float hash01(int x, int y, int z) {
    unsigned int h = (unsigned int)(x) * 374761393u
        + (unsigned int)(y) * 668265263u
        + (unsigned int)(z) * 1274126177u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (float)(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static __device__ __forceinline__ float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// Trilinear value noise in [0, 1).
static __device__ float value_noise3(float3 p) {
    const float fx = floorf(p.x), fy = floorf(p.y), fz = floorf(p.z);
    const int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    const float tx = p.x - fx, ty = p.y - fy, tz = p.z - fz;
    const float ux = tx * tx * (3.f - 2.f * tx);
    const float uy = ty * ty * (3.f - 2.f * ty);
    const float uz = tz * tz * (3.f - 2.f * tz);
    const float c000 = hash01(ix, iy, iz);
    const float c100 = hash01(ix + 1, iy, iz);
    const float c010 = hash01(ix, iy + 1, iz);
    const float c110 = hash01(ix + 1, iy + 1, iz);
    const float c001 = hash01(ix, iy, iz + 1);
    const float c101 = hash01(ix + 1, iy, iz + 1);
    const float c011 = hash01(ix, iy + 1, iz + 1);
    const float c111 = hash01(ix + 1, iy + 1, iz + 1);
    return lerpf(
        lerpf(lerpf(c000, c100, ux), lerpf(c010, c110, ux), uy),
        lerpf(lerpf(c001, c101, ux), lerpf(c011, c111, ux), uy),
        uz);
}

// 1D value noise in [0, 1), for a temporal signal such as a gust envelope.
static __device__ float value_noise1(float t) {
    const float f = floorf(t);
    const int i = (int)f;
    const float x = t - f;
    const float u = x * x * (3.f - 2.f * x);
    return lerpf(hash01(i, 0, 0), hash01(i + 1, 0, 0), u);
}

// Three decorrelated channels, used as the vector potential of the curl below.
static __device__ float3 noise_potential3(float3 p) {
    return make_float3(
        value_noise3(p),
        value_noise3(p + make_float3(37.1f, 17.7f, 91.3f)),
        value_noise3(p + make_float3(-13.9f, 71.5f, 29.1f)));
}

// Curl of the vector potential: divergence-free by construction. Central
// differences, so it costs six potential evaluations (18 value noise lookups).
static __device__ float3 curl_noise(float3 p, float eps) {
    const float3 ex = make_float3(eps, 0.f, 0.f);
    const float3 ey = make_float3(0.f, eps, 0.f);
    const float3 ez = make_float3(0.f, 0.f, eps);
    const float3 dx = (noise_potential3(p + ex) - noise_potential3(p - ex)) * (0.5f / eps);
    const float3 dy = (noise_potential3(p + ey) - noise_potential3(p - ey)) * (0.5f / eps);
    const float3 dz = (noise_potential3(p + ez) - noise_potential3(p - ez)) * (0.5f / eps);
    return make_float3(dz.y - dy.z, dx.z - dz.x, dy.x - dx.y);
}
