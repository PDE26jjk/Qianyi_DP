#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "cuda_utils.h"

#define _FDH_ __forceinline__ __device__ __host__

_FDH_ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
_FDH_ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
_FDH_ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
_FDH_ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
_FDH_ float3 operator*(float b, const float3& a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
_FDH_ float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

_FDH_ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
_FDH_ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
_FDH_ float norm(const float3& a) {
    return sqrtf(dot(a, a));
}
_FDH_ float3 normalized(const float3& a) {
    float n = norm(a);
    return n > 1e-6f ? a / n : make_float3(0.0f, 0.0f, 0.0f);
}

_FDH_ float len_sq(const float3& a) { return dot(a, a); }

_FDH_ float3 fmin3(float3 a, float3 b) {
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
_FDH_ float3 fmax3(float3 a, float3 b) {
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

_FDH_ int2 operator+(const int2& a, const int2& b) {
    return make_int2(a.x + b.x, a.y + b.y);
}
_FDH_ int3 operator+(const int3& a, const int3& b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

_FDH_ float4 operator*(const float4& a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,a.w * b);
}

// 2x2 Matrix struct for Inverse Rest Pose
struct Mat2 {
    float2 r[2];

    __device__ __host__ Mat2() {}
    __device__ __host__ Mat2(float a, float b, float c, float d) {
        r[0] = make_float2(a, b);
        r[1] = make_float2(c, d);
    }

    _FDH_ Mat2 operator*(float s) const {
        return Mat2(r[0].x * s, r[0].y * s, r[1].x * s, r[1].y * s);
    }

    _FDH_ float2 operator*(float2 v) const {
        return make_float2(
            r[0].x * v.x + r[0].y * v.y,
            r[1].x * v.x + r[1].y * v.y
            );
    }

    _FDH_ Mat2 operator*(const Mat2& B) const {
        Mat2 C;
        C.r[0].x = r[0].x * B.r[0].x + r[0].y * B.r[1].x;
        C.r[0].y = r[0].x * B.r[0].y + r[0].y * B.r[1].y;
        C.r[1].x = r[1].x * B.r[0].x + r[1].y * B.r[1].x;
        C.r[1].y = r[1].x * B.r[0].y + r[1].y * B.r[1].y;
        return C;
    }

    _FDH_ float det() const {
        return r[0].x * r[1].y - r[0].y * r[1].x;
    }

    _FDH_ Mat2 inverse() const {
        float d = det();
        if ( fabs(d) < 1e-6f ) return Mat2(0, 0, 0, 0);
        float invD = 1.0f / d;
        return Mat2(r[1].y * invD, -r[0].y * invD,
            -r[1].x * invD, r[0].x * invD);
    }
};


struct Mat3 {
    float3 r[3]; // row storage

    static __forceinline__ __device__ __host__ Mat3 zero() {return identity(0);}
    __device__ __host__ Mat3() {}
    __device__ __host__ Mat3(float3 r0, float3 r1, float3 r2) {
        r[0] = r0;
        r[1] = r1;
        r[2] = r2;
    }

    __device__ __host__ Mat3 operator+(const Mat3& B) const {
        return {
            make_float3(r[0].x + B.r[0].x, r[0].y + B.r[0].y, r[0].z + B.r[0].z),
            make_float3(r[1].x + B.r[1].x, r[1].y + B.r[1].y, r[1].z + B.r[1].z),
            make_float3(r[2].x + B.r[2].x, r[2].y + B.r[2].y, r[2].z + B.r[2].z)
        };
    }

    __device__ __host__ Mat3 operator-(const Mat3& B) const {
        return {
            make_float3(r[0].x - B.r[0].x, r[0].y - B.r[0].y, r[0].z - B.r[0].z),
            make_float3(r[1].x - B.r[1].x, r[1].y - B.r[1].y, r[1].z - B.r[1].z),
            make_float3(r[2].x - B.r[2].x, r[2].y - B.r[2].y, r[2].z - B.r[2].z)
        };
    }

    __device__ __host__ Mat3 transpose() const {
        return {
            make_float3(r[0].x, r[1].x, r[2].x),
            make_float3(r[0].y, r[1].y, r[2].y),
            make_float3(r[0].z, r[1].z, r[2].z)
        };
    }

    // (Skew-symmetric matrix)
    __device__ __host__ static Mat3 cross_mat(float3 v) {
        return {
            make_float3(0.0f, -v.z, v.y),
            make_float3(v.z, 0.0f, -v.x),
            make_float3(-v.y, v.x, 0.0f)
        };
    }

    __device__ __host__ static Mat3 identity(const float a = 1.0f) {
        return {
            make_float3(a, 0.0f, 0.0f),
            make_float3(0.0f, a, 0.0f),
            make_float3(0.0f, 0.0f, a)
        };
    }

    __device__ __host__ Mat3 operator*(float s) const {
        return {
            make_float3(r[0].x * s, r[0].y * s, r[0].z * s),
            make_float3(r[1].x * s, r[1].y * s, r[1].z * s),
            make_float3(r[2].x * s, r[2].y * s, r[2].z * s)
        };
    }

    __device__ __host__ float3 operator*(float3 v) const {
        return make_float3(
            r[0].x * v.x + r[0].y * v.y + r[0].z * v.z,
            r[1].x * v.x + r[1].y * v.y + r[1].z * v.z,
            r[2].x * v.x + r[2].y * v.y + r[2].z * v.z
            );
    }

    __device__ __host__ Mat3 operator*(const Mat3& B) const {
        Mat3 C;
        for ( int i = 0; i < 3; i++ ) {
            C.r[i].x = r[i].x * B.r[0].x + r[i].y * B.r[1].x + r[i].z * B.r[2].x;
            C.r[i].y = r[i].x * B.r[0].y + r[i].y * B.r[1].y + r[i].z * B.r[2].y;
            C.r[i].z = r[i].x * B.r[0].z + r[i].y * B.r[1].z + r[i].z * B.r[2].z;
        }
        return C;
    }

    __device__ __host__ float det() const {
        return r[0].x * (r[1].y * r[2].z - r[1].z * r[2].y) -
            r[0].y * (r[1].x * r[2].z - r[1].z * r[2].x) +
            r[0].z * (r[1].x * r[2].y - r[1].y * r[2].x);
    }

    __device__ __host__ Mat3 inverse() const {
        float d = det();
        if ( fabs(d) < 1e-6f ) return Mat3::zero();
        float invD = 1.0f / d;

        Mat3 res;
        res.r[0].x = (r[1].y * r[2].z - r[1].z * r[2].y) * invD;
        res.r[0].y = (r[0].z * r[2].y - r[0].y * r[2].z) * invD;
        res.r[0].z = (r[0].y * r[1].z - r[0].z * r[1].y) * invD;

        res.r[1].x = (r[1].z * r[2].x - r[1].x * r[2].z) * invD;
        res.r[1].y = (r[0].x * r[2].z - r[0].z * r[2].x) * invD;
        res.r[1].z = (r[0].z * r[1].x - r[0].x * r[1].z) * invD;

        res.r[2].x = (r[1].x * r[2].y - r[1].y * r[2].x) * invD;
        res.r[2].y = (r[0].y * r[2].x - r[0].x * r[2].y) * invD;
        res.r[2].z = (r[0].x * r[1].y - r[0].y * r[1].x) * invD;

        return res;
    }
    __device__ __host__ static Mat3 outer_product(float3 a, float3 b) {
        return {
            make_float3(a.x * b.x, a.x * b.y, a.x * b.z),
            make_float3(a.y * b.x, a.y * b.y, a.y * b.z),
            make_float3(a.z * b.x, a.z * b.y, a.z * b.z)
        };
    }
    _FDH_  Mat3 operator-() const {
        return { -r[0], -r[1], -r[2] };
    }
};
__device__ __host__ static float3 operator*(float3 v, const Mat3& M) {
    return make_float3(
        v.x * M.r[0].x + v.y * M.r[1].x + v.z * M.r[2].x,
        v.x * M.r[0].y + v.y * M.r[1].y + v.z * M.r[2].y,
        v.x * M.r[0].z + v.y * M.r[1].z + v.z * M.r[2].z
        );
}

struct Mat4 {
    float4 r[4];

    __device__ __host__ Mat4() {}
    __device__ __host__ Mat4(float4 r0, float4 r1, float4 r2,float4 r3) {
        r[0] = r0;
        r[1] = r1;
        r[2] = r2;
        r[3] = r3;
    }

    __device__ __host__ float4 operator*(float4 v) const {
        return make_float4(
            r[0].x * v.x + r[0].y * v.y + r[0].z * v.z + r[0].w * v.w,
            r[1].x * v.x + r[1].y * v.y + r[1].z * v.z + r[1].w * v.w,
            r[2].x * v.x + r[2].y * v.y + r[2].z * v.z + r[2].w * v.w,
            r[3].x * v.x + r[3].y * v.y + r[3].z * v.z + r[3].w * v.w
            );
    }

    __device__ __host__ Mat4 operator*(float s) const {
        Mat4 m;
        m.r[0] = make_float4(r[0].x * s, r[0].y * s, r[0].z * s, r[0].w * s);
        m.r[1] = make_float4(r[1].x * s, r[1].y * s, r[1].z * s, r[1].w * s);
        m.r[2] = make_float4(r[2].x * s, r[2].y * s, r[2].z * s, r[2].w * s);
        m.r[3] = make_float4(r[3].x * s, r[3].y * s, r[3].z * s, r[3].w * s);
        return m;
    }

    __device__ __host__ Mat4 operator*(const Mat4& B) const {
        Mat4 C;
        for ( int i = 0; i < 4; i++ ) {
            C.r[i].x = r[i].x * B.r[0].x + r[i].y * B.r[1].x + r[i].z * B.r[2].x + r[i].w * B.r[3].x;
            C.r[i].y = r[i].x * B.r[0].y + r[i].y * B.r[1].y + r[i].z * B.r[2].y + r[i].w * B.r[3].y;
            C.r[i].z = r[i].x * B.r[0].z + r[i].y * B.r[1].z + r[i].z * B.r[2].z + r[i].w * B.r[3].z;
            C.r[i].w = r[i].x * B.r[0].w + r[i].y * B.r[1].w + r[i].z * B.r[2].w + r[i].w * B.r[3].w;
        }
        return C;
    }

    static __device__ __host__ __forceinline__ float det3(float a, float b, float c,
        float d, float e, float f,
        float g, float h, float i) {
        return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h;
    }

    __device__ __host__ Mat4 inverse() const {
        float m00 = r[0].x, m01 = r[0].y, m02 = r[0].z, m03 = r[0].w;
        float m10 = r[1].x, m11 = r[1].y, m12 = r[1].z, m13 = r[1].w;
        float m20 = r[2].x, m21 = r[2].y, m22 = r[2].z, m23 = r[2].w;
        float m30 = r[3].x, m31 = r[3].y, m32 = r[3].z, m33 = r[3].w;

        float v0 = det3(m11, m12, m13, m21, m22, m23, m31, m32, m33);
        float v1 = det3(m10, m12, m13, m20, m22, m23, m30, m32, m33);
        float v2 = det3(m10, m11, m13, m20, m21, m23, m30, m31, m33);
        float v3 = det3(m10, m11, m12, m20, m21, m22, m30, m31, m32);

        float det = m00 * v0 - m01 * v1 + m02 * v2 - m03 * v3;
        if ( fabs(det) < 1e-6f ) return Mat4();

        float invDet = 1.0f / det;
        Mat4 res;

        res.r[0].x = det3(m11, m12, m13, m21, m22, m23, m31, m32, m33) * invDet;
        res.r[0].y = -det3(m01, m02, m03, m21, m22, m23, m31, m32, m33) * invDet;
        res.r[0].z = det3(m01, m02, m03, m11, m12, m13, m31, m32, m33) * invDet;
        res.r[0].w = -det3(m01, m02, m03, m11, m12, m13, m21, m22, m23) * invDet;

        res.r[1].x = -det3(m10, m12, m13, m20, m22, m23, m30, m32, m33) * invDet;
        res.r[1].y = det3(m00, m02, m03, m20, m22, m23, m30, m32, m33) * invDet;
        res.r[1].z = -det3(m00, m02, m03, m10, m12, m13, m30, m32, m33) * invDet;
        res.r[1].w = det3(m00, m02, m03, m10, m12, m13, m20, m22, m23) * invDet;

        res.r[2].x = det3(m10, m11, m13, m20, m21, m23, m30, m31, m33) * invDet;
        res.r[2].y = -det3(m00, m01, m03, m20, m21, m23, m30, m31, m33) * invDet;
        res.r[2].z = det3(m00, m01, m03, m10, m11, m13, m30, m31, m33) * invDet;
        res.r[2].w = -det3(m00, m01, m03, m10, m11, m13, m20, m21, m23) * invDet;

        res.r[3].x = -det3(m10, m11, m12, m20, m21, m22, m30, m31, m32) * invDet;
        res.r[3].y = det3(m00, m01, m02, m20, m21, m22, m30, m31, m32) * invDet;
        res.r[3].z = -det3(m00, m01, m02, m10, m11, m12, m30, m31, m32) * invDet;
        res.r[3].w = det3(m00, m01, m02, m10, m11, m12, m20, m21, m22) * invDet;

        return res;
    }
    __device__ __host__ static Mat4 outer_product(float4 a, float4 b) {
        return {
            make_float4(a.x * b.x, a.x * b.y, a.x * b.z, a.x * b.w),
            make_float4(a.y * b.x, a.y * b.y, a.y * b.z, a.y * b.w),
            make_float4(a.z * b.x, a.z * b.y, a.z * b.z, a.z * b.w),
            make_float4(a.w * b.x, a.w * b.y,a.w * b.z,a.w * b.w)
        };
    }
};
static _FDH_
float3 mul_homo(const Mat4 m, const float3 v) {
    const float4 v_ = m * make_float4(v.x, v.y, v.z, 1.f);
    return make_float3(v_.x, v_.y, v_.z);
}

#undef _FDH_
