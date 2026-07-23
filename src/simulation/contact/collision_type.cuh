#pragma once
// Deprecated, just a backup
#include "common/cuda_utils.h"

#define MAX_POINTS_PER_CELL 7
struct PointHashCell {
    int count;
    int point_indices[MAX_POINTS_PER_CELL];
};

struct CollisionResult_TP {
    int tri_idx;
    float3 weights;
    int vert_idx;
    float3 normal;
    // float3 closest_pt;
};
struct CollisionResult_EE {
    int edgeA_idx;
    int edgeB_idx;
    float2 coefs;
    float3 normal;
};

struct CollisionResult_PP {
    int p1;
    int p2;
    CollisionResult_PP() = default;
    __device__ __host__ CollisionResult_PP(int vert_idx, int p_idx): p1(vert_idx), p2(p_idx) {
    }
};
struct CollisionConstraint {
    // char type;// 
    int p0, p1, p2, p3;
    int testing_color;
};

struct UnifiedNormalConstraint {
    int v[4];         // 参与约束的 4 个顶点索引
    float w[4];       // 这 4 个顶点对应的雅可比权重
    float3 normal;    // 碰撞法线 (统一指向分离方向)
    float A_ii;       // 预计算的有效质量: sum(w[i]^2 * invMass[v[i]])
    float lambda;     // 累积冲量
    int color;
    Mat3 H;
};

static __device__ __forceinline__ int get_hash(int3 p, int table_size) {
    long long h = (p.x * 73856093LL) ^ (p.y * 19349663LL) ^ (p.z * 83492791LL);
    // Use unsigned modulo arithmetic
    return (int)(abs(h) % table_size);
}
