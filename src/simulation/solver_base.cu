#include "solver_base.cuh"

#include <thrust/execution_policy.h>
// #include <thrust/host_vector.h>


// #include <filesystem>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

// #include "constraint.cuh"
#include "geometric_operator.cuh"
#include "common/cuda_utils.h"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"


void SolverBase::init() {
    auto& params = *simulator->get_geo_params();

    // init single device pool
    pool.resize(1024);
    pool_used.assign(1024, false);

}


static __device__ int get_opposite_point(int2 edge, int3 tri, const int2* edges) {
    int v0 = edges[tri.x].x, v1 = edges[tri.x].y, v2 = edges[tri.y].y;
    if ( v0 != edge.x && v0 != edge.y ) return v0;
    if ( v1 != edge.x && v1 != edge.y ) return v1;
    return v2;
}

// static __device__ __forceinline__
// float3 mul_homo_vec(const Mat4 m, const float3 v) {
// 	const float4 v_ = m * make_float4(v.x, v.y, v.z, 0.f);
// 	return make_float3(v_.x, v_.y, v_.z);
// }


AutoGPUmem SolverBase::alloc_pool() {
    for ( size_t i = 0; i < pool_used.size(); ++i )
        if ( !pool_used[i] ) {
            pool_used[i] = true;
            return AutoGPUmem{ this,
                thrust::raw_pointer_cast(&pool[i]) };
        }
    throw std::exception("No pool available");
    // pool.resize(pool.size() * 2);
    // pool_used.resize(pool.size(), false);
    // pool_used[pool.size() / 2] = true;
    // return AutoGPUmem{ this,
    //     thrust::raw_pointer_cast(&pool[pool.size() / 2]) };
}
void SolverBase::dealloc_pool(void* p) {
    size_t idx = reinterpret_cast<int*>(p) - thrust::raw_pointer_cast(&pool[0]);
    if ( idx < pool.size() ) pool_used[idx] = false;
}

AutoGPUmem::~AutoGPUmem() { pool->dealloc_pool(ptr); }


float SolverBase::get_global_parameter(const std::string& key, float default_value) const {
    return simulator->get_parameter(key, default_value);
}
